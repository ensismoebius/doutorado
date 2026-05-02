#ifndef NN_LAYERS_LINEAR_HPP
#define NN_LAYERS_LINEAR_HPP

#include <optional>
#include <type_traits>

#include "nn/Backend.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/opencl/OpenCLContext.hpp"

/**
 * @file Linear.hpp
 * @brief Fully-connected (affine) layer: y = x W^T + b.
 *
 * How it fits in the system:
 * - Linear layers implement the learnable synaptic connections.
 * - Spiking layers (`LeakyBPTT`, `Leaky`, etc.) provide the nonlinearity and temporal dynamics.
 * - Most networks in this repo alternate: Linear → (spiking/nonlinear) → Linear → ...
 *
 * Shape convention used here:
 * - Input:  (batch, in_features)
 * - Weight: (out_features, in_features)
 * - Bias:   (out_features, 1) and is broadcast across the batch.
 * - Output: (batch, out_features)
 *
 * Backend polymorphism:
 * - `LinearImpl<Backend>` works with any backend tensor type.
 * - Parameters (`weight`, `bias`) are always stored as CPU-resident `nn::Tensor` so that
 *   existing CPU optimizers continue to work without modification.
 * - For non-xtensor backends the parameters are converted to the active backend at the start
 *   of each forward/backward pass; this conversion is zero-cost when Backend == XTensorBackend
 * since both types alias the same underlying storage.
 * - Concrete aliases (e.g. `Linear = LinearImpl<Backend>`) live in
 *   `nn/layers/Layers.hpp`.
 */
template <typename Backend>
struct LinearImpl : public Module<Backend>
{
    /// Tensor type for the active compute backend.
    using Tensor = nn::TensorImpl<Backend>;

    int in_features;  // number of input features
    int out_features; // number of output neurons
    /// Weight matrix [out_features × in_features] — always CPU-resident for optimizer compat.
    nn::Tensor weight;
    /// Bias vector [out_features × 1] — always CPU-resident for optimizer compat.
    nn::Tensor bias;
    /// CPU-side cached input; populated when `forward(..., requires_grad=true)`.
    nn::Tensor input_cache;
    /// Backend-side cached input for avoiding CPU->backend re-conversion in backward.
    std::optional<Tensor> input_cache_backend;
    // Owned view of parameter pointers. Must point to member tensors so the span
    // returned by `params()` remains valid for the lifetime of this object.
    std::array<nn::Tensor*, 2> param_ptrs_{{&weight, &bias}};

    /**
     * @brief Construct the layer and allocate uninitialized weight/bias storage.
     *
     * @param in_features_  Number of input features.
     * @param out_features_ Number of output neurons.
     *
     * Leave parameter initialization to dedicated initializers (xavier/kaiming)
     * so callers can provide deterministic seeds and sampler policy.
     */
    LinearImpl(const int in_features_, const int out_features_)
        : in_features(in_features_),
          out_features(out_features_),
          weight(nn::Tensor(out_features_, in_features_)),
          bias(nn::Tensor(out_features_, 1))
    {
    }

#ifdef DEBUG
    auto debug(const nn::Tensor& input) -> void
    {
        std::ostringstream oss;
        oss << "Linear layer forward:" << "\n"
            << "Input dims: " << input.rows() << "x" << input.cols() << "\n"
            << "Weight dims: " << weight.rows() << "x" << weight.cols() << "\n"
            << "Bias dims: " << bias.rows() << "x" << bias.cols() << "\n";
        NN_LOG_INFO(oss.str());
    }
#endif

    /**
     * @brief Forward pass: y = x W^T + b.
     *
     * CPU-resident parameters are converted to the active backend on each call.
     * For the xtensor backend this conversion is a same-type copy (zero semantic
     * overhead); for other backends it performs an explicit transfer.
     *
     * @param input  [batch × in_features]
     * @param requires_grad  Cache input for backward when true.
     * @return [batch × out_features]
     */
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        std::optional<nn::opencl::OpenCLContext::BatchScope> batch_scope;
        if constexpr (std::is_same_v<Backend, nn::OpenCLTensorBackend>)
        {
            batch_scope.emplace();
        }

        const auto shape = input.get_shape();
        if (shape.empty()) throw std::invalid_argument("Input tensor cannot be empty");

        const nn::Index in_dim = shape.back();
        if (static_cast<int>(in_dim) != in_features)
        {
            throw std::invalid_argument(
                "Linear layer forward: input features (" + std::to_string(in_dim) +
                ") do not match expected in_features (" + std::to_string(in_features) + ")");
        }

        if (requires_grad) input_cache = nn::Tensor(input);

        // Handle N-D inputs by flattening leading dimensions into a single batch dimension.
        // Input: (d0, d1, ..., in_features) -> Reshape: (B_eff, in_features)
        std::vector<nn::Index> flat_shape = {1, (nn::Index) in_features};
        if (shape.size() > 1)
        {
            nn::Index effective_batch = 1;
            for (size_t i = 0; i < shape.size() - 1; ++i) effective_batch *= shape[i];
            flat_shape[0] = effective_batch;
        }
        else
        {
            // 1D input (in_features,) is treated as a single sample (1, in_features)
            flat_shape[0] = 1;
        }

        Tensor input_flat = input.reshape(flat_shape);
        if (requires_grad)
        {
            if constexpr (std::is_same_v<Backend, nn::OpenCLTensorBackend>)
            {
                input_cache_backend = input_flat;
            }
        }

        // Convert CPU parameters only when the active backend differs from the default backend.
        Tensor result_flat;
        if constexpr (std::is_same_v<Backend, nn::Backend>)
        {
            result_flat = Tensor(input_flat.get_backend().matmul_transposed(weight.get_backend()));
            result_flat.get_backend().add_col_vector_to_rows_inplace(bias.get_backend());
        }
        else if constexpr (std::is_same_v<Backend, nn::OpenCLTensorBackend>)
        {
            Tensor weight_t(weight);
            Tensor bias_t(bias);
            input_flat.get_backend().set_gpu_resident(true);
            weight_t.get_backend().set_gpu_resident(true);
            bias_t.get_backend().set_gpu_resident(true);
            result_flat = Tensor(input_flat.get_backend().matmul_transposed_add_col_bias(
                weight_t.get_backend(), bias_t.get_backend()));
        }
        else
        {
            Tensor weight_t(weight);
            Tensor bias_t(bias);
            result_flat = input_flat.matmul_transposed(weight_t);
            result_flat.add_col_vector_to_rows_inplace(bias_t);
        }

        // Restore original leading dimensions: (d0, d1, ..., out_features)
        std::vector<nn::Index> out_shape = shape;
        out_shape.back() = (nn::Index) out_features;

        // If input was 1D, result should be 2D (1, out_features) to maintain batch consistency
        if (shape.size() == 1)
        {
            return result_flat;
        }

        result_flat.reshape(out_shape);
        return result_flat;
    }

    /**
     * @brief Backward pass: compute parameter gradients and return the input gradient.
     *
     * Parameter gradients are stored as CPU `nn::Tensor` (via `set_grad`) so that
     * CPU optimizers can consume them without backend awareness.
     *
     * @param grad_previous  [batch × out_features] gradient of the loss w.r.t. output.
     * @return [batch × in_features] gradient of the loss w.r.t. input.
     */
    auto backward(const Tensor& grad_previous) -> Tensor override
    {
        std::optional<nn::opencl::OpenCLContext::BatchScope> batch_scope;
        if constexpr (std::is_same_v<Backend, nn::OpenCLTensorBackend>)
        {
            batch_scope.emplace();
        }

        const auto shape = grad_previous.get_shape();
        if (shape.empty()) throw std::invalid_argument("Gradient tensor cannot be empty");

        const nn::Index out_dim = shape.back();
        if (static_cast<int>(out_dim) != out_features)
        {
            throw std::invalid_argument(
                "Linear layer backward: gradient features (" + std::to_string(out_dim) +
                ") do not match expected out_features (" + std::to_string(out_features) + ")");
        }

        // Flatten leading dimensions of grad_previous: (B_eff, out_features)
        std::vector<nn::Index> flat_grad_shape = {1, (nn::Index) out_features};
        if (shape.size() > 1)
        {
            nn::Index effective_batch = 1;
            for (size_t i = 0; i < shape.size() - 1; ++i) effective_batch *= shape[i];
            flat_grad_shape[0] = effective_batch;
        }
        else
        {
            flat_grad_shape[0] = 1;
        }
        Tensor grad_flat = grad_previous.reshape(flat_grad_shape);

        // Flatten cached input: (B_eff, in_features)
        const auto input_shape = input_cache.get_shape();
        std::vector<nn::Index> flat_input_shape = {1, (nn::Index) in_features};
        if (input_shape.size() > 1)
        {
            nn::Index effective_batch = 1;
            for (size_t i = 0; i < input_shape.size() - 1; ++i) effective_batch *= input_shape[i];
            flat_input_shape[0] = effective_batch;
        }
        else
        {
            flat_input_shape[0] = 1;
        }
        Tensor input_t;
        if constexpr (std::is_same_v<Backend, nn::OpenCLTensorBackend>)
        {
            if (input_cache_backend.has_value())
            {
                input_t = *input_cache_backend;
            }
            else
            {
                input_t = Tensor(input_cache);
                input_t.reshape(flat_input_shape);
            }
        }
        else
        {
            input_t = Tensor(input_cache);
            input_t.reshape(flat_input_shape);
        }

        Tensor grad_weight;
        Tensor grad_t = grad_flat.transpose();
        if constexpr (std::is_same_v<Backend, nn::OpenCLTensorBackend>)
        {
            grad_flat.get_backend().set_gpu_resident(true);
            input_t.get_backend().set_gpu_resident(true);
            grad_weight =
                Tensor(grad_flat.get_backend().matmul_lhs_transposed(input_t.get_backend()));
            grad_t.get_backend().set_gpu_resident(true);
        }
        else
        {
            // dL/dW = (dL/dY)^T · X
            grad_weight = grad_t.matmul(input_t);
        }
        weight.set_grad(nn::Tensor(grad_weight));

        // dL/db = sum_rows((dL/dY)^T), shape: (out_features, 1)
        Tensor grad_bias = grad_t.rowwise_sum();
        bias.set_grad(nn::Tensor(grad_bias));

        // dL/dX = dL/dY · W
        Tensor grad_input_flat;
        if constexpr (std::is_same_v<Backend, nn::Backend>)
        {
            grad_input_flat = Tensor(grad_flat.get_backend().matmul(weight.get_backend()));
        }
        else if constexpr (std::is_same_v<Backend, nn::OpenCLTensorBackend>)
        {
            Tensor weight_t(weight);
            weight_t.get_backend().set_gpu_resident(true);
            grad_input_flat = grad_flat.matmul(weight_t);
        }
        else
        {
            grad_input_flat = grad_flat.matmul(Tensor(weight));
        }

        // Restore original leading dimensions for the input gradient
        std::vector<nn::Index> input_out_shape = input_shape;
        // No change to last dim because it's already in_features

        if (input_shape.size() == 1)
        {
            return grad_input_flat;
        }

        grad_input_flat.reshape(input_out_shape);
        return grad_input_flat;
    }

    /// Returns the two CPU-resident trainable parameters (weight, bias).
    auto params() -> std::span<nn::Tensor*> override
    {
        return std::span<nn::Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    auto state_dict() const -> std::map<std::string, nn::Tensor> override
    {
        std::map<std::string, nn::Tensor> d;
        d["weight"] = weight;
        d["bias"] = bias;
        return d;
    }

    void load_state_dict(const std::map<std::string, nn::Tensor>& sd) override
    {
        auto itw = sd.find("weight");
        if (itw != sd.end())
        {
            weight = itw->second;
        }
        auto itb = sd.find("bias");
        if (itb != sd.end())
        {
            bias = itb->second;
        }
    }
};

#endif // NN_LAYERS_LINEAR_HPP
