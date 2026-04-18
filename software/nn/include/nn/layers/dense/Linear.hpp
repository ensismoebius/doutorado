#ifndef NN_LAYERS_LINEAR_HPP
#define NN_LAYERS_LINEAR_HPP

#include "nn/layers/base/Module.hpp"
#include "nn/tensor/Tensor.hpp"

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
 * - For non-Eigen backends the parameters are converted to the active backend at the start
 *   of each forward/backward pass; this conversion is zero-cost when Backend == Eigen since
 *   both types alias the same underlying storage.
 * - Concrete aliases (e.g. `Linear = LinearImpl<EigenBackend>`) live in
 *   `nn/layers/eigen/Layers.hpp`.
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
     * For the Eigen backend this conversion is a same-type copy (zero semantic
     * overhead); for other backends it performs an explicit transfer.
     *
     * @param input  [batch × in_features]
     * @param requires_grad  Cache input for backward when true.
     * @return [batch × out_features]
     */
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        if (static_cast<int>(input.cols()) != in_features)
        {
            throw std::invalid_argument(
                "Linear layer forward: input features (" + std::to_string(input.cols()) +
                ") do not match expected in_features (" + std::to_string(in_features) + ")");
        }

        if (requires_grad)
        {
            // Store a CPU copy so the CPU optimizer can compute gradients.
            input_cache = nn::Tensor(input);
        }

#ifdef DEBUG
        debug(nn::Tensor(input));
#endif

        // Convert CPU parameters to the active backend, then compute y = x W^T + b.
        Tensor weight_t(weight);
        Tensor bias_t(bias);
        Tensor result = input.matmul(weight_t.transpose());
        result.add_col_vector_to_rows_inplace(bias_t);
        return result;
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
        if (grad_previous.cols() != static_cast<size_t>(out_features))
        {
            throw std::invalid_argument("Linear layer backward: gradient features (" +
                                        std::to_string(grad_previous.cols()) +
                                        ") do not match expected out_features (" +
                                        std::to_string(out_features) + ")");
        }

        // Lift CPU cache and parameters into the active backend, compute gradients,
        // then download them back to CPU tensors for the optimizer.
        Tensor input_t(input_cache);
        Tensor weight_t(weight);
        // dL/dW = (dL/dY)^T · X
        Tensor grad_weight = grad_previous.transpose().matmul(input_t);
        weight.set_grad(nn::Tensor(grad_weight));
        // dL/db = sum_rows((dL/dY)^T), shape: (out_features, 1)
        Tensor grad_bias = grad_previous.transpose().rowwise_sum();
        bias.set_grad(nn::Tensor(grad_bias));
        // dL/dX = dL/dY · W
        return grad_previous.matmul(weight_t);
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
