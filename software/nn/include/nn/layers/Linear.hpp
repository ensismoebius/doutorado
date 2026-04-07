#ifndef LINEAR_CPP
#define LINEAR_CPP

#include "nn/layers/Module.hpp"
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
 */
struct Linear : public Module
{
    int in_features;   // número de entradas (features de entrada do tensor)
    int out_features;  // número de saídas (neurônios ou unidades na camada)
    nn::Tensor weight; // matriz de pesos com dimensão [out_features x in_features]
    nn::Tensor bias;   // vetor de bias com dimensão [out_features]
    // Cached input needed to compute gradients during backward.
    // Only populated when `forward(..., requires_grad=true)`.
    nn::Tensor input_cache;
    // Owned view of parameter pointers. Must point to member tensors so the span
    // returned by `params()` remains valid for the lifetime of this object.
    std::array<nn::Tensor*, 2> param_ptrs_{{&weight, &bias}};

    /**
     * @brief Inicializa pesos e bias com base no número de entradas e saídas
     *
     * @param in_features Número de entradas
     * @param out_features Número de saídas
     */
    Linear(const int in_features_, const int out_features_)
        : in_features(in_features_),
          out_features(out_features_),
          weight(nn::Tensor(out_features_, in_features_)),
          bias(nn::Tensor(out_features_, 1))
    {
        // Leave parameter initialization to dedicated initializers (xavier/kaiming)
        // so callers can provide deterministic seeds and sampler policy.
    }

#ifdef DEBUG
    // If DEBUG is defined then show the debug information
    auto debug(const nn::Tensor& input) -> void
    {
        std::cout << "Linear layer forward:" << "\n";
        std::cout << "Input dims: " << input.rows() << "x" << input.cols() << "\n";
        std::cout << "Weight dims: " << weight.rows() << "x" << weight.cols() << "\n";
        std::cout << "Bias dims: " << bias.rows() << "x" << bias.cols() << "\n";
    }
#endif

    /**
     * @brief Realiza a propagação direta (forward pass) da
     * entrada pela camada linear.
     * Aplica a transformação afim: saída = entrada * W^T + b
     * @param input Tensor de saída [batch_size x out_features]
     * @return Tensor de saída [batch_size x out_features]
     */
    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        // Validate input dimensions
        if (static_cast<int>(input.cols()) != in_features)
        {
            throw std::invalid_argument(
                "Linear layer forward: input features (" + std::to_string(input.cols()) +
                ") do not match expected in_features (" + std::to_string(in_features) + ")");
        }

        // Cache input for backward pass only if gradients are required.
        // This is the standard memory/performance trade-off used in autodiff systems.
        if (requires_grad)
        {
            input_cache = input; // salva para o backward
        }

#ifdef DEBUG
        // If DEBUG is defined then show the debug information
        debug(input);
#endif

        // Optimized linear transformation: y = x * W^T + b
        nn::Tensor result = input.matmul(weight.transpose());

        // Vectorized bias add: broadcast (out_features,1) bias across all batch rows.
        result.add_col_vector_to_rows_inplace(bias);

        return result;
    }

    /**
     * @brief Realiza a propagação reversa (backward pass) da derivada da perda
     * em relação à saída da camada.Calcula os gradientes da perda em relação aos
     * pesos, bias e à entrada da camada
     *
     * @param grad_previous gradiente da perda em relação à saída da camada [batch_size x
     * out_features]
     * @return gradiente da perda em relação à entrada da camada [batch_size x in_features]
     */
    auto backward(const nn::Tensor& grad_previous) -> nn::Tensor override
    {
        // Validate gradient dimensions
        if (grad_previous.cols() != static_cast<size_t>(out_features))
        {
            throw std::invalid_argument("Linear layer backward: gradient features (" +
                                        std::to_string(grad_previous.cols()) +
                                        ") do not match expected out_features (" +
                                        std::to_string(out_features) + ")");
        }

        // Weight gradient (matrix calculus): dL/dW = (dL/dY)^T · X
        // where:
        //   X = input_cache (batch, in_features)
        //   dL/dY = grad_previous (batch, out_features)
        nn::Tensor grad_weight = grad_previous.transpose().matmul(input_cache);

        weight.set_grad(grad_weight);

        // Bias gradient: sum over the batch dimension.
        // dL/db[j] = sum_i dL/dY[i,j]
        nn::Tensor grad_bias(out_features, 1);
        for (size_t j = 0; j < grad_previous.cols(); ++j)
        {
            float sum = 0.0f;
            for (size_t i = 0; i < grad_previous.rows(); ++i)
            {
                sum += grad_previous.at(i, j);
            }
            grad_bias.at(j, 0) = sum;
        }
        bias.set_grad(grad_bias);

        // Input gradient: dL/dX = dL/dY · W
        auto grad_input_tensor = grad_previous.matmul(weight);
        return grad_input_tensor;
    }

    /**
     * @brief Returns trainable parameters
     */
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

#endif