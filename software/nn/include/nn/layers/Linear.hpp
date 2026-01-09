#ifndef LINEAR_CPP
#define LINEAR_CPP

#include "nn/layers/Module.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @brief Camada Linear (ou camada totalmente conectada)
 * Implementa uma camada linear que aplica a transformação afim:
 * saída = entrada * W^T + b
 *
 * Onde:
 * - entrada é o tensor de entrada
 * - weight é a matriz de pesos
 * - bias é o vetor de bias
 */
struct Linear : public Module
{
    int in_features;        // número de entradas (features de entrada do tensor)
    int out_features;       // número de saídas (neurônios ou unidades na camada)
    nn::Tensor weight;      // matriz de pesos com dimensão [out_features x in_features]
    nn::Tensor bias;        // vetor de bias com dimensão [out_features]
    nn::Tensor input_cache; // armazena a entrada da camada para uso no backpropagation

    /**
     * @brief Inicializa pesos e bias com base no número de entradas e saídas
     *
     * @param in_features Número de entradas
     * @param out_features Número de saídas
     */
    Linear(const int in_features, const int out_features)
        : in_features(in_features),
          out_features(out_features),
          weight(nn::Tensor(out_features, in_features)),
          bias(nn::Tensor(out_features, 1))
    {
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

        // Cache input for backward pass only if gradients are required
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

        // Manual broadcasting of bias
        for (size_t i = 0; i < result.rows(); ++i)
        {
            for (size_t j = 0; j < result.cols(); ++j)
            {
                result.at(i, j) += bias.at(j, 0);
            }
        }

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
        if (grad_previous.cols() != out_features)
        {
            throw std::invalid_argument("Linear layer backward: gradient features (" +
                                        std::to_string(grad_previous.cols()) +
                                        ") do not match expected out_features (" +
                                        std::to_string(out_features) + ")");
        }

        // Calculate weight gradient: grad_weight = grad_previous.T * input_cache
        nn::Tensor grad_weight = grad_previous.transpose().matmul(input_cache);

        weight.set_grad(grad_weight);

        // Calculate bias gradient: sum of grad_previous columns
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

        // Calculate input gradient: grad_input = grad_previous * weight
        auto grad_input_tensor = grad_previous.matmul(weight);
        return grad_input_tensor;
    }

    /**
     * @brief Returns trainable parameters
     */
    auto params() -> std::vector<nn::Tensor*> override
    {
        return {&weight, &bias};
    }
};

#endif