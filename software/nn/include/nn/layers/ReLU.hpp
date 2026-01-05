#ifndef RELU_HPP
#define RELU_HPP

#include <cstddef>

#include "nn/layers/Module.hpp"
#include "nn/tensor/Tensor.hpp"

struct ReLU : public Module
{
    nn::Tensor relu_grad; // usado para backward

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        // Guarda o gradiente da entrada atual para usar na fase de backward only if gradients
        // required
        if (requires_grad)
        {
            // relu_grad stores which elements were > 0 (for gradient computation)
            // Create a tensor and populate with mask values
            relu_grad = nn::Tensor(input.rows(), input.cols());
            for (size_t i = 0; i < input.rows(); ++i)
            {
                for (size_t j = 0; j < input.cols(); ++j)
                {
                    relu_grad.at(i, j) = (input.at(i, j) > 0.0f) ? 1.0f : 0.0f;
                }
            }
        }

        // Calcula a ativação usando Tensor method
        return input.relu();
    }

    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override
    {
        // Element-wise multiplication of grad_output with relu_grad mask
        auto grad_input = grad_output.multiply(relu_grad);
        return grad_input;
    }
};

#endif // RELU_HPP