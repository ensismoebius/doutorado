#ifndef LEAKYRELU_HPP
#define LEAKYRELU_HPP

#include "../tensor/Tensor.hpp"
#include "core/layers/Module.hpp"

struct LeakyReLU : public Module
{
    float alpha; // negative slope
    nn::Tensor leaky_grad;

    explicit LeakyReLU(float alpha_ = 0.01F) : alpha(alpha_) {}

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        // Cache the gradient for the backward pass only if gradients required
        if (requires_grad)
        {
            // Create gradient mask: 1 for positive values, alpha for negative values
            leaky_grad = nn::Tensor(input.get_data_ref().rows(), input.get_data_ref().cols());
            const auto& input_ref = input.get_data_ref();
            leaky_grad.get_data_ref() =
                input_ref.array()
                    .unaryExpr([this](float x) { return (x > 0.0F) ? 1.0F : alpha; })
                    .matrix();
        }

        // Apply the LeakyReLU activation function using Tensor method
        return input.leaky_relu(alpha);
    }

    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override
    {
        // Element-wise multiplication of grad_output with leaky_grad mask
        auto grad_input = grad_output.multiply(leaky_grad);
        return grad_input;
    }
};

#endif // LEAKYRELU_HPP
