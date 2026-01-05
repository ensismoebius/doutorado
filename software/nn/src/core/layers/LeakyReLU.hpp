#ifndef LEAKYRELU_HPP
#define LEAKYRELU_HPP

#include "nn/tensor/Tensor.hpp"
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
            leaky_grad = nn::Tensor(input.rows(), input.cols());
            for (size_t i = 0; i < input.rows(); ++i)
            {
                for (size_t j = 0; j < input.cols(); ++j)
                {
                    leaky_grad.at(i, j) = (input.at(i, j) > 0.0f) ? 1.0f : alpha;
                }
            }
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
