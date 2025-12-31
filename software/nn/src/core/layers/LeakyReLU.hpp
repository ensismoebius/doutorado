#ifndef LEAKYRELU_HPP
#define LEAKYRELU_HPP

#include "../tensor/Tensor.hpp"
#include "core/layers/Module.hpp"

struct LeakyReLU : public Module
{
    float alpha; // negative slope
    nn::Tensor leaky_grad;

    explicit LeakyReLU(float alpha_ = 0.01F) : alpha(alpha_) {}

    auto forward(const nn::Tensor& input) -> nn::Tensor override
    {
        // Cache the gradient for the backward pass
        // Create gradient mask: 1 for positive values, alpha for negative values
        Eigen::MatrixXf grad_mask =
            (input.get_data_ref().array() > 0)
                .select(
                    Eigen::MatrixXf::Ones(input.get_data_ref().rows(), input.get_data_ref().cols()),
                    Eigen::MatrixXf::Constant(
                        input.get_data_ref().rows(), input.get_data_ref().cols(), alpha));
        leaky_grad = nn::Tensor(grad_mask);

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
