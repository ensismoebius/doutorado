#ifndef LEAKYRELU_HPP
#define LEAKYRELU_HPP

#include <Eigen/Dense>

#include "../tensor/Tensor.hpp"
#include "core/layers/Module.hpp"

struct LeakyReLU : public Module
{
    float alpha; // negative slope
    Eigen::MatrixXf leaky_grad;

    LeakyReLU(float alpha_ = 0.01F) : alpha(alpha_) {}

    auto forward(const Tensor& input) -> Tensor override
    {
        // Cache the gradient for the backward pass
        leaky_grad =
            (input.data.array() > 0)       // For each element, check if it's greater than 0
                .select(                   // Select values based on the condition
                    Eigen::MatrixXf::Ones( // If the element is greater than 0, the gradient is 1
                        input.data.rows(), //
                        input.data.cols()  //
                        ),                 //
                    Eigen::MatrixXf::Constant( // Otherwise, the gradient is alpha
                        input.data.rows(),     //
                        input.data.cols(),     //
                        alpha                  //
                        )                      //
                );

        // Apply the LeakyReLU activation function
        Eigen::MatrixXf activated =
            (input.data.array() > 0) // For each element, check if it's greater than 0
                .select(             // Select values based on the condition
                    input.data,      // If the element is greater than 0, it remains unchanged
                    input.data.array() * alpha // Otherwise, it's scaled by alpha
                );

        return {activated};
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        Eigen::MatrixXf grad_input = grad_output.data.array() * leaky_grad.array();
        return {grad_input};
    }
};

#endif // LEAKYRELU_HPP
