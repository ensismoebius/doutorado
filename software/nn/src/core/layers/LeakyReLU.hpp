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
            (input.get_data_ref().array() > 0)       // For each element, check if it's greater than 0
                .select(                   // Select values based on the condition
                    Eigen::MatrixXf::Ones( // If the element is greater than 0, the gradient is 1
                        input.get_data_ref().rows(), //
                        input.get_data_ref().cols()  //
                        ),                 //
                    Eigen::MatrixXf::Constant( // Otherwise, the gradient is alpha
                        input.get_data_ref().rows(),     //
                        input.get_data_ref().cols(),     //
                        alpha                  //
                        )                      //
                );

        // Apply the LeakyReLU activation function
        Eigen::MatrixXf activated =
            (input.get_data_ref().array() > 0) // For each element, check if it's greater than 0
                .select(             // Select values based on the condition
                    input.get_data_ref(),      // If the element is greater than 0, it remains unchanged
                    input.get_data_ref().array() * alpha // Otherwise, it's scaled by alpha
                );

        return Tensor{activated};
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        Eigen::MatrixXf grad_input = grad_output.get_data_ref().array() * leaky_grad.array();
        return Tensor{grad_input};
    }
};

#endif // LEAKYRELU_HPP
