#ifndef RELU_HPP
#define RELU_HPP

#include <Eigen/Dense>

#include "../tensor/Tensor.hpp"
#include "core/layers/Module.hpp"

struct ReLU : public Module
{
    Eigen::MatrixXf relu_grad; // usado para backward

    auto forward(const nn::Tensor& input) -> nn::Tensor override
    {
        // Guarda o gradiente da entrada atual para usar na fase de backward
        relu_grad = (input.get_data_ref().array() > 0).cast<float>();

        // Calcula a ativação
        Eigen::MatrixXf const activated = input.get_data_ref().array().max(0);
        return nn::Tensor{activated};
    }

    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override
    {
        Eigen::MatrixXf const grad_input = grad_output.get_data_ref().array() * relu_grad.array();
        return nn::Tensor{grad_input};
    }
};

#endif // RELU_HPP