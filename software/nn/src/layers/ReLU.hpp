#ifndef RELU_HPP
#define RELU_HPP

#include <Eigen/Dense>

#include "../tensor/Tensor.hpp"
#include "layers/Module.hpp"

struct ReLU : public Module
{
    Eigen::MatrixXf relu_grad; // usado para backward

    auto forward(const Tensor& input) -> Tensor override
    {
        // Guarda o gradiente da entrada atual para usar na fase de backward
        relu_grad = (input.data.array() > 0).cast<float>();

        // Calcula a ativação
        Eigen::MatrixXf const activated = input.data.array().max(0);
        return {activated};
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        Eigen::MatrixXf const grad_input = grad_output.data.array() * relu_grad.array();
        return {grad_input};
    }
};

#endif // RELU_HPP