#ifndef RELU_HPP
#define RELU_HPP

#include "../tensor/Tensor.hpp"
#include "core/layers/Module.hpp"

struct ReLU : public Module
{
    nn::Tensor relu_grad; // usado para backward

    auto forward(const nn::Tensor& input) -> nn::Tensor override
    {
        // Guarda o gradiente da entrada atual para usar na fase de backward
        // relu_grad stores which elements were > 0 (for gradient computation)
        Eigen::MatrixXf grad_mask = (input.get_data_ref().array() > 0).cast<float>();
        relu_grad = nn::Tensor(grad_mask);

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