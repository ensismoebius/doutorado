#ifndef RELU_CPP
#define RELU_CPP

#include <Eigen/Dense>
#include "../tensor/Tensor.hpp"

struct ReLU
{
    Eigen::MatrixXf mask; // usado para backward

    Tensor forward(const Tensor &input)
    {
        mask = (input.data.array() > 0).cast<float>();
        Eigen::MatrixXf activated = input.data.array().max(0);
        return Tensor(activated);
    }

    Tensor backward(const Tensor &grad_output)
    {
        Eigen::MatrixXf grad_input = grad_output.data.array() * mask.array();
        return Tensor(grad_input);
    }
};

#endif // RELU_CPP