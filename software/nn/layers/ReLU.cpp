#ifndef RELU_CPP
#define RELU_CPP

#include <Eigen/Dense>
#include "../tensor/Tensor.hpp"

struct ReLU
{
    Eigen::MatrixXf mask; // usado para backward

    auto forward(const Tensor &input) -> Tensor {
      mask = (input.data.array() > 0).cast<float>();
      Eigen::MatrixXf activated = input.data.array().max(0);
      return {activated};
    }

    auto backward(const Tensor &grad_output) -> Tensor {
      Eigen::MatrixXf grad_input = grad_output.data.array() * mask.array();
      return {grad_input};
    }
};

#endif // RELU_CPP