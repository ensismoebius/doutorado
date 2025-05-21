#ifndef RELU_CPP
#define RELU_CPP

#include "../tensor/Tensor.hpp"
#include <Eigen/Dense>

struct ReLU {
  Eigen::MatrixXf mask; // usado para backward

  auto forward(const Tensor &input) -> Tensor {
    mask = (input.data.array() > 0).cast<float>();
    Eigen::MatrixXf const activated = input.data.array().max(0);
    return {activated};
  }

  auto backward(const Tensor &grad_output) -> Tensor {
    Eigen::MatrixXf const grad_input = grad_output.data.array() * mask.array();
    return {grad_input};
  }
};

#endif // RELU_CPP