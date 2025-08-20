#ifndef LEAKYRELU_HPP
#define LEAKYRELU_HPP

#include "../tensor/Tensor.hpp"
#include "layers/Module.hpp"
#include <Eigen/Dense>

struct LeakyReLU : public Module
{
  float alpha; // negative slope
  Eigen::MatrixXf leaky_grad;

  LeakyReLU(float alpha_ = 0.01F) : alpha(alpha_) {}

  auto forward(const Tensor &input) -> Tensor override
  {
    // Guarda o gradiente da entrada atual para usar na fase de backward
    leaky_grad = (input.data.array() > 0).cast<float>() + (input.data.array() <= 0).cast<float>() * alpha;
    // Calcula a ativação
    Eigen::MatrixXf activated = input.data.array().max(0) + (input.data.array().min(0) * alpha);
    return {activated};
  }

  auto backward(const Tensor &grad_output) -> Tensor override
  {
    Eigen::MatrixXf grad_input = grad_output.data.array() * leaky_grad.array();
    return {grad_input};
  }
};

#endif // LEAKYRELU_HPP
