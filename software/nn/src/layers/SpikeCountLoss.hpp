#pragma once
#include "../tensor/Tensor.hpp"
#include "Module.hpp"
#include <Eigen/Dense>

// Spike count loss: mean squared error between output spike count and target
class SpikeCountLoss : public Module
{
public:
  SpikeCountLoss() = default;
  void set_target(const Tensor &t)
  {
    target = t;
  }
  auto forward(const Tensor &pred) -> Tensor override
  {
    // pred and target: (n_samples, 1)
    Eigen::MatrixXf diff = pred.data - target.data;
    float loss = diff.array().square().mean();
    Eigen::MatrixXf loss_mat(1, 1);
    loss_mat(0, 0) = loss;
    return Tensor(loss_mat);
  }
  auto backward(const Tensor &pred) -> Tensor override
  {
    Eigen::MatrixXf grad = 2.0F * (pred.data - target.data) / pred.data.size();
    return Tensor(grad);
  }
  void reset_parameters() {}

private:
  Tensor target;
};
