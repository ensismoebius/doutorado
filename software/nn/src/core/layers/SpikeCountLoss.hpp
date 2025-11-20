#pragma once
#include <Eigen/Dense>

#include "../tensor/Tensor.hpp"
#include "Module.hpp"

// Spike count loss: mean squared error between output spike count and target
class SpikeCountLoss : public Module
{
   public:
    SpikeCountLoss() = default;
    void set_target(const Tensor& t)
    {
        target = t;
    }
    auto forward(const Tensor& pred) -> Tensor override
    {
        // pred and target: (n_samples, 1)
        Eigen::MatrixXf diff = pred.get_data_ref() - target.get_data_ref();
        float loss = diff.array().square().mean();
        Eigen::MatrixXf loss_mat(1, 1);
        loss_mat(0, 0) = loss;
        return Tensor(loss_mat);
    }
    auto backward(const Tensor& pred) -> Tensor override
    {
        Eigen::MatrixXf grad = 2.0F * (pred.get_data_ref() - target.get_data_ref()) / pred.get_data_ref().size();
        return Tensor(grad);
    }
    void reset_parameters() {}

   private:
    Tensor target;
};
