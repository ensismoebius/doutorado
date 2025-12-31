#pragma once

#include "../tensor/Tensor.hpp"
#include "Module.hpp"

// Spike count loss: mean squared error between output spike count and target
class SpikeCountLoss : public Module
{
   public:
    SpikeCountLoss() = default;
    void set_target(const nn::Tensor& t)
    {
        target = t;
    }
    auto forward(const nn::Tensor& pred) -> nn::Tensor override
    {
        // pred and target: (n_samples, 1)
        Eigen::MatrixXf diff = pred.get_data_ref() - target.get_data_ref();
        float loss = diff.array().square().mean();
        nn::Tensor loss_tensor(1, 1);
        loss_tensor.at(0, 0) = loss;
        return loss_tensor;
    }
    auto backward(const nn::Tensor& pred) -> nn::Tensor override
    {
        Eigen::MatrixXf grad =
            2.0F * (pred.get_data_ref() - target.get_data_ref()) / pred.get_data_ref().size();
        return nn::Tensor(grad);
    }
    void reset_parameters() {}

   private:
    nn::Tensor target;
};
