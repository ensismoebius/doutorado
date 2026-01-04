#pragma once

#include "../tensor/Tensor.hpp"
#include "Module.hpp"

// Spike count loss: mean squared error between output spike count and target
class SpikeCountLoss : public Module
{
   private:
    nn::Tensor target;
    nn::Tensor last_input;
    bool training = true;

   public:
    SpikeCountLoss() = default;

    void train(bool on) override
    {
        training = on;
    }

    void set_target(const nn::Tensor& t)
    {
        target = t;
    }
    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        if (training && requires_grad)
        {
            last_input = input;
        }
        // pred and target: (n_samples, 1)
        auto diff = last_input.add(target.multiply_scalar(-1.0F));
        float sum_sq = 0.0f;
        for (size_t i = 0; i < diff.rows(); ++i)
        {
            for (size_t j = 0; j < diff.cols(); ++j)
            {
                float val = diff.at(i, j);
                sum_sq += val * val;
            }
        }
        float loss = sum_sq / static_cast<float>(diff.size());
        nn::Tensor loss_tensor(1, 1);
        loss_tensor.at(0, 0) = loss;
        return loss_tensor;
    }
    auto backward(const nn::Tensor& /*grad_output*/) -> nn::Tensor override
    {
        // Gradient: 2 * (prediction - target) / N
        auto grad_data = last_input.add(target.multiply_scalar(-1.0f));
        grad_data = grad_data.multiply_scalar(2.0f / static_cast<float>(last_input.size()));
        return nn::Tensor(grad_data);
    }
};
