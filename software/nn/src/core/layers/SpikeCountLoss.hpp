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
        float loss = (diff.get_data_ref().array().square()).mean();
        nn::Tensor loss_tensor(1, 1);
        loss_tensor.at(0, 0) = loss;
        return loss_tensor;
    }
    auto backward(const nn::Tensor& /*grad_output*/) -> nn::Tensor override
    {
        auto grad_data = 2.0F * (last_input.get_data_ref() - target.get_data_ref()) /
                         last_input.get_data_ref().size();
        return nn::Tensor(grad_data);
    }
};
