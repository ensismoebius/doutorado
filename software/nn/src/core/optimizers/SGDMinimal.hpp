#ifndef SGD_MINIMAL_HPP
#define SGD_MINIMAL_HPP

#include <span>

#include "Optimizer.hpp"

struct SGDMinimal : public Optimizer
{
    float learning_rate;

    explicit SGDMinimal(float learnningRate = 0.01F) : learning_rate(learnningRate) {}

    auto step(std::span<nn::Tensor*> params) -> void override
    {
        for (auto* param : params) [[likely]]
        {
            param->get_data_ref() -= learning_rate * param->get_grad_ref();
        }
    }

    auto zero_grad(std::span<nn::Tensor*> params) -> void override
    {
        for (auto* param : params) [[likely]]
        {
            param->zero_grad();
        }
    }
};

#endif // SGD_MINIMAL_HPP
