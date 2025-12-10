#ifndef SGD_MINIMAL_HPP
#define SGD_MINIMAL_HPP

#include "Optimizer.hpp"

struct SGDMinimal : public Optimizer
{
    float learning_rate;

    explicit SGDMinimal(float learnningRate = 0.01F) : learning_rate(learnningRate) {}

    auto step(std::vector<nn::Tensor*>& params) -> void override
    {
        for (nn::Tensor* param : params)
        {
            param->get_data_ref() -= learning_rate * param->get_grad_ref();
        }
    }

    void zero_grad(std::vector<nn::Tensor*>& params) override
    {
        for (nn::Tensor* param : params)
        {
            param->zero_grad();
        }
    }
};

#endif // SGD_MINIMAL_HPP
