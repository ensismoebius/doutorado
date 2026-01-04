#ifndef SGD_HPP
#define SGD_HPP

#include <span>
#include <stdexcept>

#include "Optimizer.hpp"

struct SGD : public Optimizer
{
    float learning_rate;
    float momentum;

    std::vector<nn::Tensor> velocity;

    explicit SGD(float lr = 0.01F, float momentum = 0.0F) : learning_rate(lr), momentum(momentum)
    {
        if (lr <= 0.0F)
        {
            throw std::invalid_argument("Learning rate must be positive");
        }
        if (lr > 1e8F)
        {
            throw std::invalid_argument("Learning rate is unreasonably large");
        }
    }

    auto attach(std::span<nn::Tensor*> paramsList) -> void override
    {
        velocity.clear();
        velocity.reserve(paramsList.size()); // Pre-allocate memory
        for (auto* param : paramsList)
        {
            if (param == nullptr)
            {
                throw std::invalid_argument("Cannot attach null parameter to optimizer");
            }
            // Initialize velocity tensor with zeros matching parameter shape
            velocity.emplace_back(param->get_grad_ref().rows(), param->get_grad_ref().cols());
            velocity.back().get_data_ref().setZero();
        }
    }

    auto step(std::span<nn::Tensor*> paramsList) -> void override
    {
        for (size_t i = 0; i < paramsList.size(); ++i) [[likely]]
        {
            if (paramsList[i] == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            auto& param = *paramsList[i];
            velocity[i].get_data_ref() =
                momentum * velocity[i].get_data_ref() - learning_rate * param.get_grad_ref();
            param.get_data_ref() += velocity[i].get_data_ref();
        }
    }

    auto zero_grad(std::span<nn::Tensor*> paramsList) -> void override
    {
        for (auto* param : paramsList) [[likely]]
        {
            if (param == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            param->zero_grad();
        }
    }
};

#endif // SGD_HPP
