#ifndef SGD_HPP
#define SGD_HPP

#include <span>

#include "Optimizer.hpp"

struct SGD : public Optimizer
{
    float learning_rate;
    float momentum;

    std::vector<Eigen::MatrixXf> velocity;

    explicit SGD(float lr = 0.01F, float momentum = 0.0F) : learning_rate(lr), momentum(momentum) {}

    auto attach(std::span<nn::Tensor*> paramsList) -> void
    {
        velocity.clear();
        velocity.reserve(paramsList.size()); // Pre-allocate memory
        std::transform(paramsList.begin(), paramsList.end(), std::back_inserter(velocity),
                       [](nn::Tensor* param) {
                           return Eigen::MatrixXf::Zero(param->get_grad_ref().rows(),
                                                         param->get_grad_ref().cols());
                       });
    }

    auto step(std::span<nn::Tensor*> paramsList) -> void override
    {
        for (size_t i = 0; i < paramsList.size(); ++i) [[likely]]
        {
            auto& param = *paramsList[i];
            velocity[i] = momentum * velocity[i] - learning_rate * param.get_grad_ref();
            param.get_data_ref() += velocity[i];
        }
    }

    auto zero_grad(std::span<nn::Tensor*> paramsList) -> void override
    {
        for (auto* param : paramsList) [[likely]]
        {
            param->zero_grad();
        }
    }
};

#endif // SGD_HPP
