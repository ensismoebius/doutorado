#ifndef SGD_HPP
#define SGD_HPP

#include "Optimizer.hpp"

struct SGD : public Optimizer
{
    float learning_rate;
    float momentum;

    std::vector<Eigen::MatrixXf> velocity;

    explicit SGD(float lr = 0.01F, float momentum = 0.0F) : learning_rate(lr), momentum(momentum) {}

    auto attach(std::vector<Tensor*>& paramsList) -> void
    {
        velocity.clear();
        for (auto* param : paramsList)
        {
            velocity.emplace_back(Eigen::MatrixXf::Zero(param->grad.rows(), param->grad.cols()));
        }
    }

    auto step(std::vector<Tensor*>& paramsList) -> void override
    {
        for (size_t i = 0; i < paramsList.size(); ++i)
        {
            auto& param = *paramsList[i];
            velocity[i] = momentum * velocity[i] - learning_rate * param.grad;
            param.data += velocity[i];
        }
    }

    auto zero_grad(std::vector<Tensor*>& paramsList) -> void override
    {
        for (auto* param : paramsList)
        {
            param->grad.setZero();
        }
    }
};

#endif // SGD_HPP
