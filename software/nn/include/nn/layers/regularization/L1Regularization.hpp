#ifndef L1_REGULARIZATION_HPP
#define L1_REGULARIZATION_HPP

#include "nn/layers/regularization/IRegularization.hpp"

class L1Regularization : public IRegularization
{
   public:
    explicit L1Regularization(float lambda) : IRegularization(lambda) {}

    auto forward(const std::vector<nn::Tensor*>& params) -> nn::Tensor override
    {
        float penalty = 0.0F;
        for (const auto* param : params)
        {
            for (size_t i = 0; i < param->rows(); ++i)
            {
                for (size_t j = 0; j < param->cols(); ++j)
                {
                    penalty += std::abs(param->at(i, j));
                }
            }
        }
        penalty *= lambda_;
        nn::Tensor loss(1, 1);
        loss.at(0, 0) = penalty;
        return loss;
    } //

    void backward(const std::vector<nn::Tensor*>& params) override
    {
        for (auto* param : params)
        {
            auto grad = param->grad();
            for (size_t i = 0; i < param->rows(); ++i)
            {
                for (size_t j = 0; j < param->cols(); ++j)
                {
                    float val = param->at(i, j);
                    float sign = (val > 0.0f) ? 1.0f : (val < 0.0f) ? -1.0f : 0.0f;
                    grad.at(i, j) = grad.at(i, j) + lambda_ * sign;
                }
            }
            param->set_grad(grad);
        }
    }
};

#endif // L1_REGULARIZATION_HPP
