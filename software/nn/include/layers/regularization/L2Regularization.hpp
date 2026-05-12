#ifndef L2_REGULARIZATION_HPP
#define L2_REGULARIZATION_HPP

#include "layers/regularization/IRegularization.hpp"

class L2Regularization : public IRegularization
{
   public:
    explicit L2Regularization(float lambda) : IRegularization(lambda) {}

    auto forward(const std::vector<nn::Tensor*>& params) -> nn::Tensor override
    {
        float penalty = 0.0F;
        for (const auto* param : params)
        {
            nn::Tensor squared = param->square();
            penalty += squared.sum();
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
                    grad.at(i, j) = grad.at(i, j) + 2.0F * lambda_ * param->at(i, j);
                }
            }
            param->set_grad(grad);
        }
    }
};

#endif // L2_REGULARIZATION_HPP