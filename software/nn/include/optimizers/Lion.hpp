#ifndef NN_OPTIMIZERS_LION_HPP
#define NN_OPTIMIZERS_LION_HPP

#include <span>
#include <stdexcept>
#include <vector>

#include "optimizers/Optimizer.hpp"

/**
 * @file Lion.hpp
 * @brief Lion (EvoLved Sign Momentum) optimizer.
 *
 * Lion was discovered by symbolic program search over optimizer space. Unlike Adam it
 * keeps only ONE state tensor per parameter (the momentum EMA, no second moment) and its
 * update is the *sign* of an interpolation, so every parameter moves by exactly `lr` in
 * magnitude regardless of gradient scale. Halving Adam's optimizer memory is the main
 * practical draw; the uniform step size is why Lion's usable lr is typically 3-10x
 * smaller than Adam's, and its weight decay 3-10x larger.
 *
 * Update rule (per parameter), matching the reference implementation exactly:
 *   p     <- p * (1 - lr_i * weight_decay)              // decoupled decay, applied FIRST
 *   u     <- sign(beta1 * m + (1 - beta1) * g)          // interpolate, then take sign
 *   p     <- p - lr_i * u
 *   m     <- beta2 * m + (1 - beta2) * g                // momentum advances AFTER the step
 *
 * Note the ordering, which is easy to get wrong from memory: the decay precedes the update,
 * and the momentum EMA is advanced *after* the parameter has already moved. The two betas
 * play different roles -- beta1 interpolates for the update, beta2 tracks the EMA.
 *
 * Per-parameter lr scales (`lr_scales_`) are honored, so SNN biophysical scalars can take
 * a reduced lr like every other optimizer here (see Optimizer.hpp).
 *
 * **Deliberate deviation from the reference**: upstream applies its weight decay to every
 * parameter. This project restricts decoupled decay to 2-D weight matrices so the SNN
 * biophysical scalars (R, C, V_th -- 1x1) never have tau=R*C or the firing threshold
 * pulled toward zero; see Optimizer::weight_decay. With weight_decay == 0 (the default)
 * the two are identical, and the ground-truth fixture exercises decay on a 2-D matrix,
 * where they also agree exactly.
 *
 * Defaults follow the paper/reference: lr=1e-4, betas=(0.9, 0.99), weight_decay=0.
 *
 * Reference: X. Chen et al., "Symbolic Discovery of Optimization Algorithms,"
 * NeurIPS 2023. arXiv:2302.06675. Ported from `lion-pytorch` 0.2.5
 * (lion_pytorch/lion_pytorch.py::update_fn).
 */
struct Lion : public Optimizer
{
    using Tensor = Optimizer::Tensor;

    float learning_rate;
    float beta1; ///< Interpolation factor for the update direction.
    float beta2; ///< EMA decay for the momentum buffer.

    std::vector<Tensor> momentum; ///< Per-parameter EMA of gradients (the only state).

    explicit Lion(float lr = 1e-4F, float beta1_ = 0.9F, float beta2_ = 0.99F)
        : learning_rate(lr), beta1(beta1_), beta2(beta2_)
    {
        if (lr <= 0.0F)
        {
            throw std::invalid_argument("Learning rate must be positive");
        }
        if (lr > 1e8F)
        {
            throw std::invalid_argument("Learning rate is unreasonably large");
        }
        if (beta1_ < 0.0F || beta1_ > 1.0F || beta2_ < 0.0F || beta2_ > 1.0F)
        {
            throw std::invalid_argument("Lion: betas must lie in [0, 1]");
        }
    }

    auto attach(std::span<Tensor*> params) -> void override
    {
        Optimizer::attach(params);

        momentum.clear();
        momentum.reserve(params.size());
        for (auto* param : params)
        {
            if (param == nullptr)
            {
                throw std::invalid_argument("Cannot attach null parameter to optimizer");
            }
            momentum.emplace_back(param->rows(), param->cols());
            momentum.back().set_zero();
        }
    }

    auto step(std::span<Tensor*> paramsList) -> void override
    {
        for (std::size_t i = 0; i < paramsList.size(); ++i) [[likely]]
        {
            if (paramsList[i] == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            auto& param = *paramsList[i];
            const float lr_i = learning_rate * (i < lr_scales_.size() ? lr_scales_[i] : 1.0F);
            const Tensor grad = param.grad();

            // Decoupled weight decay FIRST: p *= (1 - lr_i * wd). Restricted to 2-D weight
            // matrices (see the header note) -- upstream applies it to every parameter.
            if (weight_decay > 0.0F && param.rows() > 1 && param.cols() > 1)
            {
                param = param.multiply_scalar(1.0F - (lr_i * weight_decay));
            }

            // u = sign(beta1 * m + (1 - beta1) * g), then p -= lr_i * u.
            // Done elementwise: the Tensor interface has no sign(), and adding one would
            // mean extending TensorBackendParityContract across all four backends.
            for (std::size_t r = 0; r < param.rows(); ++r)
            {
                for (std::size_t c = 0; c < param.cols(); ++c)
                {
                    const float interp =
                        (beta1 * momentum[i].at(r, c)) + ((1.0F - beta1) * grad.at(r, c));
                    const float sign = (interp > 0.0F) ? 1.0F : ((interp < 0.0F) ? -1.0F : 0.0F);
                    param.at(r, c) -= lr_i * sign;
                }
            }

            // Momentum EMA advances AFTER the parameter update: m = beta2*m + (1-beta2)*g.
            momentum[i] =
                momentum[i].multiply_scalar(beta2).add(grad.multiply_scalar(1.0F - beta2));
        }
    }

    auto zero_grad(std::span<Tensor*> paramsList) -> void override
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

#endif // NN_OPTIMIZERS_LION_HPP
