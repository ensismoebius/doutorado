#ifndef NN_OPTIMIZERS_SCHEDULE_FREE_ADAMW_HPP
#define NN_OPTIMIZERS_SCHEDULE_FREE_ADAMW_HPP

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include "optimizers/Optimizer.hpp"

/**
 * @file ScheduleFreeAdamW.hpp
 * @brief Schedule-Free AdamW optimizer.
 *
 * Schedule-Free removes the learning-rate schedule. A cosine/linear decay schedule normally
 * has to know the total step count in advance, and picking it wrong costs accuracy; this
 * method instead gets schedule-like behaviour from *iterate averaging*, so training can be
 * stopped at any step without having committed to a horizon up front.
 *
 * It maintains three sequences per parameter:
 *   - z: the raw AdamW-style gradient iterate,
 *   - x: a weighted running average of z (the point that actually converges),
 *   - y: an interpolation of x and z, and the point where *gradients are evaluated*.
 *
 * The parameter tensor holds `y` while training and `x` while evaluating -- see
 * `train_mode()`. This distinction is not cosmetic: the convergence guarantee is about `x`,
 * so validating at `y` understates the method. `Trainer` brackets its validation pass with
 * `train_mode(false)/train_mode(true)` automatically.
 *
 * Per step k (matching the reference `adamw_schedulefree_reference.py` exactly):
 *   sched   = (k+1)/warmup_steps if k < warmup_steps else 1
 *   lr      = base_lr * sched                       (per-parameter: * lr_scales_[i])
 *   bc2     = 1 - beta2^(k+1)
 *   lr_max  = max(lr, lr_max)
 *   weight  = (k+1)^r * lr_max^weight_lr_power
 *   sum    += weight ;  ckp1 = weight / sum
 *   per parameter:
 *     if decay: z -= lr * decay * (decay_at_z ? z : y)
 *     v  = beta2*v + (1-beta2)*g^2
 *     z -= lr * g / (sqrt(v / bc2) + eps)
 *     x  = (1-ckp1)*x + ckp1*z
 *     y  = beta1*x + (1-beta1)*z
 *     p  = y
 *
 * With the defaults (r=0, weight_lr_power=2) and a constant lr, `weight` is constant, so
 * ckp1 = 1/(k+1) and `x` is exactly the running mean of the z iterates.
 *
 * Per-parameter lr scales (`lr_scales_`) are honored, so SNN biophysical scalars can take a
 * reduced lr as with every other optimizer here (see Optimizer.hpp).
 *
 * **Deliberate deviation from the reference**: upstream applies weight decay to every
 * parameter; this project restricts decoupled decay to 2-D weight matrices so the SNN
 * scalars (R, C, V_th) are never decayed (see Optimizer::weight_decay). With
 * weight_decay == 0 (the default) the two are identical.
 *
 * Defaults follow the reference: lr=0.0025, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
 * warmup_steps=0, r=0, weight_lr_power=2, decay_at_z=false.
 *
 * Reference: A. Defazio et al., "The Road Less Scheduled," NeurIPS 2024. arXiv:2405.15682.
 * Ported from `schedulefree` 1.4.1 (schedulefree/adamw_schedulefree_reference.py), the
 * variant that keeps x/y/z explicit rather than the memory-optimized in-place form.
 */
struct ScheduleFreeAdamW : public Optimizer
{
    using Tensor = Optimizer::Tensor;

    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int warmup_steps;
    float r_exponent;      ///< `r` in the weight formula (reference default 0).
    float weight_lr_power; ///< Reference default 2.
    bool decay_at_z;       ///< Reference default false (decay is applied at y).

    int k = 0;               ///< Step counter.
    float lr_max = 0.0F;     ///< Running max of the scheduled lr.
    float weight_sum = 0.0F; ///< Denominator of the averaging weight.
    bool in_train_mode = true;

    std::vector<Tensor> z_;
    std::vector<Tensor> x_;
    std::vector<Tensor> exp_avg_sq_;

    explicit ScheduleFreeAdamW(float lr = 0.0025F,
        float beta1_ = 0.9F,
        float beta2_ = 0.999F,
        float epsilon_ = 1e-8F,
        int warmup_steps_ = 0,
        float r_ = 0.0F,
        float weight_lr_power_ = 2.0F,
        bool decay_at_z_ = false)
        : learning_rate(lr),
          beta1(beta1_),
          beta2(beta2_),
          epsilon(epsilon_),
          warmup_steps(warmup_steps_),
          r_exponent(r_),
          weight_lr_power(weight_lr_power_),
          decay_at_z(decay_at_z_)
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
            throw std::invalid_argument("ScheduleFreeAdamW: betas must lie in [0, 1]");
        }
        if (warmup_steps_ < 0)
        {
            throw std::invalid_argument("ScheduleFreeAdamW: warmup_steps must be >= 0");
        }
    }

    auto attach(std::span<Tensor*> params) -> void override
    {
        Optimizer::attach(params);

        k = 0;
        lr_max = 0.0F;
        weight_sum = 0.0F;
        in_train_mode = true;

        z_.clear();
        x_.clear();
        exp_avg_sq_.clear();
        for (auto* param : params)
        {
            if (param == nullptr)
            {
                throw std::invalid_argument("Cannot attach null parameter to optimizer");
            }
            // z and x both start at the current parameter value; y == p already.
            z_.push_back(*param);
            x_.push_back(*param);
            exp_avg_sq_.emplace_back(param->rows(), param->cols());
            exp_avg_sq_.back().set_zero();
        }
    }

    /// Write the evaluation iterate (x) or the training iterate (y) into the parameters.
    /// Idempotent: repeated calls with the same value do nothing.
    auto train_mode(bool on) -> void override
    {
        if (on == in_train_mode) return;

        for (std::size_t i = 0; i < attached_params_.size(); ++i)
        {
            if (attached_params_[i] == nullptr) continue;
            auto& param = *attached_params_[i];
            if (!on)
            {
                // Switch to eval: p currently holds y; save nothing (y is recomputable
                // from x and z) and expose the averaged iterate x.
                param = x_[i];
            }
            else
            {
                // Back to train: restore y = beta1*x + (1-beta1)*z.
                param = x_[i].multiply_scalar(beta1).add(z_[i].multiply_scalar(1.0F - beta1));
            }
        }
        in_train_mode = on;
    }

    auto step(std::span<Tensor*> paramsList) -> void override
    {
        if (!in_train_mode)
        {
            // Stepping while the parameters hold x would corrupt the y sequence. The
            // reference raises here too rather than silently producing wrong iterates.
            throw std::runtime_error(
                "ScheduleFreeAdamW::step() called in eval mode; call train_mode(true) first");
        }

        const float sched = (warmup_steps > 0 && k < warmup_steps)
                                ? (static_cast<float>(k + 1) / static_cast<float>(warmup_steps))
                                : 1.0F;
        const float lr = learning_rate * sched;
        const float bias_correction2 = 1.0F - std::pow(beta2, static_cast<float>(k + 1));

        lr_max = std::max(lr, lr_max);
        const float weight =
            std::pow(static_cast<float>(k + 1), r_exponent) * std::pow(lr_max, weight_lr_power);
        weight_sum += weight;
        const float ckp1 = (weight_sum > 0.0F) ? (weight / weight_sum) : 0.0F;

        for (std::size_t i = 0; i < paramsList.size(); ++i) [[likely]]
        {
            if (paramsList[i] == nullptr)
            {
                throw std::invalid_argument("Parameter pointer is null");
            }
            auto& param = *paramsList[i];
            const float lr_i = lr * (i < lr_scales_.size() ? lr_scales_[i] : 1.0F);
            const Tensor grad = param.grad();

            // Weight decay, applied to z at either y (default) or z. Restricted to 2-D
            // weight matrices -- see the header note.
            if (weight_decay > 0.0F && param.rows() > 1 && param.cols() > 1)
            {
                // p currently holds y, so `param` IS the y iterate.
                const Tensor& decay_point = decay_at_z ? z_[i] : param;
                z_[i] = z_[i].add(decay_point.multiply_scalar(-lr_i * weight_decay));
            }

            // v = beta2*v + (1-beta2)*g^2 ; denom = sqrt(v / bc2) + eps
            exp_avg_sq_[i] = exp_avg_sq_[i].multiply_scalar(beta2).add(
                grad.multiply(grad).multiply_scalar(1.0F - beta2));
            const Tensor denom =
                exp_avg_sq_[i].multiply_scalar(1.0F / bias_correction2).sqrt().add_scalar(epsilon);

            // z -= lr * g / denom
            z_[i] = z_[i].add(grad.divide(denom).multiply_scalar(-lr_i));

            // x = (1-ckp1)*x + ckp1*z
            x_[i] = x_[i].multiply_scalar(1.0F - ckp1).add(z_[i].multiply_scalar(ckp1));

            // y = beta1*x + (1-beta1)*z ; parameters hold y while training.
            param = x_[i].multiply_scalar(beta1).add(z_[i].multiply_scalar(1.0F - beta1));
        }

        k += 1;
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

#endif // NN_OPTIMIZERS_SCHEDULE_FREE_ADAMW_HPP
