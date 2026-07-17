#ifndef MAELOSS_HPP
#define MAELOSS_HPP

#include <cmath>
#include <limits>
#include <sstream>

#include "layers/base/Module.hpp"
#include "logging/Logger.hpp"

/**
 * @file MAELoss.hpp
 * @brief Mean Absolute Error loss module.
 *
 * Design notes for this codebase:
 * - This is implemented as a `Module` for a PyTorch-like training loop.
 * - `forward()` returns a scalar tensor (shape 1x1) representing the loss value.
 * - `backward()` returns the gradient w.r.t. the prediction (dL/d(prediction)), with the
 *   same shape as the last `input` passed to `forward()`.
 *
 * Usage pattern:
 * - Call `set_target(target)` before `forward(pred)`.
 * - During training with gradients, `forward(..., requires_grad=true)` caches the last
 *   prediction internally for use in `backward()`.
 */
template <typename Backend>
class MAELossImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   private:
    static constexpr float kMaxValueFactor = 2.0F;

    Tensor last_input_;
    Tensor last_target_;
    bool target_set_ = false;
    bool training_ = true;

   public:
    /// Optional gradient-norm clip applied inside backward(). 0 = disabled (default), which
    /// makes backward() the exact MAE gradient. Prefer TrainerConfig::grad_clip_norm.
    float max_gradient_norm = 0.0F;

    MAELossImpl() = default;

    void train(bool on) override
    {
        training_ = on;
    }

    void set_target(const Tensor& target)
    {
        last_target_ = target;
        target_set_ = true;
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        if (!target_set_)
        {
            throw std::runtime_error("MAELoss: target has not been set. Call set_target() first.");
        }

        if (training_ && requires_grad)
        {
            last_input_ = input;
        }

        {
            std::ostringstream _dbg_oss;
            _dbg_oss << "MAELoss shapes input=" << input.rows() << "x" << input.cols()
                     << " target=" << last_target_.rows() << "x" << last_target_.cols();
            NN_LOG_DEBUG(_dbg_oss.str());
        }

        Tensor diff = input;
        diff.subtract_inplace(last_target_);

        double acc = 0.0;
        const size_t n = diff.size();
        for (size_t r = 0; r < diff.rows(); ++r)
        {
            for (size_t c = 0; c < diff.cols(); ++c)
            {
                acc += std::fabs(static_cast<double>(diff.at(r, c)));
            }
        }

        float mae = static_cast<float>(acc / static_cast<double>(n));
        if (std::isfinite(mae))
        {
            mae = std::min(mae, std::numeric_limits<float>::max() / kMaxValueFactor);
        }

        Tensor loss_tensor(1, 1);
        loss_tensor.at(0, 0) = mae;
        return loss_tensor;
    }

    auto backward(const Tensor& /*prediction*/) -> Tensor override
    {
        Tensor grad = last_input_;
        grad.subtract_inplace(last_target_);

        const float denom = static_cast<float>(grad.size());
        for (size_t r = 0; r < grad.rows(); ++r)
        {
            for (size_t c = 0; c < grad.cols(); ++c)
            {
                const float v = grad.at(r, c);
                float sign = 0.0F;
                if (v > 0.0F)
                {
                    sign = 1.0F;
                }
                else if (v < 0.0F)
                {
                    sign = -1.0F;
                }
                grad.at(r, c) = sign / denom;
            }
        }

        const float grad_check = grad.norm();
        if (!std::isfinite(grad_check)) [[unlikely]]
        {
            NN_LOG_ERROR(                                                       //
                "Warning: Non-finite gradients detected in MAE backward pass"); //
            Tensor zero_grad(last_input_.rows(), last_input_.cols());           //
            for (size_t i = 0; i < zero_grad.rows(); ++i)                       //
            {                                                                   //
                for (size_t j = 0; j < zero_grad.cols(); ++j)                   //
                {                                                               //
                    zero_grad.at(i, j) = 0.0F;                                  //
                }
            } //
            return zero_grad; //
        } //

        // Optional gradient-norm clipping. OFF by default (0), so backward() returns the
        // true MAE gradient. This was unconditional at norm 1.0, which silently made MAELoss
        // not the MAE gradient and contradicted TrainerConfig::grad_clip_norm (default 0 =
        // "no clipping"). Same defect as MSELoss — see the long note there. Prefer
        // TrainerConfig::grad_clip_norm; this is only an escape hatch.
        if (max_gradient_norm > 0.0F && grad_check > max_gradient_norm) [[unlikely]]
        {
            grad.multiply_scalar_inplace(max_gradient_norm / grad_check);
        }

        return grad;
    }
};

#endif // MAELOSS_HPP
