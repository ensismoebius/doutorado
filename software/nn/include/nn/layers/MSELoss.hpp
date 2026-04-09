#ifndef MSELOSS_HPP
#define MSELOSS_HPP

#include <cmath>
#include <limits>
#include <sstream>

#include "nn/layers/Module.hpp"
#include "nn/logging/Logger.hpp"

/**
 * @file MSELoss.hpp
 * @brief Mean Squared Error loss module.
 *
 * Design notes for this codebase:
 * - This is implemented as a `Module` for a PyTorch-like training loop.
 * - `forward()` returns a *scalar* tensor (shape 1x1) representing the loss value.
 * - `backward()` returns the gradient w.r.t. the prediction (dL/d(prediction)), with the
 *   same shape as the last `input` passed to `forward()`.
 *
 * Usage pattern:
 * - Call `set_target(target)` before `forward(pred)`.
 * - During training with gradients, `forward(..., requires_grad=true)` caches the last
 *   prediction internally for use in `backward()`.
 */
template <typename Backend>
class MSELossImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   private:
    // Constants for numerical stability in loss and gradient computations.
    static constexpr float kMaxValueFactor = 2.0F;
    static constexpr float kMseGradientFactor = 2.0F;
    static constexpr float kMaxGradientNorm = 1.0F;

    Tensor last_input_;
    Tensor last_target_;
    bool target_set_ = false;

    bool training_ = true;

   public:
    MSELossImpl() = default;

    void train(bool on) override
    {
        training_ = on;
    }

    // Forward computes the loss value as a Tensor (scalar) with numerical stability checks
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        // Check if target has been set
        if (!target_set_)
        {
            throw std::runtime_error("MSELoss: target has not been set. Call set_target() first.");
        }

        if (training_ && requires_grad)
        {
            // Cache the prediction for the backward pass.
            // Note: we do not currently cache the target here because it is provided
            // via set_target() and stored in `last_target`.
            last_input_ = input;
        }

        // Use last_target set by set_target. Previously we emitted debug
        // information to stderr and temporary files to diagnose shape
        // mismatches; that was noisy in production. Keep a debug macro call
        // (no-op by default) so developers can enable it locally if needed.
        {
            std::ostringstream _dbg_oss;
            _dbg_oss << "MSELoss shapes input=" << input.rows() << "x" << input.cols()
                     << " target=" << last_target_.rows() << "x" << last_target_.cols();
            NN_LOG_DEBUG(_dbg_oss.str());
        }

        float mse = input.mean_squared_error(last_target_);

        // Clip extremely large values to prevent overflow (but let NaN/Inf propagate)
        if (std::isfinite(mse))
        {
            mse = std::min(mse, std::numeric_limits<float>::max() / kMaxValueFactor);
        }
        Tensor loss_tensor(1, 1);
        loss_tensor.at(0, 0) = mse;
        return loss_tensor;
    }

    // Set the target tensor for the loss
    void set_target(const Tensor& target)
    {
        // Contract: target must be shape-compatible with the prediction passed to forward().
        last_target_ = target;
        target_set_ = true;
    }

    // Backward computes the gradient of the loss w.r.t. prediction with gradient clipping
    auto backward(const Tensor& /* prediction */) -> Tensor override
    {
        // Returns dL/d(prediction) for the *last* cached prediction.
        // The `prediction` argument is unused because `last_input` is cached in forward().
        // Compute gradient: 2 * (prediction - target) / num_elements
        Tensor grad = last_input_;
        grad.subtract_inplace(last_target_);
        float factor = kMseGradientFactor / static_cast<float>(last_input_.size());
        grad.multiply_scalar_inplace(factor);

        // Check for invalid gradients using norm (if norm is NaN or Inf, gradients are invalid)
        float grad_check = grad.norm();
        if (!std::isfinite(grad_check)) [[unlikely]]
        {
            NN_LOG_ERROR("Warning: Non-finite gradients detected in MSE backward pass");
            // Return zero gradient to prevent further issues
            Tensor zero_grad(last_input_.rows(), last_input_.cols());
            // Initialize to zero
            for (size_t i = 0; i < zero_grad.rows(); ++i)
            {
                for (size_t j = 0; j < zero_grad.cols(); ++j)
                {
                    zero_grad.at(i, j) = 0.0f;
                }
            }
            return zero_grad;
        }

        // Gradient clipping to prevent explosion
        float grad_norm = grad.norm();
        if (grad_norm > kMaxGradientNorm) [[unlikely]]
        {
            grad.multiply_scalar_inplace(kMaxGradientNorm / grad_norm);
        }

        return grad;
    }
};

#endif // MSELOSS_HPP
