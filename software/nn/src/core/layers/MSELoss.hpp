#ifndef MSELOSS_HPP
#define MSELOSS_HPP

#include <Eigen/Dense>
#include <iostream>
#include <limits>

#include "../tensor/Tensor.hpp"
#include "Module.hpp"

class MSELoss : public Module
{
   private:
    // Constants for numerical stability
    static constexpr float MAX_VALUE_FACTOR = 2.0F;
    static constexpr float MSE_GRADIENT_FACTOR = 2.0F;
    static constexpr float MAX_GRADIENT_NORM = 1.0F;

    nn::Tensor last_input;
    nn::Tensor last_target;

   public:
    MSELoss() = default;

    // Forward computes the loss value as a Tensor (scalar) with numerical stability checks
    auto forward(const nn::Tensor& prediction) -> nn::Tensor override
    {
        // Cache the prediction for the backward pass
        last_input = prediction;

        // Use last_target set by set_target
        Eigen::MatrixXf diff = prediction.get_data_ref() - last_target.get_data_ref();

        // Check for invalid values in the predictions
        if (!prediction.get_data_ref().allFinite()) [[unlikely]]
        {
            std::cerr << "Warning: Non-finite values detected in predictions\n";
            // Return a very large but finite loss
            return nn::Tensor{Eigen::MatrixXf::Constant(
                1, 1, std::numeric_limits<float>::max() / MAX_VALUE_FACTOR)};
        }

        // Compute MSE with careful reduction
        float sum_squared = 0.0F;
        long count = diff.size();

#pragma omp parallel for reduction(+ : sum_squared)
        for (long i = 0; i < count; ++i)
        {
            float val = diff(i);
            sum_squared += val * val;
        }

        float mse = sum_squared / static_cast<float>(count);

        // Clip extremely large values to prevent overflow
        mse = std::min(mse, std::numeric_limits<float>::max() / MAX_VALUE_FACTOR);

        return nn::Tensor{Eigen::MatrixXf::Constant(1, 1, mse)};
    }

    // Set the target tensor for the loss
    void set_target(const nn::Tensor& target)
    {
        last_target = target;
    }

    // Backward computes the gradient of the loss w.r.t. prediction with gradient clipping
    auto backward(const nn::Tensor& /* prediction */) -> nn::Tensor override
    {
        Eigen::MatrixXf grad =
            MSE_GRADIENT_FACTOR * (last_input.get_data_ref() - last_target.get_data_ref()) / last_input.get_data_ref().size();

        // Check for invalid gradients
        if (!grad.allFinite()) [[unlikely]]
        {
            std::cerr << "Warning: Non-finite gradients detected in MSE backward pass\n";
            grad.setZero(); // Return zero gradient to prevent further issues
            return nn::Tensor{grad};
        }

        // Gradient clipping to prevent explosion
        float grad_norm = grad.norm();
        constexpr float max_grad_norm = 1.0F;
        if (grad_norm > max_grad_norm) [[unlikely]]
        {
            grad *= max_grad_norm / grad_norm;
        }

        return nn::Tensor{grad};
    }
};

#endif // MSELOSS_HPP
