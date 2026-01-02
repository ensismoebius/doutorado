#ifndef MSELOSS_HPP
#define MSELOSS_HPP

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
    bool target_set = false;

    bool training = true;

   public:
    MSELoss() = default;

    void train(bool on) override
    {
        training = on;
    }

    // Forward computes the loss value as a Tensor (scalar) with numerical stability checks
    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        // Check if target has been set
        if (!target_set)
        {
            throw std::runtime_error("MSELoss: target has not been set. Call set_target() first.");
        }

        if (training && requires_grad)
        {
            // Cache the prediction for the backward pass
            last_input = input;
        }

        // Use last_target set by set_target
        float mse = input.mean_squared_error(last_target);

        // Check for invalid values in the predictions
        if (!input.get_data_ref().allFinite()) [[unlikely]]
        {
            std::cerr << "Warning: Non-finite values detected in predictions\n";
            // Return a very large but finite loss
            return nn::Tensor{Eigen::MatrixXf::Constant(
                1, 1, std::numeric_limits<float>::max() / MAX_VALUE_FACTOR)};
        }

        // Clip extremely large values to prevent overflow
        mse = std::min(mse, std::numeric_limits<float>::max() / MAX_VALUE_FACTOR);

        return nn::Tensor{Eigen::MatrixXf::Constant(1, 1, mse)};
    }

    // Set the target tensor for the loss
    void set_target(const nn::Tensor& target)
    {
        last_target = target;
        target_set = true;
    }

    // Backward computes the gradient of the loss w.r.t. prediction with gradient clipping
    auto backward(const nn::Tensor& /* prediction */) -> nn::Tensor override
    {
        // Compute gradient: 2 * (prediction - target) / num_elements
        // Create a copy to avoid modifying last_target
        nn::Tensor negated_target = last_target;
        negated_target.multiply_scalar(-1.0f);
        auto diff = last_input.add(negated_target);
        auto grad = diff.multiply_scalar(MSE_GRADIENT_FACTOR /
                                         static_cast<float>(last_input.get_data_ref().size()));

        // Check for invalid gradients
        if (!grad.get_data_ref().allFinite()) [[unlikely]]
        {
            std::cerr << "Warning: Non-finite gradients detected in MSE backward pass\n";
            // Return zero gradient to prevent further issues
            return nn::Tensor{Eigen::MatrixXf::Zero(last_input.get_data_ref().rows(),
                                                    last_input.get_data_ref().cols())};
        }

        // Gradient clipping to prevent explosion
        float grad_norm = grad.norm();
        if (grad_norm > MAX_GRADIENT_NORM) [[unlikely]]
        {
            grad.multiply_scalar(MAX_GRADIENT_NORM / grad_norm);
        }

        return grad;
    }
};

#endif // MSELOSS_HPP
