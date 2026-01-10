#ifndef SURROGATE_GRADIENT_HPP
#define SURROGATE_GRADIENT_HPP

#include <cmath>

#include "nn/tensor/Tensor.hpp"

/**
 * @file SurrogateGradient.hpp
 * @brief Surrogate gradient functions for spiking neurons.
 *
 * Background:
 * - A spiking neuron typically emits spikes via a hard threshold: S = Θ(V - V_th).
 * - The true derivative dS/dV is zero almost everywhere and undefined at the threshold.
 * - To train spiking networks with gradient descent, we keep the *forward* spike rule,
 *   but replace dS/dV with a smooth approximation during the *backward* pass.
 *
 * In this project, spiking layers (e.g., `LeakyBPTT`) call these functions to compute
 * a differentiable proxy for dS/dV.
 */

// Interface for surrogate gradient functions
class ISurrogateGradient
{
   public:
    virtual ~ISurrogateGradient() = default;

    /**
     * @brief Vector form: compute surrogate derivative dS/dV for each element.
     *
     * @param v_mem_pre_spike Membrane potential before applying threshold/reset.
     * @param voltage_threshold Spike threshold V_th.
     * @return Tensor of same shape as v_mem_pre_spike containing surrogate derivatives.
     */
    [[nodiscard]] virtual auto calculate(const nn::Tensor& v_mem_pre_spike,
                                         float voltage_threshold) const -> nn::Tensor = 0;

    /**
     * @brief Scalar form of `calculate` for single values.
     *
     * This is useful for tight loops where allocating a full Tensor is unnecessary.
     */
    [[nodiscard]] virtual auto calculate_scalar(float v_mem_pre_spike,
                                                float voltage_threshold) const -> float = 0;
};

// Exponential / SuperSpike surrogate gradient
class ExponentialSurrogate : public ISurrogateGradient
{
   public:
    /**
     * @param sharpness Controls the width/scale of the surrogate around threshold.
     *        Smaller values make the surrogate more sharply peaked near V_th.
     *        Larger values make gradients spread over a wider voltage range.
     */
    explicit ExponentialSurrogate(float sharpness = 1.0F) : sharpness_(sharpness) {}

    [[nodiscard]] auto calculate(const nn::Tensor& v_mem_pre_spike, float voltage_threshold) const
        -> nn::Tensor override
    {
        // Compute element-wise: (1/sharpness) * exp(-|v - threshold| / sharpness)
        nn::Tensor result(v_mem_pre_spike.rows(), v_mem_pre_spike.cols());
        for (size_t i = 0; i < v_mem_pre_spike.rows(); ++i)
        {
            for (size_t j = 0; j < v_mem_pre_spike.cols(); ++j)
            {
                result.at(i, j) = calculate_scalar(v_mem_pre_spike.at(i, j), voltage_threshold);
            }
        }
        return result;
    }

    [[nodiscard]] auto calculate_scalar(float v_mem_pre_spike, float voltage_threshold) const
        -> float override
    {
        float diff_abs = std::abs(v_mem_pre_spike - voltage_threshold);
        return (1.0F / sharpness_) * std::exp(-diff_abs / sharpness_);
    }

   private:
    float sharpness_;
};

// Boxcar surrogate gradient
class BoxcarSurrogate : public ISurrogateGradient
{
   public:
    /**
     * @param window Width of the non-zero region around the threshold.
     *        The derivative is 1.0 when |V - V_th| < window/2, else 0.0.
     */
    explicit BoxcarSurrogate(float window = 0.5F) : window_(window) {}

    [[nodiscard]] auto calculate(const nn::Tensor& v_mem_pre_spike, float voltage_threshold) const
        -> nn::Tensor override
    {
        // Boxcar: return 1.0 if |v - threshold| < window/2, else 0.0
        nn::Tensor result(v_mem_pre_spike.rows(), v_mem_pre_spike.cols());
        for (size_t i = 0; i < v_mem_pre_spike.rows(); ++i)
        {
            for (size_t j = 0; j < v_mem_pre_spike.cols(); ++j)
            {
                result.at(i, j) = calculate_scalar(v_mem_pre_spike.at(i, j), voltage_threshold);
            }
        }
        return result;
    }

    [[nodiscard]] auto calculate_scalar(float v_mem_pre_spike, float voltage_threshold) const
        -> float override
    {
        float half_window = window_ / 2.0F;
        float diff_abs = std::abs(v_mem_pre_spike - voltage_threshold);
        return (diff_abs < half_window) ? 1.0F : 0.0F;
    }

   private:
    float window_;
};

#endif // SURROGATE_GRADIENT_HPP
