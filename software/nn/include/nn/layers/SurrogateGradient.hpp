#ifndef SURROGATE_GRADIENT_HPP
#define SURROGATE_GRADIENT_HPP

#include <cmath>

#include "nn/tensor/Tensor.hpp"

// Interface for surrogate gradient functions
class ISurrogateGradient
{
   public:
    virtual ~ISurrogateGradient() = default;

    [[nodiscard]] virtual auto calculate(const nn::Tensor& v_mem_pre_spike,
                                         float voltage_threshold) const -> nn::Tensor = 0;

    [[nodiscard]] virtual auto calculate_scalar(float v_mem_pre_spike,
                                                float voltage_threshold) const -> float = 0;
};

// Exponential / SuperSpike surrogate gradient
class ExponentialSurrogate : public ISurrogateGradient
{
   public:
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
