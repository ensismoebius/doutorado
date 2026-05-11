#ifndef EXPONENTIAL_SURROGATE_HPP
#define EXPONENTIAL_SURROGATE_HPP

#include <cmath>
#include <stdexcept>

#include "nn/layers/spiking/ISurrogateGradient.hpp"

class ExponentialSurrogate : public ISurrogateGradient
{
   public:
    explicit ExponentialSurrogate(float sharpness = 1.0F) : sharpness_(sharpness)
    {
        if (sharpness_ <= 0.0F)
        {
            throw std::invalid_argument("ExponentialSurrogate sharpness must be > 0");
        }
    }

    [[nodiscard]] auto calculate(const nn::Tensor& v_mem_pre_spike, float voltage_threshold) const
        -> nn::Tensor override
    {
        nn::Tensor result(v_mem_pre_spike.rows(), v_mem_pre_spike.cols());
        for (size_t i = 0; i < v_mem_pre_spike.rows(); ++i)
        {
            for (size_t j = 0; j < v_mem_pre_spike.cols(); ++j)
            {
                result.at(i, j) = calculate_scalar(v_mem_pre_spike.at(i, j), voltage_threshold);
            }
        }
        return result;
    } // LCOV_EXCL_LINE

    [[nodiscard]] auto calculate_scalar(float v_mem_pre_spike, float voltage_threshold) const
        -> float override
    {
        float diff_abs = std::abs(v_mem_pre_spike - voltage_threshold);
        return (1.0F / sharpness_) * std::exp(-diff_abs / sharpness_);
    }

    [[nodiscard]] float sharpness() const
    {
        return sharpness_;
    }

   private:
    float sharpness_;
};

#endif // EXPONENTIAL_SURROGATE_HPP
