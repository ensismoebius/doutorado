#ifndef BOXCAR_SURROGATE_HPP
#define BOXCAR_SURROGATE_HPP

#include <cmath>
#include <stdexcept>

#include "nn/layers/spiking/ISurrogateGradient.hpp"

class BoxcarSurrogate : public ISurrogateGradient
{
   public:
    explicit BoxcarSurrogate(float window = 0.5F) : window_(window)
    {
        if (window_ <= 0.0F)
        {
            throw std::invalid_argument("BoxcarSurrogate window must be > 0");
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
        float half_window = window_ / 2.0F;
        float diff_abs = std::abs(v_mem_pre_spike - voltage_threshold);
        return (diff_abs < half_window) ? 1.0F : 0.0F;
    }

    [[nodiscard]] float width() const
    {
        return window_;
    }

   private:
    float window_;
};

#endif // BOXCAR_SURROGATE_HPP
