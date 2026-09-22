#ifndef ARCTAN_SURROGATE_HPP
#define ARCTAN_SURROGATE_HPP

#include <cmath>
#include <numbers>
#include <stdexcept>

#include "layers/spiking/ISurrogateGradient.hpp"

// Derivative of S(U) = (1/pi)*arctan(pi*alpha*U/2) + 0.5 w.r.t. U, i.e.
// alpha / (2*(1 + (pi*alpha*U/2)^2)). Matches snnTorch's ATan surrogate
// (its default spike-gradient estimator since 2023) — chosen over the
// project's exponential/boxcar surrogates because arctan's heavier tails
// keep gradient flowing further from threshold, avoiding the saturation
// fast-sigmoid-style surrogates show on this task.
class ArcTanSurrogate : public ISurrogateGradient
{
   public:
    explicit ArcTanSurrogate(float alpha = 2.0F) : alpha_(alpha)
    {
        if (alpha_ <= 0.0F)
        {
            throw std::invalid_argument("ArcTanSurrogate alpha must be > 0");
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
    }

    [[nodiscard]] auto calculate_scalar(float v_mem_pre_spike, float voltage_threshold) const
        -> float override
    {
        const float diff = v_mem_pre_spike - voltage_threshold;
        const float x = (std::numbers::pi_v<float> / 2.0F) * alpha_ * diff;
        return alpha_ / (2.0F * (1.0F + x * x));
    }

    [[nodiscard]] float alpha() const
    {
        return alpha_;
    }

   private:
    float alpha_;
};

#endif // ARCTAN_SURROGATE_HPP
