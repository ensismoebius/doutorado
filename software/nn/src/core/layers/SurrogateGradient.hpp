#ifndef SURROGATE_GRADIENT_HPP
#define SURROGATE_GRADIENT_HPP

#include "../tensor/Tensor.hpp"

// Interface for surrogate gradient functions
class ISurrogateGradient
{
   public:
    virtual ~ISurrogateGradient() = default;

    [[nodiscard]] virtual auto calculate(const nn::Tensor& v_mem_pre_spike,
                                         float voltage_threshold) const -> nn::Tensor = 0;
};

// Exponential / SuperSpike surrogate gradient
class ExponentialSurrogate : public ISurrogateGradient
{
   public:
    explicit ExponentialSurrogate(float sharpness = 1.0F) : sharpness_(sharpness) {}

    [[nodiscard]] auto calculate(const nn::Tensor& v_mem_pre_spike, float voltage_threshold) const
        -> nn::Tensor override
    {
        const auto& v_mem = v_mem_pre_spike.get_data_ref();
        const auto diff_abs = (v_mem.array() - voltage_threshold).abs();

        nn::Tensor result(v_mem.rows(), v_mem.cols());
        result.get_data_ref() = (1.0F / sharpness_) * ((-diff_abs / sharpness_).array().exp());
        return result;
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
        const auto& v_mem = v_mem_pre_spike.get_data_ref();
        const auto diff_abs = (v_mem.array() - voltage_threshold).abs();

        nn::Tensor result(v_mem.rows(), v_mem.cols());
        result.get_data_ref() = (diff_abs.array() < (window_ / 2.0F)).cast<float>();
        return result;
    }

   private:
    float window_;
};

#endif // SURROGATE_GRADIENT_HPP
