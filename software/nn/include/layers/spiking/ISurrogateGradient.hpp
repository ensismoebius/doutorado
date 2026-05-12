#ifndef I_SURROGATE_GRADIENT_HPP
#define I_SURROGATE_GRADIENT_HPP

#include "tensor/Tensor.hpp"

class ISurrogateGradient
{
   public:
    virtual ~ISurrogateGradient() = default;

    [[nodiscard]] virtual auto calculate(
        const nn::Tensor& v_mem_pre_spike, float voltage_threshold) const -> nn::Tensor = 0;

    [[nodiscard]] virtual auto calculate_scalar(
        float v_mem_pre_spike, float voltage_threshold) const -> float = 0;
};

#endif // I_SURROGATE_GRADIENT_HPP
