#ifndef LEAKYRELU_HPP
#define LEAKYRELU_HPP

#include "nn/layers/base/Module.hpp"

/**
 * @file LeakyReLU.hpp
 * @brief LeakyReLU activation (dense, non-spiking).
 *
 * Contract:
 * - `forward(requires_grad=true)` caches a per-element slope mask for `backward()`.
 * - `backward()` assumes `forward()` was called on the same instance and shape.
 * - This layer is *stateless* besides the cached mask.
 */

template <typename Backend>
struct LeakyReLUImpl : public Module<Backend>
{
    /// Tensor type for the active compute backend.
    using Tensor = typename Module<Backend>::Tensor;
    float alpha; // negative slope
    Tensor leaky_grad;

    explicit LeakyReLUImpl(float alpha_ = 0.01F) : alpha(alpha_) {}

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        if (requires_grad)
        {
            Tensor binary_mask = input > 0.0f;
            binary_mask.multiply_scalar_inplace(1.0f - alpha);
            leaky_grad = binary_mask;
            leaky_grad.add_scalar_inplace(alpha);
        }
        //
        return input.leaky_relu(alpha);
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    { //
        // Element-wise multiplication of grad_output with leaky_grad mask
        auto grad_input = grad_output.multiply(leaky_grad);
        return grad_input;
    }
};
//
#endif // LEAKYRELU_HPP //
