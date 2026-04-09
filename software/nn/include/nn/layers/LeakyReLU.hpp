#ifndef LEAKYRELU_HPP
#define LEAKYRELU_HPP

#include "nn/layers/Module.hpp"
#include "nn/tensor/Tensor.hpp"

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
        // Cache the gradient for the backward pass only if gradients required
        if (requires_grad)
        {
            // Create gradient mask: 1 for positive values, alpha for negative values
            leaky_grad = Tensor(input.rows(), input.cols());
            for (size_t i = 0; i < input.rows(); ++i)
            {
                for (size_t j = 0; j < input.cols(); ++j)
                {
                    leaky_grad.at(i, j) = (input.at(i, j) > 0.0f) ? 1.0f : alpha;
                }
            }
        }

        // Apply the LeakyReLU activation function using Tensor method
        return input.leaky_relu(alpha);
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        // Element-wise multiplication of grad_output with leaky_grad mask
        auto grad_input = grad_output.multiply(leaky_grad);
        return grad_input;
    }
};

#endif // LEAKYRELU_HPP
