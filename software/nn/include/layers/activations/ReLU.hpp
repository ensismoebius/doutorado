#ifndef RELU_HPP
#define RELU_HPP

#include "layers/base/Module.hpp"
#include "tensor/Tensor.hpp"

/**
 * @file ReLU.hpp
 * @brief ReLU activation (dense, non-spiking).
 *
 * Notes:
 * - This is the classic piecewise-linear ReLU: $\max(0, x)$.
 * - `forward(requires_grad=true)` caches a 0/1 mask so `backward()` can multiply
 *   incoming gradients elementwise.
 * - Like the other small activations in this repo, it assumes a 2D tensor layout
 *   (rows x cols) and does not manage a computation graph.
 *
 * Backend polymorphism:
 * - `ReLUImpl<Backend>` works with any backend tensor type that implements `relu()`,
 *   `operator>`, and `multiply()`.
 * - The convenience alias `nn::ReLU = ReLUImpl<Backend>` is declared in
 *   `include/layers/Layers.hpp`.
 */
template <typename Backend>
struct ReLUImpl : public Module<Backend>
{
    using Tensor = nn::TensorImpl<Backend>;

    Tensor relu_grad; // cached mask used in the backward pass

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        if (requires_grad)
        {
            // Build gradient mask with backend-vectorized compare.
            relu_grad = input > 0.0f;
        }
        return input.relu();
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        return grad_output.multiply(relu_grad);
    }
};

#endif // RELU_HPP