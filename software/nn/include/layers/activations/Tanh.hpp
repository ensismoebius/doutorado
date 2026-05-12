#ifndef NN_LAYERS_ACTIVATIONS_TANH_HPP
#define NN_LAYERS_ACTIVATIONS_TANH_HPP

#include "layers/activations/Sigmoid.hpp"
#include "layers/base/Module.hpp"
#include "tensor/Tensor.hpp"

namespace nn::activation {

// Free functions on CPU-resident nn::Tensor.

inline auto tanh(const nn::Tensor& x) -> nn::Tensor
{
    return sigmoid(x * 2.0f) * 2.0f - 1.0f;
}

inline auto tanh_grad(const nn::Tensor& tanh_out) -> nn::Tensor
{
    const nn::Tensor ones = nn::Tensor::ones(tanh_out.rows(), tanh_out.cols());
    return ones - (tanh_out * tanh_out);
}

} // namespace nn::activation

// Module for use in Sequential and similar composable models.
template <typename Backend>
struct TanhImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

    Tensor tanh_out_;

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        // tanh(x) = 2*sigmoid(2x) - 1
        const Tensor ones = Tensor::ones(input.rows(), input.cols());
        const Tensor two  = ones + ones;
        const Tensor sig  = ones.divide(ones + (input * -2.0f).exp());
        Tensor out = sig * two - ones;
        if (requires_grad)
            tanh_out_ = out;
        return out;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        const Tensor ones = Tensor::ones(tanh_out_.rows(), tanh_out_.cols());
        return grad_output.multiply(ones - tanh_out_ * tanh_out_);
    }
};

#endif // NN_LAYERS_ACTIVATIONS_TANH_HPP
