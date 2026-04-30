#ifndef NN_LAYERS_ACTIVATIONS_SIGMOID_HPP
#define NN_LAYERS_ACTIVATIONS_SIGMOID_HPP

#include <cmath>

#include "nn/layers/base/Module.hpp"
#include "nn/tensor/Tensor.hpp"

namespace nn::activation {

// Scalar overload — used by layers that compute sigmoid element-wise in loops.
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float sigmoid_grad(float sigmoid_out) { return sigmoid_out * (1.0f - sigmoid_out); }

// Tensor overloads — used by LSTM and any layer with xtensor-only internal compute.
inline auto sigmoid(const nn::Tensor& x) -> nn::Tensor
{
    const nn::Tensor ones = nn::Tensor::ones(x.rows(), x.cols());
    return ones.divide(ones + (x * -1.0f).exp());
}

inline auto sigmoid_grad(const nn::Tensor& sigmoid_out) -> nn::Tensor
{
    const nn::Tensor ones = nn::Tensor::ones(sigmoid_out.rows(), sigmoid_out.cols());
    return sigmoid_out * (ones - sigmoid_out);
}

} // namespace nn::activation

// Module for use in Sequential and similar composable models.
template <typename Backend>
struct SigmoidImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

    Tensor sigmoid_out_;

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        const Tensor ones = Tensor::ones(input.rows(), input.cols());
        Tensor out = ones.divide(ones + (input * -1.0f).exp());
        if (requires_grad)
            sigmoid_out_ = out;
        return out;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        const Tensor ones = Tensor::ones(sigmoid_out_.rows(), sigmoid_out_.cols());
        return grad_output.multiply(sigmoid_out_ * (ones - sigmoid_out_));
    }
};

#endif // NN_LAYERS_ACTIVATIONS_SIGMOID_HPP
