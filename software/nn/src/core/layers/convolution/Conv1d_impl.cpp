/**
 * @file Conv1d_impl.cpp
 * @brief Implementation of the Conv1d layer.
 *
 * NOTE: Currently a no-op placeholder. The nn::Tensor type only supports 2D and 4D
 * tensors natively (via Eigen matrices). A true 1D convolution requires either:
 * 1. Adding 3D tensor support to the Tensor class
 * 2. Reshaping 3D input to 2D, applying 1D conv, then reshaping back
 *
 * This implementation validates input but returns it unchanged to allow
 * code that uses Conv1d to compile and run.
 */

#include <cmath>
#include <random>

#include "nn/Backend.hpp"
#include "nn/layers/convolution/Conv1d.hpp"

template <typename Backend>
Conv1dImpl<Backend>::Conv1dImpl(int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    int dilation)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding),
      dilation_(dilation),
      weights_(nn::Tensor(static_cast<size_t>(kernel_size) * in_channels, out_channels)),
      bias_(nn::Tensor(1, out_channels))
{
    initialize_weights_he();
}

template <typename Backend>
auto Conv1dImpl<Backend>::forward(const typename Conv1dImpl<Backend>::Tensor& input,
    bool requires_grad) -> typename Conv1dImpl<Backend>::Tensor
{
    const auto shape = input.get_shape();

    if (shape.size() != 3) [[unlikely]]
    {
        NN_LOG_WARN("Conv1d: input is not 3-D (no-op)");
        return input;
    }

    const int in_ch = static_cast<int>(shape[1]);
    if (in_ch != in_channels_)
    {
        NN_LOG_WARN("Conv1d: input channels mismatch (no-op)");
        return input;
    }

    if (requires_grad)
    {
        input_cache_ = input;
    }

    (void) requires_grad;

    return input;
}

template <typename Backend>
auto Conv1dImpl<Backend>::backward(
    const typename Conv1dImpl<Backend>::Tensor& grad_output)
    -> typename Conv1dImpl<Backend>::Tensor
{
    return grad_output;
}

template <typename Backend>
auto Conv1dImpl<Backend>::get_weights() const
    -> const typename Conv1dImpl<Backend>::Tensor&
{
    return weights_;
}

template <typename Backend>
auto Conv1dImpl<Backend>::get_weights() -> typename Conv1dImpl<Backend>::Tensor&
{
    return weights_;
}

template <typename Backend>
const typename Conv1dImpl<Backend>::Tensor& Conv1dImpl<Backend>::get_bias() const
{
    return bias_;
}

template <typename Backend>
auto Conv1dImpl<Backend>::get_bias() -> typename Conv1dImpl<Backend>::Tensor&
{
    return bias_;
}

template <typename Backend>
void Conv1dImpl<Backend>::set_weights(const typename Conv1dImpl<Backend>::Tensor& weights)
{
    weights_ = weights;
}

template <typename Backend>
void Conv1dImpl<Backend>::set_bias(const typename Conv1dImpl<Backend>::Tensor& bias)
{
    bias_ = bias;
}

template <typename Backend>
void Conv1dImpl<Backend>::initialize_weights_he()
{
    const float fan_in = static_cast<float>(in_channels_ * kernel_size_);
    const float std = std::sqrt(2.0f / fan_in);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std);

    for (size_t i = 0; i < weights_.size(); ++i)
    {
        weights_.at(i) = dist(gen);
    }

    for (size_t i = 0; i < bias_.size(); ++i)
    {
        bias_.at(i) = 0.0f;
    }
}

template <typename Backend>
auto Conv1dImpl<Backend>::compute_output_length(int input_length) const -> int
{
    return (input_length + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
}

template class Conv1dImpl<nn::Backend>;