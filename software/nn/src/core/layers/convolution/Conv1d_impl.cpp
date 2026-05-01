/**
 * @file Conv1d_impl.cpp
 * @brief Implementation of the 1D convolution layer (im2col / col2im approach).
 *
 * Input:  (B, C_in, L)
 * Output: (B, C_out, L_out)   L_out = (L + 2P - D*(K-1) - 1) / S + 1
 *
 * Weights shape: (C_in * K, C_out) — each column is one output filter flattened
 * over (input_channel, kernel_position).
 *
 * Forward per batch item b:
 *   col  (L_out, C_in*K) : im2col unrolling of input[b]
 *   out  (L_out, C_out)  : col @ weights_ + bias_broadcast
 *   output[b, oc, lo]    = out[lo, oc]
 *
 * Backward per batch item b:
 *   d_out    (L_out, C_out) : grad_output[b, :, :].T
 *   d_weights                += col.T @ d_out
 *   d_bias                   += colwise sum of d_out
 *   d_col   (L_out, C_in*K) : d_out @ weights_.T   (col2im routing)
 *   dx[b]                    : scatter d_col back via im2col inverse
 */

#include <cmath>
#include <random>
#include <stdexcept>

#include "nn/Backend.hpp"
#include "nn/layers/convolution/Conv1d.hpp"

// ============ Constructor ============

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
      // Weights: (C_in * K, C_out); bias: (1, C_out)
      weights_(Tensor(static_cast<nn::Index>(in_channels * kernel_size),
                      static_cast<nn::Index>(out_channels))),
      bias_(Tensor(1, static_cast<nn::Index>(out_channels)))
{
    initialize_weights_he();
}

// ============ Forward ============

template <typename Backend>
auto Conv1dImpl<Backend>::forward(const typename Conv1dImpl<Backend>::Tensor& input,
    bool requires_grad) -> typename Conv1dImpl<Backend>::Tensor
{
    const auto shape = input.get_shape();

    if (shape.size() != 3)
        throw std::invalid_argument("Conv1d: input must be 3-D (B, C_in, L)");

    const int B   = static_cast<int>(shape[0]);
    const int C_in = static_cast<int>(shape[1]);
    const int L   = static_cast<int>(shape[2]);

    if (C_in != in_channels_)
        throw std::invalid_argument("Conv1d: input channels mismatch");

    const int L_out = compute_output_length(L);
    if (L_out <= 0)
        throw std::invalid_argument("Conv1d: output length <= 0");

    const int K = kernel_size_, S = stride_, P = padding_, D = dilation_;
    const int C_out = out_channels_;

    if (requires_grad) input_cache_ = input;

    Tensor output(static_cast<nn::Index>(B),
                  static_cast<nn::Index>(C_out),
                  static_cast<nn::Index>(L_out));

    for (int b = 0; b < B; ++b)
    {
        // im2col: (L_out, C_in * K)
        Tensor col(static_cast<nn::Index>(L_out),
                   static_cast<nn::Index>(C_in * K));

        for (int lo = 0; lo < L_out; ++lo)
        {
            for (int ic = 0; ic < C_in; ++ic)
            {
                for (int k = 0; k < K; ++k)
                {
                    int li = lo * S - P + k * D;
                    float v = (li >= 0 && li < L)
                        ? input.at(static_cast<nn::Index>(b),
                                   static_cast<nn::Index>(ic),
                                   static_cast<nn::Index>(li))
                        : 0.0f;
                    col.at(static_cast<nn::Index>(lo),
                           static_cast<nn::Index>(ic * K + k)) = v;
                }
            }
        }

        // (L_out, C_in*K) @ (C_in*K, C_out) = (L_out, C_out)
        Tensor out_2d = col.matmul(weights_);
        // Add bias (1, C_out) to every row
        out_2d = out_2d.add_row_broadcast(bias_);

        // Write: out_2d[lo, oc] → output[b, oc, lo]
        for (int oc = 0; oc < C_out; ++oc)
            for (int lo = 0; lo < L_out; ++lo)
                output.at(static_cast<nn::Index>(b),
                           static_cast<nn::Index>(oc),
                           static_cast<nn::Index>(lo)) =
                    out_2d.at(static_cast<nn::Index>(lo),
                               static_cast<nn::Index>(oc));
    }

    return output;
}

// ============ Backward ============

template <typename Backend>
auto Conv1dImpl<Backend>::backward(
    const typename Conv1dImpl<Backend>::Tensor& grad_output)
    -> typename Conv1dImpl<Backend>::Tensor
{
    const auto shape = input_cache_.get_shape();
    const int B    = static_cast<int>(shape[0]);
    const int C_in = static_cast<int>(shape[1]);
    const int L    = static_cast<int>(shape[2]);
    const int L_out = compute_output_length(L);

    const int K = kernel_size_, S = stride_, P = padding_, D = dilation_;
    const int C_out = out_channels_;

    Tensor dx(static_cast<nn::Index>(B),
              static_cast<nn::Index>(C_in),
              static_cast<nn::Index>(L));
    dx.setZero();

    Tensor d_weights(weights_.rows(), weights_.cols());
    d_weights.setZero();

    Tensor d_bias(1, static_cast<nn::Index>(C_out));
    d_bias.setZero();

    for (int b = 0; b < B; ++b)
    {
        // Rebuild im2col for this batch item
        Tensor col(static_cast<nn::Index>(L_out),
                   static_cast<nn::Index>(C_in * K));

        for (int lo = 0; lo < L_out; ++lo)
            for (int ic = 0; ic < C_in; ++ic)
                for (int k = 0; k < K; ++k)
                {
                    int li = lo * S - P + k * D;
                    float v = (li >= 0 && li < L)
                        ? input_cache_.at(static_cast<nn::Index>(b),
                                          static_cast<nn::Index>(ic),
                                          static_cast<nn::Index>(li))
                        : 0.0f;
                    col.at(static_cast<nn::Index>(lo),
                           static_cast<nn::Index>(ic * K + k)) = v;
                }

        // Gather d_out: (L_out, C_out) from grad_output[b, oc, lo]
        Tensor d_out(static_cast<nn::Index>(L_out),
                     static_cast<nn::Index>(C_out));
        for (int oc = 0; oc < C_out; ++oc)
            for (int lo = 0; lo < L_out; ++lo)
                d_out.at(static_cast<nn::Index>(lo),
                          static_cast<nn::Index>(oc)) =
                    grad_output.at(static_cast<nn::Index>(b),
                                    static_cast<nn::Index>(oc),
                                    static_cast<nn::Index>(lo));

        // d_weights += col.T @ d_out : (C_in*K, L_out) @ (L_out, C_out) = (C_in*K, C_out)
        d_weights.add_inplace(col.transpose().matmul(d_out));

        // d_bias += column sums of d_out: (1, C_out)
        for (int oc = 0; oc < C_out; ++oc)
        {
            float s = 0.0f;
            for (int lo = 0; lo < L_out; ++lo)
                s += d_out.at(static_cast<nn::Index>(lo), static_cast<nn::Index>(oc));
            d_bias.at(0, static_cast<nn::Index>(oc)) += s;
        }

        // d_col = d_out @ weights_.T : (L_out, C_out) @ (C_out, C_in*K) = (L_out, C_in*K)
        Tensor d_col = d_out.matmul(weights_.transpose());

        // col2im: scatter d_col back to dx[b]
        for (int lo = 0; lo < L_out; ++lo)
            for (int ic = 0; ic < C_in; ++ic)
                for (int k = 0; k < K; ++k)
                {
                    int li = lo * S - P + k * D;
                    if (li >= 0 && li < L)
                        dx.at(static_cast<nn::Index>(b),
                               static_cast<nn::Index>(ic),
                               static_cast<nn::Index>(li)) +=
                            d_col.at(static_cast<nn::Index>(lo),
                                      static_cast<nn::Index>(ic * K + k));
                }
    }

    weights_.set_grad(d_weights);
    bias_.set_grad(d_bias);

    return dx;
}

// ============ Getters / Setters ============

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

// ============ Private helpers ============

template <typename Backend>
void Conv1dImpl<Backend>::initialize_weights_he()
{
    const float fan_in = static_cast<float>(in_channels_ * kernel_size_);
    const float std_val = std::sqrt(2.0f / fan_in);

    std::mt19937 gen(42u);
    std::normal_distribution<float> dist(0.0f, std_val);

    for (nn::Index i = 0; i < static_cast<nn::Index>(weights_.size()); ++i)
        weights_.at(i) = dist(gen);

    for (nn::Index i = 0; i < static_cast<nn::Index>(bias_.size()); ++i)
        bias_.at(i) = 0.0f;
}

template <typename Backend>
auto Conv1dImpl<Backend>::compute_output_length(int input_length) const -> int
{
    return (input_length + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
}

template class Conv1dImpl<nn::Backend>;
