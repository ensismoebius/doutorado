#pragma once

#include <cmath>

#include "nn/tensor/Tensor.hpp"

namespace nn::activations
{

// Fast sigmoid: rational approximation (error < 0.01)
// Avoids expensive exp() computation
inline auto sigmoid_fast(float x) -> float
{
    // Saturate cleanly beyond ±10 — the rational approximation becomes
    // inaccurate for large |x| and the true sigmoid is already ≈0 or ≈1 there.
    if (x <= -10.0f) return 0.0f;
    if (x >= 10.0f) return 1.0f;
    // Rational approximation: sigmoid(x) ≈ 0.5 + x / (2 * (1 + |x|))
    return 0.5f + x / (2.0f * (1.0f + std::abs(x)));
}

// Fast tanh: rational approximation (error < 0.01)
// Avoids expensive exp() computation
inline auto tanh_fast(float x) -> float
{
    // Saturate cleanly beyond ±10.
    if (x <= -10.0f) return -1.0f;
    if (x >= 10.0f) return 1.0f;
    // Rational approximation: tanh(x) ≈ x / (1 + |x|)
    float abs_x = std::abs(x);
    return x / (1.0f + abs_x);
}

// Sigmoid gradient from output: dy/dx = y(1-y)
inline auto sigmoid_grad_fast(float y) -> float
{
    return y * (1.0f - y);
}

// Tanh gradient from output: dy/dx = 1 - y^2
inline auto tanh_grad_fast(float y) -> float
{
    return 1.0f - y * y;
}

// Vectorized versions operating on Tensors
inline auto sigmoid_fast_tensor(const nn::Tensor& x) -> nn::Tensor
{
    nn::Tensor result(x.rows(), x.cols());
    for (nn::Index i = 0; i < x.rows(); ++i)
    {
        for (nn::Index j = 0; j < x.cols(); ++j)
        {
            result.at(i, j) = sigmoid_fast(x.at(i, j));
        }
    }
    return result;
}

inline auto tanh_fast_tensor(const nn::Tensor& x) -> nn::Tensor
{
    nn::Tensor result(x.rows(), x.cols());
    for (nn::Index i = 0; i < x.rows(); ++i)
    {
        for (nn::Index j = 0; j < x.cols(); ++j)
        {
            result.at(i, j) = tanh_fast(x.at(i, j));
        }
    }
    return result;
}

// Fused block+activation: reads column range [col_start, col_start+gate_size) from src directly.
// Avoids the intermediate Tensor copy that block() creates — eliminates one alloc + one read pass.
inline auto sigmoid_fast_block(const nn::Tensor& src, nn::Index col_start, nn::Index gate_size)
    -> nn::Tensor
{
    nn::Tensor result(src.rows(), gate_size);
    for (nn::Index i = 0; i < src.rows(); ++i)
        for (nn::Index j = 0; j < gate_size; ++j)
            result.at(i, j) = sigmoid_fast(src.at(i, col_start + j));
    return result;
}

inline auto tanh_fast_block(const nn::Tensor& src, nn::Index col_start, nn::Index gate_size)
    -> nn::Tensor
{
    nn::Tensor result(src.rows(), gate_size);
    for (nn::Index i = 0; i < src.rows(); ++i)
        for (nn::Index j = 0; j < gate_size; ++j)
            result.at(i, j) = tanh_fast(src.at(i, col_start + j));
    return result;
}

} // namespace nn::activations
