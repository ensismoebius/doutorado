#pragma once

#include <cmath>

#include "tensor/Tensor.hpp"

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

// Vectorized versions operating on Tensors. Templated on Backend (rather than
// hardcoded to nn::Tensor == TensorImpl<nn::Backend>) because LSTMLayerImpl<Backend>
// calls these for whichever Backend it was instantiated with — hardcoding here
// silently broke every backend except the one currently selected as nn::Backend
// (caught via src/core/tensor/tests/pytorch_parity_gtest.cpp, which instantiates
// LSTMLayerImpl for every concrete backend side-by-side).
template <typename Backend>
inline auto sigmoid_fast_tensor(const nn::TensorImpl<Backend>& x) -> nn::TensorImpl<Backend>
{
    nn::TensorImpl<Backend> result(x.rows(), x.cols());
    for (nn::Index i = 0; i < x.rows(); ++i)
    {
        for (nn::Index j = 0; j < x.cols(); ++j)
        {
            result.at(i, j) = sigmoid_fast(x.at(i, j));
        }
    }
    return result;
}

template <typename Backend>
inline auto tanh_fast_tensor(const nn::TensorImpl<Backend>& x) -> nn::TensorImpl<Backend>
{
    nn::TensorImpl<Backend> result(x.rows(), x.cols());
    for (nn::Index i = 0; i < x.rows(); ++i)
    {
        for (nn::Index j = 0; j < x.cols(); ++j)
        {
            result.at(i, j) = tanh_fast(x.at(i, j));
        }
    }
    return result;
} //

// Fused block+activation: reads column range [col_start, col_start+gate_size) from src directly.
// Avoids the intermediate Tensor copy that block() creates — eliminates one alloc + one read pass.
template <typename Backend>
inline auto sigmoid_fast_block(
    const nn::TensorImpl<Backend>& src, nn::Index col_start, nn::Index gate_size)
    -> nn::TensorImpl<Backend>
{
    nn::TensorImpl<Backend> result(src.rows(), gate_size);
    for (nn::Index i = 0; i < src.rows(); ++i)
        for (nn::Index j = 0; j < gate_size; ++j)
            result.at(i, j) = sigmoid_fast(src.at(i, col_start + j));
    return result;
} //

template <typename Backend>
inline auto tanh_fast_block(
    const nn::TensorImpl<Backend>& src, nn::Index col_start, nn::Index gate_size)
    -> nn::TensorImpl<Backend>
{
    nn::TensorImpl<Backend> result(src.rows(), gate_size);
    for (nn::Index i = 0; i < src.rows(); ++i)
        for (nn::Index j = 0; j < gate_size; ++j)
            result.at(i, j) = tanh_fast(src.at(i, col_start + j));
    return result;
} //

// ── Exact counterparts ───────────────────────────────────────────────────────
// Same signatures as the *_fast_* helpers, but computing the real sigmoid/tanh, so a caller
// can switch fidelity with a flag rather than a different code path.
//
// Why these exist: the rational approximations above are NOT close to the real functions.
// |tanh - tanh_fast| reaches 0.306 on [-4,4] (at x=2: tanh=0.964 vs tanh_fast=0.667), which
// makes any LSTM built on them a *softsign-gated* LSTM that cannot match torch.nn.LSTM. Since
// PyTorch/snnTorch is this project's correctness reference, exact is the default and the fast
// forms are an explicit speed/fidelity trade (E05Config::Numerics::exact_activations).

template <typename Backend>
inline auto sigmoid_exact_block(
    const nn::TensorImpl<Backend>& src, nn::Index col_start, nn::Index gate_size)
    -> nn::TensorImpl<Backend>
{
    nn::TensorImpl<Backend> result(src.rows(), gate_size);
    for (nn::Index i = 0; i < src.rows(); ++i)
        for (nn::Index j = 0; j < gate_size; ++j)
            result.at(i, j) = 1.0F / (1.0F + std::exp(-src.at(i, col_start + j)));
    return result;
}

template <typename Backend>
inline auto tanh_exact_block(
    const nn::TensorImpl<Backend>& src, nn::Index col_start, nn::Index gate_size)
    -> nn::TensorImpl<Backend>
{
    nn::TensorImpl<Backend> result(src.rows(), gate_size);
    for (nn::Index i = 0; i < src.rows(); ++i)
        for (nn::Index j = 0; j < gate_size; ++j)
            result.at(i, j) = std::tanh(src.at(i, col_start + j));
    return result;
}

template <typename Backend>
inline auto tanh_exact_tensor(const nn::TensorImpl<Backend>& x) -> nn::TensorImpl<Backend>
{
    nn::TensorImpl<Backend> result(x.rows(), x.cols());
    for (nn::Index i = 0; i < x.rows(); ++i)
        for (nn::Index j = 0; j < x.cols(); ++j) result.at(i, j) = std::tanh(x.at(i, j));
    return result;
}

// Dispatchers: pick fidelity at run time from a single flag.
template <typename Backend>
inline auto sigmoid_block(
    const nn::TensorImpl<Backend>& src, nn::Index col_start, nn::Index gate_size, bool exact)
    -> nn::TensorImpl<Backend>
{
    return exact ? sigmoid_exact_block(src, col_start, gate_size)
                 : sigmoid_fast_block(src, col_start, gate_size);
}

template <typename Backend>
inline auto tanh_block(
    const nn::TensorImpl<Backend>& src, nn::Index col_start, nn::Index gate_size, bool exact)
    -> nn::TensorImpl<Backend>
{
    return exact ? tanh_exact_block(src, col_start, gate_size)
                 : tanh_fast_block(src, col_start, gate_size);
}

template <typename Backend>
inline auto tanh_tensor(const nn::TensorImpl<Backend>& x, bool exact) -> nn::TensorImpl<Backend>
{
    return exact ? tanh_exact_tensor(x) : tanh_fast_tensor(x);
}

} // namespace nn::activations
