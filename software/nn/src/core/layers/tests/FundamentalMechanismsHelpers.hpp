/**
 * @file src/core/layers/tests/FundamentalMechanismsHelpers.hpp
 * @brief Finite-difference gradient checker shared by the split
 *        fundamental_mechanisms_*_gtest.cpp files.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>

#include "tensor/Tensor.hpp"

namespace
{

// Returns max relative error between analytic gradient and central finite differences.
// f : input → scalar loss
// df: input → gradient tensor (same shape as x0)
// eps: perturbation size (1e-3 is safe for float32)
inline float finite_diff_max_err(std::function<float(const nn::Tensor&)> f,
    std::function<nn::Tensor(const nn::Tensor&)> df,
    const nn::Tensor& x0,
    float eps = 1e-3f)
{
    nn::Tensor analytic = df(x0);
    float max_err = 0.0f;

    for (size_t r = 0; r < x0.rows(); ++r)
    {
        for (size_t c = 0; c < x0.cols(); ++c)
        {
            // Central difference
            nn::Tensor xp = x0;
            xp.at(r, c) += eps;
            nn::Tensor xm = x0;
            xm.at(r, c) -= eps;
            float numeric = (f(xp) - f(xm)) / (2.0f * eps);
            float analyt = analytic.at(r, c);

            float denom = std::max(std::abs(analyt), std::abs(numeric));
            denom = std::max(denom, 1e-6f); // avoid divide-by-zero for near-zero grads
            float rel = std::abs(analyt - numeric) / denom;
            if (rel > max_err) max_err = rel;
        }
    }
    return max_err;
}

// Sum all elements of a tensor (for scalar loss construction)
inline float tensor_sum(const nn::Tensor& t)
{
    float s = 0.0f;
    for (size_t r = 0; r < t.rows(); ++r)
        for (size_t c = 0; c < t.cols(); ++c) s += t.at(r, c);
    return s;
}

} // namespace
