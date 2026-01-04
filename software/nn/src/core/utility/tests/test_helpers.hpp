#pragma once

#include <random>

#include "core/tensor/Tensor.hpp"

namespace test_helpers
{

inline auto make_random_tensor(sizet_t rows, sizet_t cols, float lower = -1.0F, float upper = 1.0F)
    -> nn::Tensor
{
    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(lower, upper);
    nn::Tensor t(rows, cols);
    for (sizet_t i = 0; i < rows; ++i)
    {
        for (sizet_t j = 0; j < cols; ++j)
        {
            t.at(i, j) = dist(gen);
        }
    }
    return t;
}

inline auto make_constant_tensor(sizet_t rows, sizet_t cols, float value) -> nn::Tensor
{
    nn::Tensor t(rows, cols);
    for (sizet_t i = 0; i < rows; ++i)
    {
        for (sizet_t j = 0; j < cols; ++j)
        {
            t.at(i, j) = value;
        }
    }
    return t;
}

inline auto make_ones_tensor(sizet_t rows, sizet_t cols) -> nn::Tensor
{
    return make_constant_tensor(rows, cols, 1.0F);
}

inline auto make_zeros_tensor(sizet_t rows, sizet_t cols) -> nn::Tensor
{
    return make_constant_tensor(rows, cols, 0.0F);
}

inline auto tensor_is_approx(const nn::Tensor& a, const nn::Tensor& b, float tolerance = 1e-6F)
    -> bool
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        return false;
    }

    for (sizet_t i = 0; i < a.rows(); ++i)
    {
        for (sizet_t j = 0; j < a.cols(); ++j)
        {
            if (std::abs(a.at(i, j) - b.at(i, j)) > tolerance)
            {
                return false;
            }
        }
    }
    return true;
}

inline auto tensor_is_zero(const nn::Tensor& t, float tolerance = 1e-6F) -> bool
{
    for (sizet_t i = 0; i < t.rows(); ++i)
    {
        for (sizet_t j = 0; j < t.cols(); ++j)
        {
            if (std::abs(t.at(i, j)) > tolerance)
            {
                return false;
            }
        }
    }
    return true;
}

inline auto tensor_norm(const nn::Tensor& t) -> float
{
    float sum = 0.0F;
    for (sizet_t i = 0; i < t.rows(); ++i)
    {
        for (sizet_t j = 0; j < t.cols(); ++j)
        {
            const float val = t.at(i, j);
            sum += val * val;
        }
    }
    return std::sqrt(sum);
}

inline auto tensor_subtract(const nn::Tensor& a, const nn::Tensor& b) -> nn::Tensor
{
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        throw std::invalid_argument("Tensor dimensions must match for subtraction");
    }

    nn::Tensor result(a.rows(), a.cols());
    for (sizet_t i = 0; i < a.rows(); ++i)
    {
        for (sizet_t j = 0; j < a.cols(); ++j)
        {
            result.at(i, j) = a.at(i, j) - b.at(i, j);
        }
    }
    return result;
}

inline void tensor_fill_with_value(nn::Tensor& t, float value)
{
    for (sizet_t i = 0; i < t.rows(); ++i)
    {
        for (sizet_t j = 0; j < t.cols(); ++j)
        {
            t.at(i, j) = value;
        }
    }
}

inline void tensor_set_value_at(nn::Tensor& t, sizet_t row, sizet_t col, float value)
{
    t.at(row, col) = value;
}

} // namespace test_helpers
