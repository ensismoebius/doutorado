#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <vector>
struct Tensor
{
    Eigen::MatrixXf data;
    Eigen::MatrixXf grad;

    Tensor() = default;

    Tensor(const size_t rows, const size_t cols) : data(rows, cols), grad(rows, cols)
    {
        data.setZero();
        grad.setZero();
    }

    Tensor(const Eigen::MatrixXf& m) : data(m), grad(Eigen::MatrixXf::Zero(m.rows(), m.cols())) {}

    // Convenience constructor for scalar
    Tensor(const float v)
        : data(Eigen::MatrixXf::Constant(1, 1, v)), grad(Eigen::MatrixXf::Zero(1, 1))
    {
    }

    [[nodiscard]] auto get_shape() const -> std::vector<long>
    {
        return {static_cast<long>(data.rows()), static_cast<long>(data.cols())};
    }

    // Slice by row indices (used by Dataset/TensorDataset)
    [[nodiscard]] auto slice(const std::vector<int>& indices) const -> Tensor
    {
        if (indices.empty())
        {
            return {};
        }
        const int rows = static_cast<int>(indices.size());
        const int cols = static_cast<int>(data.cols());
        Eigen::MatrixXf out(rows, cols);
        for (int i = 0; i < rows; ++i)
        {
            out.row(i) = data.row(indices[i]);
        }
        return {out};
    }

    // Operator overloads for Tensor operations
    auto operator-(const Tensor& other) const -> Tensor
    {
        Tensor result(data.rows(), data.cols());
        result.data = data - other.data;
        return result;
    }

    auto operator*(float scalar) const -> Tensor
    {
        Tensor result(data.rows(), data.cols());
        result.data = data * scalar;
        return result;
    }

    [[nodiscard]] auto square() const -> Tensor
    {
        Tensor result(data.rows(), data.cols());
        result.data = data.array().square();
        return result;
    }

    [[nodiscard]] auto mean() const -> Tensor
    {
        return {data.mean()};
    }

    [[nodiscard]] auto get_data() const -> const Eigen::MatrixXf&
    {
        return data;
    }

    [[nodiscard]] auto size() const -> long
    {
        return data.size();
    }
};
