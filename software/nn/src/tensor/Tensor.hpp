#pragma once

#include <Eigen/Dense>

struct Tensor
{
    Eigen::MatrixXf data;
    Eigen::MatrixXf grad;

    Tensor() = default;

    Tensor(const int rows, const int cols) : data(rows, cols), grad(rows, cols)
    {
        data.setZero();
        grad.setZero();
    }

    Tensor(const Eigen::MatrixXf& data)
        : data(data), grad(Eigen::MatrixXf::Zero(data.rows(), data.cols()))
    {
    }

    // Returns the shape of the tensor as a vector of longs
    [[nodiscard]] auto get_shape() const -> std::vector<long>
    {
        return {data.rows(), data.cols()};
    }

    // Slices the tensor based on the provided indices
    [[nodiscard]] auto slice(const std::vector<int>& indices) const -> Tensor
    {
        Eigen::MatrixXf sliced_data(indices.size(), data.cols());
        for (long i = 0; i < indices.size(); ++i)
        {
            sliced_data.row(i) = data.row(indices[i]);
        }
        return {sliced_data};
    }
};
