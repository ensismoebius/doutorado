#ifndef BATCHING_HPP
#define BATCHING_HPP

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include "../tensor/Tensor.hpp"

struct Batch
{
    Tensor x;
    Tensor y;
};

std::vector<Batch> create_batches(const Tensor &x, const Tensor &y, int batch_size)
{
    int n_samples = x.data.rows();
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<Batch> batches;
    for (int i = 0; i < n_samples; i += batch_size)
    {
        int actual_batch_size = std::min(batch_size, n_samples - i);
        Eigen::MatrixXf x_batch(actual_batch_size, x.data.cols());
        Eigen::MatrixXf y_batch(actual_batch_size, y.data.cols());

        for (int j = 0; j < actual_batch_size; ++j)
        {
            x_batch.row(j) = x.data.row(indices[i + j]);
            y_batch.row(j) = y.data.row(indices[i + j]);
        }

        batches.push_back({Tensor(x_batch), Tensor(y_batch)});
    }

    return batches;
}

#endif // BATCHING_HPP
