#ifndef PARALLEL_BATCHING_HPP
#define PARALLEL_BATCHING_HPP

#include <omp.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "../tensor/Tensor.hpp"
#include "batching.hpp"

/**
 * @brief Creates batches in parallel using OpenMP.
 * Optimizes batch creation through parallel shuffling and concatenation.
 */
inline auto create_batches_parallel(const std::vector<Tensor>& inputSamples,
                                    const std::vector<Tensor>& targets, const int batch_size,
                                    const int num_threads = 8) -> std::vector<Batch>
{
    const int n_samples = static_cast<int>(inputSamples.size());

    // Early return for empty input
    if (n_samples == 0) return {};

    // Create and initialize indices in parallel
    std::vector<int> indices(n_samples);
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n_samples; ++i)
    {
        indices[i] = i;
    }

// Parallel Fisher-Yates shuffle with thread-local RNG
#pragma omp parallel num_threads(num_threads)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

#pragma omp for
        for (int i = n_samples - 1; i > 0; --i)
        {
            std::uniform_int_distribution<> dis(0, i);
            int j = dis(gen);
            std::swap(indices[i], indices[j]);
        }
    }

    // Calculate dimensions
    const Eigen::Index input_rows = inputSamples[0].data.rows();
    const Eigen::Index input_cols = inputSamples[0].data.cols();
    const Eigen::Index target_rows = targets[0].data.rows();
    const Eigen::Index target_cols = targets[0].data.cols();

    // Calculate number of batches
    const int n_batches = (n_samples + batch_size - 1) / batch_size;
    std::vector<Batch> batches(n_batches);

// Create batches in parallel
#pragma omp parallel for num_threads(num_threads)
    for (int batch_idx = 0; batch_idx < n_batches; ++batch_idx)
    {
        const int start_idx = batch_idx * batch_size;
        const int actual_batch_size = std::min(batch_size, n_samples - start_idx);

        // Pre-allocate matrices for this batch
        Eigen::MatrixXf x_concat(input_rows * actual_batch_size, input_cols);
        Eigen::MatrixXf y_concat(target_rows * actual_batch_size, target_cols);

// Fill matrices in parallel within each batch
#pragma omp parallel for num_threads(2) schedule(static)
        for (int j = 0; j < actual_batch_size; ++j)
        {
            const int idx = indices[start_idx + j];
            x_concat.block(j * input_rows, 0, input_rows, input_cols) = inputSamples[idx].data;
            y_concat.block(j * target_rows, 0, target_rows, target_cols) = targets[idx].data;
        }

        batches[batch_idx] = {Tensor(x_concat), Tensor(y_concat)};
    }

    return batches;
}

/**
 * @brief Creates pre-allocated batch buffers for efficient batch creation
 */
struct BatchBufferPool
{
    std::vector<Eigen::MatrixXf> input_buffers;
    std::vector<Eigen::MatrixXf> target_buffers;
    const int batch_size;
    const Eigen::Index input_rows;
    const Eigen::Index input_cols;
    const Eigen::Index target_rows;
    const Eigen::Index target_cols;

    BatchBufferPool(const std::vector<Tensor>& inputSamples, const std::vector<Tensor>& targets,
                    int batch_size, int num_buffers)
        : batch_size(batch_size),
          input_rows(inputSamples[0].data.rows()),
          input_cols(inputSamples[0].data.cols()),
          target_rows(targets[0].data.rows()),
          target_cols(targets[0].data.cols())
    {
        // Pre-allocate buffers
        input_buffers.resize(num_buffers, Eigen::MatrixXf(input_rows * batch_size, input_cols));
        target_buffers.resize(num_buffers, Eigen::MatrixXf(target_rows * batch_size, target_cols));
    }

    auto get_buffer(int buffer_idx) -> Batch
    {
        return {Tensor(input_buffers[buffer_idx]), Tensor(target_buffers[buffer_idx])};
    }
};

/**
 * @brief Creates batches using pre-allocated buffer pool for better memory efficiency
 */
inline auto create_batches_with_pool(const std::vector<Tensor>& inputSamples,
                                     const std::vector<Tensor>& targets, BatchBufferPool& pool,
                                     const std::vector<int>& indices, int batch_start,
                                     int buffer_idx) -> Batch
{
    const int n_samples = static_cast<int>(inputSamples.size());
    const int actual_batch_size = std::min(pool.batch_size, n_samples - batch_start);

    auto& x_concat = pool.input_buffers[buffer_idx];
    auto& y_concat = pool.target_buffers[buffer_idx];

#pragma omp parallel for schedule(static)
    for (int j = 0; j < actual_batch_size; ++j)
    {
        const int idx = indices[batch_start + j];
        x_concat.block(j * pool.input_rows, 0, pool.input_rows, pool.input_cols) =
            inputSamples[idx].data;
        y_concat.block(j * pool.target_rows, 0, pool.target_rows, pool.target_cols) =
            targets[idx].data;
    }

    return pool.get_buffer(buffer_idx);
}

#endif // PARALLEL_BATCHING_HPP
