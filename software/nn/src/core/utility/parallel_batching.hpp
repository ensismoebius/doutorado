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
inline auto create_batches_parallel(const std::vector<nn::Tensor>& inputSamples,
                                    const std::vector<nn::Tensor>& targets, const int batch_size,
                                    const int num_threads = 8) -> std::vector<Batch>
{
    const int n_samples = static_cast<int>(inputSamples.size());

    if (n_samples == 0) return {};

    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

// Parallel Fisher-Yates shuffle with thread-local RNG
#pragma omp parallel num_threads(num_threads)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

#pragma omp for
        for (int i = n_samples - 1; i > 0; --i)
        {
            std::uniform_int_distribution<> dis(0, i);
            const int j = dis(gen);
            std::swap(indices[i], indices[j]);
        }
    }

    const size_t input_rows = inputSamples[0].rows();
    const size_t input_cols = inputSamples[0].cols();
    const size_t target_rows = targets[0].rows();
    const size_t target_cols = targets[0].cols();

    const int n_batches = (n_samples + batch_size - 1) / batch_size;
    std::vector<Batch> batches(n_batches);

// Create batches in parallel
#pragma omp parallel for num_threads(num_threads)
    for (int batch_idx = 0; batch_idx < n_batches; ++batch_idx)
    {
        const int start_idx = batch_idx * batch_size;
        const int actual_batch_size = std::min(batch_size, n_samples - start_idx);

        nn::Tensor x_concat(input_rows * static_cast<size_t>(actual_batch_size), input_cols);
        nn::Tensor y_concat(target_rows * static_cast<size_t>(actual_batch_size), target_cols);

// Fill matrices in parallel within each batch
#pragma omp parallel for num_threads(2) schedule(static)
        for (int j = 0; j < actual_batch_size; ++j)
        {
            const int idx = indices[start_idx + j];
            x_concat.setBlock(static_cast<size_t>(j) * input_rows, 0, inputSamples[idx]);
            y_concat.setBlock(static_cast<size_t>(j) * target_rows, 0, targets[idx]);
        }

        batches[batch_idx] = {std::move(x_concat), std::move(y_concat)};
    }

    return batches;
}

/**
 * @brief Creates pre-allocated batch buffers for efficient batch creation
 */
struct BatchBufferPool
{
    std::vector<nn::Tensor> input_buffers;
    std::vector<nn::Tensor> target_buffers;
    const int batch_size;
    const size_t input_rows;
    const size_t input_cols;
    const size_t target_rows;
    const size_t target_cols;

    BatchBufferPool(const std::vector<nn::Tensor>& inputSamples,
                    const std::vector<nn::Tensor>& targets, int batch_size, int num_buffers)
        : batch_size(batch_size),
          input_rows(inputSamples[0].rows()),
          input_cols(inputSamples[0].cols()),
          target_rows(targets[0].rows()),
          target_cols(targets[0].cols())
    {
        input_buffers.reserve(num_buffers);
        target_buffers.reserve(num_buffers);

        for (int i = 0; i < num_buffers; ++i)
        {
            input_buffers.emplace_back(input_rows * static_cast<size_t>(batch_size), input_cols);
            target_buffers.emplace_back(target_rows * static_cast<size_t>(batch_size),
                                        target_cols);
        }
    }

    auto get_buffer(int buffer_idx) -> Batch
    {
        return {input_buffers[buffer_idx], target_buffers[buffer_idx]};
    }
};

/**
 * @brief Creates batches using pre-allocated buffer pool for better memory efficiency
 */
inline auto create_batches_with_pool(const std::vector<nn::Tensor>& inputSamples,
                                     const std::vector<nn::Tensor>& targets, BatchBufferPool& pool,
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
        x_concat.setBlock(static_cast<size_t>(j) * pool.input_rows, 0, inputSamples[idx]);
        y_concat.setBlock(static_cast<size_t>(j) * pool.target_rows, 0, targets[idx]);
    }

    return pool.get_buffer(buffer_idx);
}

#endif // PARALLEL_BATCHING_HPP
