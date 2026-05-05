/**
 * @file batching.cpp
 * @brief Batch construction helpers for simple dataset-in-memory workflows.
 */

#include "nn/utility/batching.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "nn/tensor/Tensor.hpp"

auto create_batches(std::span<const nn::Tensor> inputSamples,
    std::span<const nn::Tensor> targets,
    const int batch_size) -> std::vector<Batch>
{
    if (batch_size <= 0)
    {
        throw std::invalid_argument("Batch size must be positive");
    }

    if (inputSamples.empty() || targets.empty())
    {
        throw std::invalid_argument("Input and target samples must not be empty");
    }

    if (inputSamples.size() != targets.size())
    {
        throw std::invalid_argument("Input and target sample counts must match");
    }

    const int n_samples = static_cast<int>(inputSamples.size());
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle indices
    std::random_device rdn;
    std::mt19937 gen(rdn());
    std::ranges::shuffle(indices, gen);

    std::vector<Batch> batches;
    batches.reserve((n_samples + batch_size - 1) / batch_size);

    for (int i = 0; i < n_samples; i += batch_size)
    {
        int actual_batch_size = std::min(batch_size, n_samples - i);

        // Get input and target dimensions from first sample
        int input_rows = inputSamples[0].rows();
        int input_cols = inputSamples[0].cols();
        int target_rows = targets[0].rows();
        int target_cols = targets[0].cols();

        // Create batch tensors with proper dimensions
        nn::Tensor x_batch(input_rows * actual_batch_size, input_cols);
        nn::Tensor y_batch(target_rows * actual_batch_size, target_cols);

        for (int j = 0; j < actual_batch_size; ++j)
        {
            int idx = indices[i + j];
            x_batch.setBlock(j * input_rows, 0, inputSamples[idx]);
            y_batch.setBlock(j * target_rows, 0, targets[idx]);
        }

        batches.push_back({x_batch, y_batch});
    }

    return batches;
}

auto batch_to_string(const Batch& batch) -> std::string
{
    std::ostringstream oss;
    const int input_samples = batch.inputs.rows();
    const int input_sample_size = batch.inputs.cols();
    const int target_samples = batch.targets.rows();
    const int target_sample_size = batch.targets.cols();

    oss << "Batch(inputs: samples=" << input_samples << ", sample_size=" << input_sample_size
        << "; targets: samples=" << target_samples << ", sample_size=" << target_sample_size << ")";
    return oss.str();
}
