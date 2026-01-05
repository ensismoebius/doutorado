#include "nn/utility/batching.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "nn/tensor/Tensor.hpp"

auto create_batches(const std::vector<nn::Tensor>& inputSamples,
                    const std::vector<nn::Tensor>& targets, const int batch_size)
    -> std::vector<Batch>
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

    for (int i = 0; i < n_samples; i += batch_size)
    {
        int actual_batch_size = std::min(batch_size, n_samples - i);

        std::vector<nn::Tensor> x_batch_vec;
        std::vector<nn::Tensor> y_batch_vec;

        for (int j = 0; j < actual_batch_size; ++j)
        {
            int idx = indices[i + j];
            x_batch_vec.push_back(inputSamples[idx]);
            y_batch_vec.push_back(targets[idx]);
        }

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
            x_batch.setBlock(j * input_rows, 0, x_batch_vec[j]);
            y_batch.setBlock(j * target_rows, 0, y_batch_vec[j]);
        }

        batches.push_back({x_batch, y_batch});
    }

    return batches;
}
