#include "batching.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "core/tensor/Tensor.hpp"

auto create_batches(const std::vector<nn::Tensor>& inputSamples, const std::vector<nn::Tensor>& targets,
                    const int batch_size) -> std::vector<Batch>
{
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
        Eigen::Index input_rows = inputSamples[0].get_data_ref().rows();
        Eigen::Index input_cols = inputSamples[0].get_data_ref().cols();
        Eigen::Index target_rows = targets[0].get_data_ref().rows();
        Eigen::Index target_cols = targets[0].get_data_ref().cols();

        // Create concatenated matrices with proper dimensions
        Eigen::MatrixXf x_concat(input_rows * actual_batch_size, input_cols);
        Eigen::MatrixXf y_concat(target_rows * actual_batch_size, target_cols);

        for (int j = 0; j < actual_batch_size; ++j)
        {
            x_concat.block(j * input_rows, 0, input_rows, input_cols) = x_batch_vec[j].get_data_ref();
            y_concat.block(j * target_rows, 0, target_rows, target_cols) = y_batch_vec[j].get_data_ref();
        }

        batches.push_back({nn::Tensor(x_concat), nn::Tensor(y_concat)});
    }

    return batches;
}
