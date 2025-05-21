#include "batching.hpp"
#include "tensor/Tensor.hpp"
#include <algorithm>
#include <random>

auto create_batches(const Tensor &inputSamples, const Tensor &targets, const int batch_size) -> std::vector<Batch> {

  int const n_samples = static_cast<int>(inputSamples.data.rows());
  std::vector<int> indices(n_samples);
  std::iota(indices.begin(), indices.end(), 0);

  std::random_device rdn;
  std::mt19937 gen(rdn());
  std::ranges::shuffle(indices, gen);

  std::vector<Batch> batches;
  for (int i = 0; i < n_samples; i += batch_size) {
    int const actual_batch_size = std::min(batch_size, n_samples - i);
    Eigen::MatrixXf x_batch(actual_batch_size, inputSamples.data.cols());
    Eigen::MatrixXf y_batch(actual_batch_size, targets.data.cols());

    for (int j = 0; j < actual_batch_size; ++j) {
      x_batch.row(j) = inputSamples.data.row(indices[i + j]);
      y_batch.row(j) = targets.data.row(indices[i + j]);
    }

    batches.push_back({Tensor(x_batch), Tensor(y_batch)});
  }

  return batches;
}
