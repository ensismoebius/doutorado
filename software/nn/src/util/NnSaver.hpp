#include "cnpy.h"
#include "tensor/Tensor.hpp"
#include <span>
#include <string>

struct NnSaver {

  static auto save_weights(const std::string &prefix, Tensor &weights, Tensor &bias) -> void {
    cnpy::npy_save(prefix + "_weights.npy", weights.data.data(), {static_cast<size_t>(weights.data.rows()), static_cast<size_t>(weights.data.cols())}, "w");
    cnpy::npy_save(prefix + "_bias.npy", bias.data.data(), {static_cast<size_t>(bias.data.size())}, "w");
  }

  static auto load_weights(const std::string &prefix, Tensor &weights, Tensor &bias) -> void {
    auto loadedWeights = cnpy::npy_load(prefix + "_weights.npy");
    auto loadedBias = cnpy::npy_load(prefix + "_bias.npy");

    // Convert an array pointer to an iterable
    std::span<const float> const b_span(loadedBias.data<float>(), loadedBias.num_vals);
    std::span<const float> const w_span(loadedWeights.data<float>(), loadedWeights.num_vals);

    // Reconstruct the bias and the weights
    bias.data = Eigen::VectorXf(loadedBias.shape[0]);
    weights.data = Eigen::MatrixXf(loadedWeights.shape[0], loadedWeights.shape[1]);

    for (int i = 0; i < bias.data.size(); ++i) {
      bias.data(i) = b_span[i];
    }

    for (int i = 0; i < weights.data.size(); ++i) {
      weights.data(i) = w_span[i];
    }
  }
};