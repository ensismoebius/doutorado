#include "synthetic_spike_data.hpp"
#include "tensor/Tensor.hpp"
#include <cmath>
#include <cstdlib>

// Generates synthetic spike train data using Poisson encoding
// Optionally returns the real-valued input used for encoding

auto generate_autoencoder_spike_data(int n_samples, int input_dim, int n_steps, float max_rate,
                                     float timeStep) -> tuple<vector<Tensor>, vector<Tensor>> {

  vector<Tensor> spike_inputs;
  vector<Tensor> spike_targets;

  Eigen::MatrixXf real;

  // Generate random real-valued input matrix
  real = Eigen::MatrixXf::Random(n_samples, input_dim);

  // Normalize real values to [0, 1]
  for (int step = 0; step < n_steps; ++step) {

    // Scale real values to [0, max_rate]
    Eigen::MatrixXf spikes = Eigen::MatrixXf::Zero(n_samples, input_dim);

    for (int i = 0; i < n_samples; ++i) {
      for (int j = 0; j < input_dim; ++j) {

        float rate = std::abs(real(i, j)) * max_rate;

        // Poisson spike generation
        if (rate * timeStep > static_cast<float>(rand()) / RAND_MAX) {
          spikes(i, j) = 1.0F;
        }
      }
    }

    // Store the spike train for this time step
    spike_inputs.emplace_back(spikes);
    spike_targets.emplace_back(spikes);
  }
  return {spike_inputs, spike_targets};
}
