#include "synthetic_spike_data.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>

// Generates synthetic spike train data using Poisson encoding
// Optionally returns the real-valued input used for encoding

auto generate_synthetic_spike_data(int n_samples, int input_dim, int n_steps, float max_rate,
                                   float timeStep) -> std::vector<Eigen::MatrixXf> {

  std::vector<Eigen::MatrixXf> spike_trains;
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

    std::cout << "Spikes:" << spikes.rows() << "x" << spikes.cols() << "\n" << spikes << "\n";
    // Store the spike train for this time step
    spike_trains.push_back(spikes);
  }
  return spike_trains;
}
