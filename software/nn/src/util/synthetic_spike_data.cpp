#include "synthetic_spike_data.hpp"
#include <cstdlib>
#include <cmath>

// Generates synthetic spike train data using Poisson encoding
// Optionally returns the real-valued input used for encoding

auto generate_synthetic_spike_data(int n_samples, int input_dim, int n_steps, float max_rate, float timeStep, Eigen::MatrixXf* real_valued) -> std::vector<Eigen::MatrixXf> {
    std::vector<Eigen::MatrixXf> spike_trains;
    Eigen::MatrixXf real;
    if (real_valued != nullptr) {
        real = Eigen::MatrixXf::Random(n_samples, input_dim);
        *real_valued = real;
    } else {
        real = Eigen::MatrixXf::Random(n_samples, input_dim);
    }
    for (int step = 0; step < n_steps; ++step) {
        Eigen::MatrixXf spikes = Eigen::MatrixXf::Zero(n_samples, input_dim);
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < input_dim; ++j) {
                float rate = std::abs(real(i, j)) * max_rate;
                if (rate * timeStep > static_cast<float>(rand()) / RAND_MAX) {
                    spikes(i, j) = 1.0F;
                }
            }
        }
        spike_trains.push_back(spikes);
    }
    return spike_trains;
}
