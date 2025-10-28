#ifndef SYNTHETIC_SPIKE_DATA_HPP
#define SYNTHETIC_SPIKE_DATA_HPP

#include <Eigen/Dense>
#include <vector>

#include "tensor/Tensor.hpp"

using std::tuple;
using std::vector;

/**
 * @brief Generates synthetic spike train data using Poisson encoding.
 *
 * @param n_samples Number of samples.
 * @param input_dim Number of input features.
 * @param n_steps Number of time steps for spike train.
 * @param max_rate Maximum firing rate.
 * @param timeStep Time step size.
 * @param[out] real_valued (optional) Pointer to store the real-valued input matrix used for
 * encoding.
 * @return std::vector<Eigen::MatrixXf> Vector of spike trains (one per time step).
 */
auto generate_autoencoder_spike_data(int n_samples, int input_dim, int n_steps, float max_rate,
                                     float timeStep) -> tuple<vector<Tensor>, vector<Tensor>>;

/**
 * @brief Generates synthetic spike train data of ones.
 * @param n_samples Number of samples.
 * @param input_dim Number of input features.
 * @param n_steps Number of time steps for spike train.
 * @return std::vector<Eigen::MatrixXf> Vector of spike trains (one per time step).
 */
auto generate_autoencoder_spike_data_of_ones(int n_samples, int input_dim, int n_steps)
    -> tuple<vector<Tensor>, vector<Tensor>>;

#endif // SYNTHETIC_SPIKE_DATA_HPP
