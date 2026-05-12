#ifndef SYNTHETIC_SPIKE_DATA_HPP
#define SYNTHETIC_SPIKE_DATA_HPP

#include <vector>

#include "tensor/Tensor.hpp"

using std::tuple;
using std::vector;

/**
 * @file synthetic_spike_data.hpp
 * @brief Small synthetic data generators for demos/tests.
 *
 * Return format:
 * - Both functions return a pair of vectors: {inputs_over_time, targets_over_time}.
 * - Each vector has length `n_steps` (T).
 * - Each element is a 2D tensor shaped (n_samples, input_dim) i.e. (B, F) for a single time slice.
 *
 * How demos typically consume this:
 * - Models like `LifBPTT` expect a flattened time-major matrix of shape (T*B, F).
 * - Common flattening: `flat(t * n_samples + b, f) = seq[t](b, f)`.
 *
 * Determinism note:
 * - The Poisson generator uses a process-global RNG in the .cpp file.
 * - Results are non-deterministic across runs unless you seed deterministically.
 */
/**
 * @brief Generates synthetic spike train data using Poisson encoding.
 *
 * @param n_samples Number of samples (batch size B).
 * @param input_dim Number of input features (F).
 * @param n_steps Number of time steps for spike train (T).
 * @param max_rate Maximum firing rate scaling for the [0,1] real-valued base signal.
 * @param timeStep Time step size used to convert rate into a per-step spike probability.
 * @return {inputs_over_time, targets_over_time} where each is a length-T vector of (B,F) tensors.
 */
auto generate_autoencoder_spike_data(
    int n_samples, int input_dim, int n_steps, float max_rate, float timeStep)
    -> tuple<vector<nn::Tensor>, vector<nn::Tensor>>;

/**
 * This is a convenience generator used for debugging/"can it learn at all" tests.
 * It produces an always-spiking input/target at every time step.
 *
 * Returned shapes match generate_autoencoder_spike_data():
 * - vectors of length `n_steps`
 * - each tensor is (n_samples, input_dim)
 *
 * For `LifBPTT` demos, you still typically flatten into a single (T*B, F) tensor.
 * @brief Generates synthetic spike train data of ones.
 * @param n_samples Number of samples.
 * @param input_dim Number of input features.
 * @param n_steps Number of time steps for spike train.
 * @return {inputs_over_time, targets_over_time} where each is a length-T vector of (B,F) tensors.
 */
auto generate_autoencoder_spike_data_of_ones(int n_samples, int input_dim, int n_steps)
    -> tuple<vector<nn::Tensor>, vector<nn::Tensor>>;

#endif // SYNTHETIC_SPIKE_DATA_HPP
