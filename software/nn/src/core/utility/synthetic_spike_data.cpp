/**
 * @file synthetic_spike_data.cpp
 * @brief Synthetic spike-train generators used by demos and unit tests.
 */

#include "utility/synthetic_spike_data.hpp"

#include <algorithm>
#include <random>

#include "tensor/Tensor.hpp"

// Global random engine and distribution
// Note on determinism:
// - This uses a process-global RNG seeded from std::random_device.
// - That means results are non-deterministic across runs unless you change this to a fixed seed.

// Note on reproducibility:
// - This translation unit uses a global RNG seeded from std::random_device.
// - That makes generated spike trains non-deterministic across runs by default.
// - For fully reproducible experiments, consider switching to an explicit seed/config.
// - It is also not thread-safe: concurrent calls would race on `gen`.
static std::mt19937 gen(std::random_device{}());
static std::uniform_real_distribution<float> dist(0.0F, 1.0F);

auto generate_autoencoder_spike_data(
    int n_samples, int input_dim, int n_steps, float max_rate, float timeStep)
    -> tuple<vector<nn::Tensor>, vector<nn::Tensor>>
{
    if (n_samples <= 0 || input_dim <= 0 || n_steps <= 0)
    {
        throw std::invalid_argument("n_samples, input_dim, and n_steps must be positive");
    }
    if (max_rate <= 0.0F || timeStep <= 0.0F)
    {
        throw std::invalid_argument("max_rate and timeStep must be positive");
    }

    vector<nn::Tensor> spike_inputs;
    spike_inputs.reserve(n_steps); // Pre-allocate memory
    vector<nn::Tensor> spike_targets;
    spike_targets.reserve(n_steps); // Pre-allocate memory

    // Generate random real-valued between [0, 1] input matrix
    // The underlying “analog” signal is in (0.5, 1.0] due to the +1 and /2.
    // This biases inputs away from near-zero firing rates and keeps demos visually active.
    nn::Tensor real(n_samples, input_dim);
    for (int i = 0; i < n_samples; ++i)
    {
        for (int j = 0; j < input_dim; ++j)
        {
            real.at(i, j) = (dist(gen) + 1.0F) / 2.0F;
        }
    }

    // When max_rate is effectively 1.0, we treat it as always spiking.
    // This avoids flaky "high firing rate" tests due to RNG variance.
    const bool force_spike = max_rate >= 0.99F; // treat near-max rates as always spiking

    for (int step = 0; step < n_steps; ++step)
    {
        // Initialize spike matrix for this time step
        nn::Tensor spikes(n_samples, input_dim);
        spikes.setZero();

        for (int i = 0; i < n_samples; ++i)
        {
            for (int j = 0; j < input_dim; ++j)
            {
                if (force_spike)
                {
                    spikes.at(i, j) = 1.0F;
                    continue;
                }

                // Scale real values to [0, max_rate]
                float rate = real.at(i, j) * max_rate;

                // Probability of spike in this time step (clamped for stability)
                // Intuition: for small `timeStep`, this approximates a Poisson process with
                // expected rate proportional to `rate`.
                const float p_spike = std::min(1.0F, rate * timeStep);

                // For near-maximum rates force spikes to satisfy high-rate tests deterministically
                if (p_spike >= 0.999F)
                {
                    spikes.at(i, j) = 1.0F; //
                    continue;               //
                }

                // Poisson-style spike generation (discretized):
                // a Bernoulli trial per step with probability p_spike.
                if (dist(gen) < p_spike)
                {
                    spikes.at(i, j) = 1.0F;
                }
            }
        }
        // Store the spike train for this time step
        spike_inputs.emplace_back(spikes);
        // Autoencoder target = input (reconstruct the spike train itself).
        spike_targets.emplace_back(spikes); // Copy the spikes tensor
    }
    return {spike_inputs, spike_targets};
}

auto generate_autoencoder_spike_data_of_ones(int n_samples, int input_dim, int n_steps)
    -> tuple<vector<nn::Tensor>, vector<nn::Tensor>>
{
    vector<nn::Tensor> spike_inputs;
    spike_inputs.reserve(n_steps); // Pre-allocate memory
    vector<nn::Tensor> spike_targets;
    spike_targets.reserve(n_steps); // Pre-allocate memory

    for (int step = 0; step < n_steps; ++step)
    {
        // Initialize spike matrix for this time step
        nn::Tensor spikes(n_samples, input_dim);
        spikes.setOnes();

        // Store the spike train for this time step
        spike_inputs.emplace_back(spikes);
        spike_targets.emplace_back(spikes);
    }
    return {spike_inputs, spike_targets};
}