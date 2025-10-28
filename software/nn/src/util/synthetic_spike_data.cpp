#include "synthetic_spike_data.hpp"

#include <random>

#include "tensor/Tensor.hpp"

auto generate_autoencoder_spike_data(int n_samples, int input_dim, int n_steps, float max_rate,
                                     float timeStep) -> tuple<vector<Tensor>, vector<Tensor>>
{
    // Random engine
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);

    vector<Tensor> spike_inputs;
    vector<Tensor> spike_targets;

    // Generate random real-valued between [0, 1] input matrix
    Eigen::MatrixXf real;
    real = (Eigen::MatrixXf::Random(n_samples, input_dim).array() + 1.0F) / 2.0F;

    // Normalize real values to [0, 1]
    for (int step = 0; step < n_steps; ++step)
    {
        // Initialize spike matrix for this time step
        Eigen::MatrixXf spikes = Eigen::MatrixXf::Zero(n_samples, input_dim);

        for (int i = 0; i < n_samples; ++i)
        {
            for (int j = 0; j < input_dim; ++j)
            {
                // Scale real values to [0, max_rate]
                float rate = real(i, j) * max_rate;

                // Probability of spike in this time step
                const float p_spike = rate * timeStep;

                // Poisson spike generation
                if (dist(gen) < p_spike)
                {
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

auto generate_autoencoder_spike_data_of_ones(int n_samples, int input_dim, int n_steps)
    -> tuple<vector<Tensor>, vector<Tensor>>
{
    vector<Tensor> spike_inputs;
    vector<Tensor> spike_targets;

    // Normalize real values to [0, 1]
    for (int step = 0; step < n_steps; ++step)
    {
        // Initialize spike matrix for this time step
        Eigen::MatrixXf spikes = Eigen::MatrixXf::Ones(n_samples, input_dim);

        // Store the spike train for this time step
        spike_inputs.emplace_back(spikes);
        spike_targets.emplace_back(spikes);
    }
    return {spike_inputs, spike_targets};
}
