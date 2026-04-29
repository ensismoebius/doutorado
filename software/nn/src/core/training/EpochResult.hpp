#ifndef NN_TRAINING_EPOCH_RESULT_HPP
#define NN_TRAINING_EPOCH_RESULT_HPP

#include <limits>

namespace nn::training
{

/**
 * @brief Per-epoch training metrics including SNN energy-efficiency indicators.
 *
 * Spike sparsity fields are filled by the Trainer when the model exposes them
 * (e.g., via SpikeCountLoss::last_mean_rate()). For ANN models they remain NaN/0.
 *
 * Energy estimate: SOPs (Synaptic OPerations) = Σ_layers(spikes × fan_out).
 * Compare SOPs vs ANN FLOP count to quantify the SNN energy advantage claimed
 * in SpikingJelly (Science Advances 2023) [26].
 */
struct EpochResult
{
    int epoch = 0;
    float train_loss = 0.0F;
    float val_loss = 0.0F;
    float epoch_ms = 0.0F;

    /// Mean firing rate across all SNN neurons in the last training batch [0,1].
    /// NaN when not measured. Target range: [0.05, 0.80] (see SpikeCountLoss).
    float mean_spike_rate = std::numeric_limits<float>::quiet_NaN();

    /// Estimated synaptic operations (SOPs) for one forward pass.
    /// SOPs = Σ_layer(total_spikes * fan_out). 0 when not measured.
    long long sops = 0LL;
};

} // namespace nn::training

#endif // NN_TRAINING_EPOCH_RESULT_HPP
