#pragma once

/**
 * @file Experiment04Config.hpp
 * @brief Runtime configuration for Experiment04 (LSTM Autoencoder).
 *
 * Mirrors the experiment03 Config pattern so that profiles can be loaded from
 * JSON using the same YAML/JSON loader convention. Deliberately kept flat and
 * simple — use a JSON profile to override defaults.
 *
 * Comparison parity with experiment03:
 *   - Same dataset routing (audio-window, eeg-window, fused-window, protocol)
 *   - Same loss function (MSE)
 *   - Same logging format (epoch / train_loss)
 *   - Adds LSTM-specific hyperparameters (hidden_size, latent_size, seq_len)
 */

#include <string>

struct Experiment04Config
{
    // ---- Program / device ----
    std::string program_device = "cpu";

    // ---- Dataset ----
    std::string dataset_root   = ".";
    std::string subject_regex  = ".*";
    /// "audio-window" | "eeg-window" | "fused-window" | "protocol"
    std::string dataset_type   = "audio-window";

    // ---- Data loading ----
    int batch_size          = 1;    ///< samples per iteration
    int max_batches_per_epoch = 0;  ///< 0 = unlimited
    unsigned sampler_shuffle_seed = 42u;

    // ---- LSTM Architecture ----
    int input_size  = 64;   ///< D — feature dimension per time step
    int seq_len     = 32;   ///< T — number of time steps per sample
    int hidden_size = 128;  ///< H — LSTM hidden dimension
    int latent_size = 16;   ///< Z — bottleneck dimension
    int num_layers  = 1;    ///< stacked LSTM layers (encoder and decoder)

    // ---- Training ----
    int   epochs        = 30;
    float learning_rate = 1e-3f;
    float adam_beta1    = 0.9f;
    float adam_beta2    = 0.999f;
    float adam_epsilon  = 1e-8f;

    // ---- Gradient clipping ----
    float grad_clip_norm = 1.0f;  ///< 0 = disabled

    // ---- Output ----
    std::string results_dir = ".";
    std::string run_tag     = "experiment04";
};
