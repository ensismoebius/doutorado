#ifndef EXPERIMENT03_AUTOENCODER_CONFIG_HPP
#define EXPERIMENT03_AUTOENCODER_CONFIG_HPP

#include <vector>

/**
 * @file AutoencoderConfig.hpp
 * @brief Shared configuration struct for experiment03 autoencoders.
 *
 * The initial experiment03 models were scaffold-like symmetric MLP stacks.
 * This config now carries enough information for modality-aware redesigns,
 * including multimodal branch hints and architecture-family selection.
 */
enum class AutoencoderArchitecture
{
    Auto,
    ResidualDense,
    DualBranchFusion
};

struct AutoencoderConfig
{
    int input_features = 128;     ///< Dimensionality of the raw input vector (modality-specific).
    int hidden_size = 64;         ///< Width of intermediate hidden layers.
    int latent_size = 32;         ///< Bottleneck dimensionality.
    int depth = 1;                ///< Number of hidden layers on each side (>=1).
    std::vector<int> layer_sizes; ///< Optional explicit hidden-layer widths (overrides
                                  ///< depth/hidden_size tapering).

    AutoencoderArchitecture architecture = AutoencoderArchitecture::Auto;
    int branch_hidden_size = 0; ///< Multimodal branch projection width. 0 = auto.
    int fusion_hidden_size = 0; ///< Shared fusion width. 0 = auto.
    int residual_blocks = 1;    ///< Residual blocks per dense stage where supported.

    // Optional modality split hints for multimodal models.
    int eeg_features = 0;
    int audio_features = 0;

    // SNN-specific parameters (ignored by ANN models).
    float time_step = 1.0F;   ///< Simulation time step passed to Leaky/LeakyIntegrator.
    float resistance = 1.0F;  ///< Membrane resistance.
    float capacitance = 1.0F; ///< Membrane capacitance.
};

#endif // EXPERIMENT03_AUTOENCODER_CONFIG_HPP
