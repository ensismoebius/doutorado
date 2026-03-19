#ifndef EXPERIMENT03_AUTOENCODER_CONFIG_HPP
#define EXPERIMENT03_AUTOENCODER_CONFIG_HPP

/**
 * @file AutoencoderConfig.hpp
 * @brief Shared configuration struct for all autoencoder scaffold models in experiment03.
 *
 * Both ANN and SNN autoencoders accept this struct so experiments can be
 * driven from a single config object (e.g. parsed from YAML / CLI).
 */
struct AutoencoderConfig
{
    int input_features = 128; ///< Dimensionality of the raw input vector (modality-specific).
    int hidden_size = 64;     ///< Width of intermediate hidden layers.
    int latent_size = 32;     ///< Bottleneck dimensionality.
    int depth = 1;            ///< Number of hidden layers on each side (>=1).

    // SNN-specific parameters (ignored by ANN models).
    float time_step = 1.0F;   ///< Simulation time step passed to Leaky/LeakyIntegrator.
    float resistance = 1.0F;  ///< Membrane resistance.
    float capacitance = 1.0F; ///< Membrane capacitance.
};

#endif // EXPERIMENT03_AUTOENCODER_CONFIG_HPP
