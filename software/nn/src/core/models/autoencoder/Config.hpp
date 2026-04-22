/**
 * @file Config.hpp
 * @brief Configuration for autoencoder models.
 *
 * This config follows PyTorch-style configuration patterns, where model
 * hyperparameters are passed as a structured config object rather than
 * numerous constructor arguments.
 *
 * Key Design Decisions:
 *   - Optional parameters use std::optional for clarity
 *   - Layer specs use vector<string> for declarative configuration
 *   - SNN parameters are grouped and documented separately
 *
 * Example Usage:
 *   @code
 *   AutoencoderConfig cfg{
 *       .input_features = 128,
 *       .hidden_size = 64,
 *       .latent_size = 32,
 *       .depth = 2
 *   };
 *   AudioWindowAutoencoder model(cfg);
 *   @endcode
 */
#ifndef NN_MODELS_AUTOENCODER_CONFIG_HPP
#define NN_MODELS_AUTOENCODER_CONFIG_HPP

#include <optional>
#include <string>
#include <vector>

#include "AutoencoderArchitecture.hpp"

/**
 * @namespace nn::models::autoencoder
 * @brief Core autoencoder implementations
 */
namespace nn::models::autoencoder
{

/**
 * @struct AutoencoderConfig
 * @brief Configuration struct for autoencoder models.
 *
 * This follows the "configuration object" pattern from PyTorch Lightning
 * and similar frameworks. All hyperparameters are in one place.
 *
 * Dimensions Note:
 *   - input_features: raw data dimension (e.g., 128 for audio, 64 for EEG)
 *   - hidden_size: width of intermediate layers
 *   - latent_size: bottleneck dimension (smaller = more compression)
 *   - depth: number of hidden layers on EACH side (encoder AND decoder)
 */
struct AutoencoderConfig
{
    // ==================== Loss ====================
    /** @brief Reconstruction loss type (e.g., "mse", "l1") */
    std::string loss_type = "mse";

    // ==================== Architecture ====================
    /** @brief Input feature dimension (e.g., 128 for audio MFCCs) */
    int input_features = 128;

    /** @brief Hidden layer width */
    int hidden_size = 64;

    /** @brief Latent space (bottleneck) dimension */
    int latent_size = 32;

    /** @brief Number of hidden layers per side (encoder/decoder) */
    int depth = 1;

    /**
     * @brief Explicit layer widths (optional override).
     *
     * If provided, this takes precedence over depth/hidden_size.
     * Example: {128, 64, 32} for 3-layer encoder.
     */
    std::vector<int> layer_sizes;

    // ==================== Declarative Layer Specs (Advanced) ====================
    /** @brief Declarative encoder stage definitions */
    std::vector<std::string> encoder_layer_spec;

    /** @brief Declarative decoder stage definitions */
    std::vector<std::string> decoder_layer_spec;

    /** @brief Declarative branch encoder (multimodal branch 1) */
    std::vector<std::string> branch_encoder_layer_spec;

    /** @brief Declarative branch decoder (multimodal branch 1) */
    std::vector<std::string> branch_decoder_layer_spec;

    /** @brief Declarative fusion encoder (after branch merge) */
    std::vector<std::string> fusion_encoder_layer_spec;

    /** @brief Declarative fusion decoder (before branch split) */
    std::vector<std::string> fusion_decoder_layer_spec;

    // ==================== Architecture Family ====================
    /** @brief Which architecture family to use */
    AutoencoderArchitecture architecture = AutoencoderArchitecture::Auto;

    // ==================== Multimodal Parameters ====================
    /** @brief Branch hidden size for dual-branch models (0 = auto) */
    int branch_hidden_size = 0;

    /** @brief Fusion hidden size for multimodal models (0 = auto) */
    int fusion_hidden_size = 0;

    /** @brief Residual blocks per dense stage (ResidualDense only) */
    int residual_blocks = 1;

    // ==================== Modality Split ====================
    /** @brief EEG features dimension (for multimodal) */
    int eeg_features = 0;

    /** @brief Audio features dimension (for multimodal) */
    int audio_features = 0;

    // ==================== SNN Parameters (Spiking Models) ====================
    /**
     * @name Spiking Neural Network Parameters
     *
     * These parameters are ignored by ANN (ReLU) models but used
     * by SNN variants (LeakyIntegrator activation).
     *
     * SNN Theory:
     *   - time_step: simulation discretisation (smaller = finer temporal resolution)
     *   - resistance, capacitance: membrane constants (affects leak rate)
     *   @{
     */
    /** @brief Simulation time step (seconds) */
    float time_step = 1.0F;

    /** @brief Membrane resistance (Ohms) */
    float resistance = 1.0F;

    /** @brief Membrane capacitance (Farads) */
    float capacitance = 1.0F;
    /** @} */

    // ==================== Initialization ====================
    /** @brief Random seed for weight initialization */
    std::optional<unsigned int> initializer_seed = std::nullopt;

    /** @brief Initialization sampler type */
    std::string initializer_sampler_type;
};

} // namespace nn::models::autoencoder

#endif // NN_MODELS_AUTOENCODER_CONFIG_HPP
