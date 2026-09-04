#ifndef NN_MODELS_AUTOENCODER_AUTOENCODER_CONFIG_HPP
#define NN_MODELS_AUTOENCODER_AUTOENCODER_CONFIG_HPP

#include <optional>
#include <string>
#include <vector>

/**
 * @file AutoencoderConfig.hpp
 * @brief Shared configuration struct for autoencoderRunner autoencoders.
 *
 * The initial autoencoderRunner models were scaffold-like symmetric MLP stacks.
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
    std::string loss_type = "mse"; ///< Reconstruction loss token used by training loop.

    int input_features = 128;     ///< Dimensionality of the raw input vector (modality-specific).
    int hidden_size = 64;         ///< Width of intermediate hidden layers.
    int latent_size = 32;         ///< Bottleneck dimensionality.
    int depth = 1;                ///< Number of hidden layers on each side (>=1).
    std::vector<int> layer_sizes; ///< Optional explicit hidden-layer widths (overrides
                                  ///< depth/hidden_size tapering).
    std::vector<std::string> encoder_layer_spec;        ///< Declarative encoder stages.
    std::vector<std::string> decoder_layer_spec;        ///< Declarative decoder stages.
    std::vector<std::string> branch_encoder_layer_spec; ///< Declarative branch encoder stages.
    std::vector<std::string> branch_decoder_layer_spec; ///< Declarative branch decoder stages.
    std::vector<std::string> fusion_encoder_layer_spec; ///< Declarative fusion encoder stages.
    std::vector<std::string> fusion_decoder_layer_spec; ///< Declarative fusion decoder stages.

    AutoencoderArchitecture architecture = AutoencoderArchitecture::Auto;
    int branch_hidden_size = 0; ///< Multimodal branch projection width. 0 = auto.
    int fusion_hidden_size = 0; ///< Shared fusion width. 0 = auto.
    int residual_blocks = 1;    ///< Residual blocks per dense stage where supported.

    // Optional modality split hints for multimodal models.
    int eeg_features = 0;
    int audio_features = 0;

    // SNN-specific parameters (ignored by ANN models).

    /// Number of time steps the spiking stack unrolls over (BPTT sequence length).
    ///
    /// The SNN autoencoders use `LifBPTT`, which consumes a TIME-MAJOR `(T*B, F)` tensor
    /// and unrolls the membrane simulation internally over `time_steps`, giving true
    /// temporal credit assignment. It REQUIRES `input.rows() % time_steps == 0` and
    /// throws otherwise — callers must supply time-major batches, there is no
    /// single-frame fallback. Distinct from `delta_t` below, which is the simulation
    /// step SIZE (`delta_t`, a float), not a count.
    ///
    /// **0 means UNSET and is not a usable value.** The SNN builders raise rather than
    /// assume 1, because `time_steps = 1` is a silent downgrade: LifBPTT would unroll a
    /// single step and behave exactly like the old single-step `Lif`, producing a model
    /// with no temporal credit assignment that still trains and reports a loss. Any
    /// spiking model must state its sequence length explicitly. Ignored by ANN models,
    /// which never construct a LifBPTT.
    int time_steps = 0;

    /// Simulation step SIZE (delta-t), i.e. how long ONE time step lasts. Feeds the
    /// membrane decay `beta = exp(-delta_t / (R*C))`. Distinct from `time_steps`, which
    /// is HOW MANY steps there are. Renamed from `time_step` because one letter of
    /// difference from `time_steps` was a standing source of confusion.
    float delta_t = 1.0F;
    float resistance = 1.0F;        ///< Membrane resistance.
    float capacitance = 1.0F;       ///< Membrane capacitance.
    float voltage_threshold = 1.0F; ///< Spiking Lif firing threshold (encoder). Lower it when
                                    ///< the input current is small (e.g. normalized spike frames)
                                    ///< so encoder neurons actually fire.

    /// Firing-rate regularization weight for the SNN encoder. 0 = disabled.
    /// Pushes each encoder Lif layer's mean firing rate into
    /// [firing_rate_min, firing_rate_max], preventing dead (rate->0, latent
    /// collapse) and bursting (rate->1) neurons. Mirrors the mechanism used by
    /// the dsnn classifier (Training::firing_rate_reg_lambda). Ignored by the
    /// ANN autoencoder (no spiking layers).
    float firing_rate_reg_lambda = 0.0F;
    float firing_rate_min = 0.05F; ///< Lower band edge (dead-neuron guard).
    float firing_rate_max = 0.80F; ///< Upper band edge (burst guard).

    // Initializer controls propagated from sampler options for reproducibility.
    std::optional<unsigned int> initializer_seed = std::nullopt;
    std::string initializer_sampler_type;
};

#endif // NN_MODELS_AUTOENCODER_AUTOENCODER_CONFIG_HPP
