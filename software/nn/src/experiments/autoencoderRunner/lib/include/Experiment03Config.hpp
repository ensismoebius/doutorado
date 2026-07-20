/**
 * @file src/experiments/03/lib/include/Experiment03Config.hpp
 * @brief Runtime configuration structure for Experiment03.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "AutoencoderConfig.hpp"
#include "Experiment03AutoencoderType.hpp"
#include "data_loaders/10.1117/datasets/raw/Dataset101117.hpp"
#include "data_loaders/10.1117/schema/Metadata.hpp"
#include "data_loaders/runtime/DataLoader.hpp"
#include "windowing/WindowSpec.hpp"

using std::optional;
using std::string;

struct Config
{
    string profile_name; // Name of profile used to seed defaults.

    string device; // Execution device token: cpu|opencl

    string dataset_subject_filter_regex; // Subject directory regex with an ID capture group
    string dataset_root_path;            // Path containing subject directories

    size_t training_batch_size; // Batch size for DataLoader and optimizer step granularity
    size_t training_max_batches_per_epoch; // 0 = consume the full dataset each epoch

    // Legacy controls (used when sampler_default_type is not provided)
    bool sampler_shuffle_samples; // Shuffle samples before batching?
    optional<unsigned int>
        sampler_shuffle_seed; // RNG seed for shuffling (ignored if sampler_shuffle_samples=false)

    // Explicit default sampler selector.
    string sampler_default_type; // Empty means use legacy shuffle behavior.

    // Weighted sampler options
    std::vector<double> sampler_weights;
    std::optional<size_t> sampler_weighted_num_samples;

    // Distributed sampler options
    size_t sampler_distributed_num_replicas;
    size_t sampler_distributed_rank;
    bool sampler_distributed_shuffle;
    bool sampler_distributed_drop_last;

    // Input modality used by Dataset101117.
    Protocol101117InputMode dataset_input_mode;

    // Dataset variant used by experiment03.
    Experiment03DatasetType dataset_type;

    // Autoencoder variant used by experiment03.
    Experiment03AutoencoderType autoencoder_type;

    // Autoencoder architecture hyperparameters.
    int autoencoder_hidden_size;
    int autoencoder_latent_size;
    int autoencoder_depth;
    std::vector<int> autoencoder_layer_sizes; // Empty = infer with depth/hidden-size tapering
    std::vector<std::string> autoencoder_encoder_layer_spec;
    std::vector<std::string> autoencoder_decoder_layer_spec;
    std::vector<std::string> autoencoder_branch_encoder_layer_spec;
    std::vector<std::string> autoencoder_branch_decoder_layer_spec;
    std::vector<std::string> autoencoder_fusion_encoder_layer_spec;
    std::vector<std::string> autoencoder_fusion_decoder_layer_spec;
    int autoencoder_input_features; // 0 = infer from dataset batch shape
    int autoencoder_eeg_features;   // 0 = infer from dataset/window config
    int autoencoder_audio_features; // 0 = infer from dataset/window config
    AutoencoderArchitecture autoencoder_architecture;
    int autoencoder_branch_hidden_size;
    int autoencoder_fusion_hidden_size;
    int autoencoder_residual_blocks;
    float autoencoder_time_step;
    float autoencoder_resistance;
    float autoencoder_capacitance;

    // Training hyperparameters.
    string training_optimizer_type;        // Optimizer token: adam|sgd
    float training_learning_rate;          // Shared optimizer learning rate
    float training_optimizer_momentum;     // SGD momentum term
    float training_optimizer_adam_beta1;   // Adam first-moment decay
    float training_optimizer_adam_beta2;   // Adam second-moment decay
    float training_optimizer_adam_epsilon; // Adam numerical stability epsilon
    string training_loss_type;             // Reconstruction loss token: mse|mae
    size_t training_epochs;                // Number of passes over the dataset
    bool training_lr_plateau_enabled;      // Enable ReduceLROnPlateau scheduler
    float training_lr_plateau_factor;      // Multiplicative LR decay factor
    size_t training_lr_plateau_patience;   // Epochs without improvement before decaying LR
    float training_lr_plateau_min_delta;   // Minimum validation improvement to reset patience
    bool training_normalize_inputs = true; // Enable per-modality input normalization (audio:
                                           // column-wise mean-std; EEG: per-window z-score)

    bool validation_modality_diagnostics_enabled; // Report EEG/Audio validation losses per epoch

    // Window specs used by windowing datasets.
    nn::windowing::WindowSpec window_eeg_config;
    nn::windowing::WindowSpec window_audio_config;

    // Number of batches to prefetch in background.
    // Increase this to overlap I/O with training when RAM allows.
    std::size_t prefetch_lookahead;
    // Maximum RAM in MB for prefetched batches (0 = unlimited).
    std::size_t prefetch_ram_cap_mb;
    // Use precomputed per-subject shard files for faster I/O when available.
    // Note: shard detection is automatic; the legacy `use_shards` flag has been removed.

    // Parsed and resolved sampler options for DataLoader construction.
    DataLoader::DefaultSamplerOptions sampler_resolved_options;

    // K-Fold cross-validation settings.
    bool kfold_enabled;                // Enable k-fold training/validation loop.
    size_t kfold_n_splits;             // Number of folds (k); minimum 2.
    bool kfold_shuffle;                // Shuffle indices before fold assignment.
    optional<unsigned int> kfold_seed; // Optional RNG seed for reproducible shuffling.

    // Test set split (applied before k-fold).
    // If > 0 and < 1, fraction of data to hold out as test set.
    // Remaining data is used for train/validation (k-fold when enabled).
    float test_split = 0.0f;

    // Returns effective input feature count, using inferred dataset shape when override is unset.
    auto effective_autoencoder_input_features(int observed_input_features) const -> int
    {
        return autoencoder_input_features > 0 ? autoencoder_input_features
                                              : observed_input_features;
    }

    // Returns effective EEG feature count, inferring from dataset/window shape when required.
    auto effective_autoencoder_eeg_features() const -> int
    {
        if (autoencoder_eeg_features > 0)
        {
            return autoencoder_eeg_features;
        }

        if (autoencoder_type == Experiment03AutoencoderType::FusedWindowAnn ||
            autoencoder_type == Experiment03AutoencoderType::FusedWindowSnn)
        {
            return static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels) *
                   window_eeg_config.window_size;
        }

        if ((autoencoder_type == Experiment03AutoencoderType::ProtocolAnn ||
                autoencoder_type == Experiment03AutoencoderType::ProtocolSnn) &&
            dataset_input_mode == Protocol101117InputMode::Concatenated)
        {
            return static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels) *
                   static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples());
        }

        return 0;
    }

    // Returns effective audio feature count, inferring from dataset/window shape when required.
    auto effective_autoencoder_audio_features() const -> int
    {
        if (autoencoder_audio_features > 0)
        {
            return autoencoder_audio_features;
        }

        if (autoencoder_type == Experiment03AutoencoderType::FusedWindowAnn ||
            autoencoder_type == Experiment03AutoencoderType::FusedWindowSnn)
        {
            return window_audio_config.window_size;
        }

        if ((autoencoder_type == Experiment03AutoencoderType::ProtocolAnn ||
                autoencoder_type == Experiment03AutoencoderType::ProtocolSnn) &&
            dataset_input_mode == Protocol101117InputMode::Concatenated)
        {
            return static_cast<int>(nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples());
        }

        return 0;
    }

    // Resolves auto architecture to an explicit one based on model/data mode when set to Auto.
    auto effective_autoencoder_architecture() const -> AutoencoderArchitecture
    {
        if (autoencoder_architecture != AutoencoderArchitecture::Auto)
        {
            return autoencoder_architecture;
        }

        const bool is_fused = autoencoder_type == Experiment03AutoencoderType::FusedWindowAnn ||
                              autoencoder_type == Experiment03AutoencoderType::FusedWindowSnn;
        const bool is_protocol_concat =
            (autoencoder_type == Experiment03AutoencoderType::ProtocolAnn ||
                autoencoder_type == Experiment03AutoencoderType::ProtocolSnn) &&
            dataset_input_mode == Protocol101117InputMode::Concatenated;

        if (is_fused || is_protocol_concat)
        {
            return AutoencoderArchitecture::DualBranchFusion;
        }

        return AutoencoderArchitecture::ResidualDense;
    }

    // Enable OpenCL profiling instrumentation when running with OpenCL.
    // Default: false (disabled).
    bool opencl_profiling_enabled = false;
};
