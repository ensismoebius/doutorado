/**
 * @file src/experiments/03/lib/src/experiment03_helpers.cpp
 * @brief Implementation of small helper utilities for Experiment03.
 */

#include "experiment03_helpers.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "AudioWindowAutoencoder.hpp"
#include "AudioWindowSpikingAutoencoder.hpp"
#include "EegWindowAutoencoder.hpp"
#include "EegWindowSpikingAutoencoder.hpp"
#include "FusedWindowAutoencoder.hpp"
#include "FusedWindowSpikingAutoencoder.hpp"
#include "ProtocolAutoencoder.hpp"
#include "ProtocolSpikingAutoencoder.hpp"
#include "nn/dataLoaders/runtime/BatchPrefetcher.hpp"
#include "nn/logging/Logger.hpp"

namespace experiment03
{

auto to_sqlite_dataset_type(Experiment03DatasetType dataset_type)
    -> nn::dataLoaders::SqliteDatasetType
{
    switch (dataset_type)
    {
        case Experiment03DatasetType::Protocol:
            return nn::dataLoaders::SqliteDatasetType::Protocol;
        case Experiment03DatasetType::EegWindow:
            return nn::dataLoaders::SqliteDatasetType::EegWindow;
        case Experiment03DatasetType::AudioWindow:
            return nn::dataLoaders::SqliteDatasetType::AudioWindow;
        case Experiment03DatasetType::FusedWindow:
            return nn::dataLoaders::SqliteDatasetType::FusedWindow;
    }

    return nn::dataLoaders::SqliteDatasetType::Protocol;
}

auto build_autoencoder_model(const Config& config, nn::Index input_features)
    -> std::unique_ptr<Module<nn::EigenTensorBackend>>
{
    AutoencoderConfig model_cfg{};
    model_cfg.input_features =
        config.effective_autoencoder_input_features(static_cast<int>(input_features));
    model_cfg.loss_type = config.training_loss_type;
    model_cfg.hidden_size = config.autoencoder_hidden_size;
    model_cfg.latent_size = config.autoencoder_latent_size;
    model_cfg.depth = config.autoencoder_depth;
    model_cfg.layer_sizes = config.autoencoder_layer_sizes;
    model_cfg.encoder_layer_spec = config.autoencoder_encoder_layer_spec;
    model_cfg.decoder_layer_spec = config.autoencoder_decoder_layer_spec;
    model_cfg.branch_encoder_layer_spec = config.autoencoder_branch_encoder_layer_spec;
    model_cfg.branch_decoder_layer_spec = config.autoencoder_branch_decoder_layer_spec;
    model_cfg.fusion_encoder_layer_spec = config.autoencoder_fusion_encoder_layer_spec;
    model_cfg.fusion_decoder_layer_spec = config.autoencoder_fusion_decoder_layer_spec;
    model_cfg.architecture = config.effective_autoencoder_architecture();
    model_cfg.branch_hidden_size = config.autoencoder_branch_hidden_size;
    model_cfg.fusion_hidden_size = config.autoencoder_fusion_hidden_size;
    model_cfg.residual_blocks = config.autoencoder_residual_blocks;
    model_cfg.time_step = config.autoencoder_time_step;
    model_cfg.resistance = config.autoencoder_resistance;
    model_cfg.capacitance = config.autoencoder_capacitance;
    model_cfg.eeg_features = config.effective_autoencoder_eeg_features();
    model_cfg.audio_features = config.effective_autoencoder_audio_features();
    model_cfg.initializer_seed = config.sampler_shuffle_seed;
    model_cfg.initializer_sampler_type = config.sampler_default_type;

    switch (config.autoencoder_type)
    {
        case Experiment03AutoencoderType::ProtocolAnn:
            return std::make_unique<ProtocolAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::EegWindowAnn:
            return std::make_unique<EegWindowAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::AudioWindowAnn:
            return std::make_unique<AudioWindowAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::FusedWindowAnn:
            return std::make_unique<FusedWindowAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::ProtocolSnn:
            return std::make_unique<ProtocolSpikingAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::EegWindowSnn:
            return std::make_unique<EegWindowSpikingAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::AudioWindowSnn:
            return std::make_unique<AudioWindowSpikingAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::FusedWindowSnn:
            return std::make_unique<FusedWindowSpikingAutoencoder>(model_cfg);
    }

    throw std::runtime_error("Unsupported autoencoder type");
}

auto fit_input_transform(
    const Config& config, size_t max_batches, const std::vector<int>* trial_ids)
    -> std::shared_ptr<nn::transforms::ITransform>
{
    if (!config.training_normalize_inputs)
    {
        return nullptr;
    }

    const auto eeg_cols =
        static_cast<nn::Index>(std::max(0, config.effective_autoencoder_eeg_features()));
    const auto audio_cols =
        static_cast<nn::Index>(std::max(0, config.effective_autoencoder_audio_features()));
    const bool has_eeg = eeg_cols > 0;
    const bool has_audio = audio_cols > 0;

    if (!has_eeg && !has_audio)
    {
        return nullptr;
    }

    using nn::transforms::AudioMeanStdNormalize;
    using nn::transforms::EEGWindowZScore;
    using nn::transforms::FusedModalityTransform;

    std::shared_ptr<AudioMeanStdNormalize> audio_norm;
    bool observed_fused_layout = false;
    bool observed_layout_mismatch = false;
    if (has_audio)
    {
        audio_norm = std::make_shared<AudioMeanStdNormalize>();

        std::unique_ptr<IBatchSource> fitting_src =
            trial_ids ? std::make_unique<SqliteBatchSource>(config.dataset_root_path,
                            config.training_batch_size,
                            to_sqlite_dataset_type(config.dataset_type),
                            config.window_eeg_config,
                            config.window_audio_config,
                            config.dataset_input_mode,
                            *trial_ids)
                      : std::make_unique<SqliteBatchSource>(config.dataset_root_path,
                            config.training_batch_size,
                            to_sqlite_dataset_type(config.dataset_type),
                            config.window_eeg_config,
                            config.window_audio_config,
                            config.dataset_input_mode);

        auto fitting_prefetcher = std::make_unique<BatchPrefetcher>(std::move(fitting_src),
            max_batches,
            config.prefetch_lookahead,
            config.prefetch_ram_cap_mb * std::size_t{1024 * 1024});

        while (fitting_prefetcher->hasNext())
        {
            auto maybe = fitting_prefetcher->next();
            if (!maybe.has_value())
            {
                break;
            }

            const Batch& fitting_batch = maybe.value();
            const nn::Index cols = fitting_batch.inputs.cols();
            const nn::Index required_fused_cols = eeg_cols + audio_cols;
            if (has_eeg && cols >= required_fused_cols)
            {
                observed_fused_layout = true;
                audio_norm->accumulate(fitting_batch.inputs.block(
                    0, eeg_cols, fitting_batch.inputs.rows(), audio_cols));
            }
            else
            {
                observed_layout_mismatch = has_eeg && has_audio;
                audio_norm->accumulate(fitting_batch.inputs);
            }
        }

        try
        {
            audio_norm->finalize();
        }
        catch (const std::runtime_error& e)
        {
            NN_LOG_WARN(std::string{"AudioMeanStdNormalize: fitting scan empty; "
                                    "audio normalization disabled. Reason: "} +
                        e.what());
            audio_norm.reset();
        }
    }

    auto eeg_zscore = has_eeg ? std::make_shared<EEGWindowZScore>() : nullptr;
    if (has_eeg && has_audio && observed_fused_layout && !observed_layout_mismatch)
    {
        return std::make_shared<FusedModalityTransform>(
            eeg_cols, audio_cols, eeg_zscore, audio_norm);
    }

    if (observed_layout_mismatch)
    {
        NN_LOG_WARN(
            "Input normalization fallback: configured EEG/audio split does not match "
            "observed batch shape; using full-vector mean-std normalization.");
    }

    if (audio_norm)
    {
        return std::static_pointer_cast<nn::transforms::ITransform>(audio_norm);
    }

    return has_eeg ? std::static_pointer_cast<nn::transforms::ITransform>(eeg_zscore)
                   : std::static_pointer_cast<nn::transforms::ITransform>(audio_norm);
}

auto modality_val_losses_from_batch(const nn::Tensor& val_inputs,
    const nn::Tensor& val_reconstruction,
    size_t eeg_features,
    size_t audio_features) -> std::pair<float, float>
{
    if (eeg_features == 0 || audio_features == 0)
    {
        return {0.0F, 0.0F};
    }

    const size_t total_features = eeg_features + audio_features;
    if (static_cast<size_t>(val_inputs.cols()) < total_features ||
        static_cast<size_t>(val_reconstruction.cols()) < total_features)
    {
        return {0.0F, 0.0F};
    }

    const auto eeg_target =
        val_inputs.block(0, 0, val_inputs.rows(), static_cast<nn::Index>(eeg_features));
    const auto eeg_pred = val_reconstruction.block(
        0, 0, val_reconstruction.rows(), static_cast<nn::Index>(eeg_features));

    const auto audio_target = val_inputs.block(0,
        static_cast<nn::Index>(eeg_features),
        val_inputs.rows(),
        static_cast<nn::Index>(audio_features));
    const auto audio_pred = val_reconstruction.block(0,
        static_cast<nn::Index>(eeg_features),
        val_reconstruction.rows(),
        static_cast<nn::Index>(audio_features));

    return {eeg_pred.mean_squared_error(eeg_target), audio_pred.mean_squared_error(audio_target)};
}

ReconstructionLoss::ReconstructionLoss(const std::string& loss_type)
{
    const std::string normalized = loss_type.empty() ? "mse" : loss_type;
    if (normalized == "mse")
    {
        mse_ = std::make_unique<MSELoss>();
        return;
    }
    if (normalized == "mae")
    {
        mae_ = std::make_unique<MAELoss>();
        return;
    }
    throw std::invalid_argument("Unsupported training_loss_type: " + loss_type);
}

auto ReconstructionLoss::set_target(const nn::Tensor& target) -> void
{
    if (mse_)
    {
        mse_->set_target(target);
    }
    if (mae_)
    {
        mae_->set_target(target);
    }
}

auto ReconstructionLoss::forward(const nn::Tensor& input, bool requires_grad) -> nn::Tensor
{
    if (mse_)
    {
        return mse_->forward(input, requires_grad);
    }
    return mae_->forward(input, requires_grad);
}

auto ReconstructionLoss::backward(const nn::Tensor& grad_output) -> nn::Tensor
{
    if (mse_)
    {
        return mse_->backward(grad_output);
    }
    return mae_->backward(grad_output);
}

} // namespace experiment03
