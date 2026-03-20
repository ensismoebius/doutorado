/**
 * @file src/experiments/03/lib/src/experiment03.cpp
 * @brief Core implementation of the Experiment03 driver.
 *
 * Contains experiment lifecycle management: dataset discovery, data loader and
 * prefetcher setup, model construction and the training loop. Public-facing
 * configuration lives in `lib/include/experiment03.hpp`.
 */

#include "experiment03.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>

#include "AudioWindowAutoencoder.hpp"
#include "AudioWindowSpikingAutoencoder.hpp"
#include "AutoencoderConfig.hpp"
#include "EegWindowAutoencoder.hpp"
#include "EegWindowSpikingAutoencoder.hpp"
#include "FusedWindowAutoencoder.hpp"
#include "FusedWindowSpikingAutoencoder.hpp"
#include "ProtocolAutoencoder.hpp"
#include "ProtocolSpikingAutoencoder.hpp"
#include "dataset_info.hpp"
#include "experiment03.hpp"
#include "nn/dataLoaders/10.1117/protocol/Protocol101117Dataset.hpp"
#include "nn/dataLoaders/10.1117/schema/METADATA.hpp"
#include "nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp"
#include "nn/dataLoaders/10.1117/windowing/AudioWindowDataset.hpp"
#include "nn/dataLoaders/10.1117/windowing/EEGWindowDataset.hpp"
#include "nn/dataLoaders/10.1117/windowing/FusedWindowDataset.hpp"
#include "nn/dataLoaders/BatchPrefetcher.hpp"
#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/layers/MSELoss.hpp"
#include "nn/optimizers/Adam.hpp"
#include "progress.hpp"

using std::cerr;
using std::cout;
using std::exception;
using std::make_shared;

using nn::dataLoaders::ImaginedSpeechSchema_10_1117;

namespace
{
auto dataset_type_to_string(Experiment03DatasetType dataset_type) -> const char*
{
    switch (dataset_type)
    {
        case Experiment03DatasetType::Protocol:
            return "protocol";
        case Experiment03DatasetType::EegWindow:
            return "eeg-window";
        case Experiment03DatasetType::AudioWindow:
            return "audio-window";
        case Experiment03DatasetType::FusedWindow:
            return "fused-window";
    }

    return "unknown";
}

auto autoencoder_type_to_string(Experiment03AutoencoderType autoencoder_type) -> const char*
{
    switch (autoencoder_type)
    {
        case Experiment03AutoencoderType::ProtocolAnn:
            return "protocol-ann";
        case Experiment03AutoencoderType::EegWindowAnn:
            return "eeg-window-ann";
        case Experiment03AutoencoderType::AudioWindowAnn:
            return "audio-window-ann";
        case Experiment03AutoencoderType::FusedWindowAnn:
            return "fused-window-ann";
        case Experiment03AutoencoderType::ProtocolSnn:
            return "protocol-snn";
        case Experiment03AutoencoderType::EegWindowSnn:
            return "eeg-window-snn";
        case Experiment03AutoencoderType::AudioWindowSnn:
            return "audio-window-snn";
        case Experiment03AutoencoderType::FusedWindowSnn:
            return "fused-window-snn";
    }

    return "unknown";
}

auto is_autoencoder_compatible(
    Experiment03DatasetType dataset_type, Experiment03AutoencoderType autoencoder_type) -> bool
{
    switch (dataset_type)
    {
        case Experiment03DatasetType::Protocol:
            return autoencoder_type == Experiment03AutoencoderType::ProtocolAnn ||
                   autoencoder_type == Experiment03AutoencoderType::ProtocolSnn;
        case Experiment03DatasetType::EegWindow:
            return autoencoder_type == Experiment03AutoencoderType::EegWindowAnn ||
                   autoencoder_type == Experiment03AutoencoderType::EegWindowSnn;
        case Experiment03DatasetType::AudioWindow:
            return autoencoder_type == Experiment03AutoencoderType::AudioWindowAnn ||
                   autoencoder_type == Experiment03AutoencoderType::AudioWindowSnn;
        case Experiment03DatasetType::FusedWindow:
            return autoencoder_type == Experiment03AutoencoderType::FusedWindowAnn ||
                   autoencoder_type == Experiment03AutoencoderType::FusedWindowSnn;
    }

    return false;
}

auto build_autoencoder_model(const Config& config, int input_features) -> std::unique_ptr<Module>
{
    AutoencoderConfig model_cfg{};
    model_cfg.input_features = input_features;
    model_cfg.hidden_size = config.ae_hidden_size;
    model_cfg.latent_size = config.ae_latent_size;
    model_cfg.depth = config.ae_depth;
    model_cfg.architecture = config.ae_architecture;
    model_cfg.branch_hidden_size = config.ae_branch_hidden_size;
    model_cfg.fusion_hidden_size = config.ae_fusion_hidden_size;
    model_cfg.residual_blocks = config.ae_residual_blocks;
    model_cfg.time_step = config.ae_time_step;
    model_cfg.resistance = config.ae_resistance;
    model_cfg.capacitance = config.ae_capacitance;

    if (config.autoencoder_type == Experiment03AutoencoderType::FusedWindowAnn ||
        config.autoencoder_type == Experiment03AutoencoderType::FusedWindowSnn)
    {
        model_cfg.eeg_features = static_cast<int>(ImaginedSpeechSchema_10_1117.eeg_channels) *
                                 config.eeg_window_spec.window_size;
        model_cfg.audio_features = config.audio_window_spec.window_size;
        if (model_cfg.architecture == AutoencoderArchitecture::Auto)
        {
            model_cfg.architecture = AutoencoderArchitecture::DualBranchFusion;
        }
    }
    else if (config.autoencoder_type == Experiment03AutoencoderType::ProtocolAnn ||
             config.autoencoder_type == Experiment03AutoencoderType::ProtocolSnn)
    {
        if (config.input_mode == Protocol101117InputMode::Concatenated)
        {
            model_cfg.audio_features =
                static_cast<int>(ImaginedSpeechSchema_10_1117.audioSamples());
            model_cfg.eeg_features = static_cast<int>(ImaginedSpeechSchema_10_1117.eeg_channels) *
                                     model_cfg.audio_features;
            if (model_cfg.architecture == AutoencoderArchitecture::Auto)
            {
                model_cfg.architecture = AutoencoderArchitecture::DualBranchFusion;
            }
        }
        else if (model_cfg.architecture == AutoencoderArchitecture::Auto)
        {
            model_cfg.architecture = AutoencoderArchitecture::ResidualDense;
        }
    }
    else if (model_cfg.architecture == AutoencoderArchitecture::Auto)
    {
        model_cfg.architecture = AutoencoderArchitecture::ResidualDense;
    }

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

auto is_snn_type(Experiment03AutoencoderType t) -> bool
{
    return t == Experiment03AutoencoderType::ProtocolSnn ||
           t == Experiment03AutoencoderType::EegWindowSnn ||
           t == Experiment03AutoencoderType::AudioWindowSnn ||
           t == Experiment03AutoencoderType::FusedWindowSnn;
}
} // namespace

Experiment03::Experiment03(const Config& config) : config_(config)
{
    // Dataset, DataLoader and BatchPrefetcher will be initialized in `run()`
    // after parsing CLI params and discovering subjects.
}

int Experiment03::run()
{
    try
    {
        // Initialize processed samples count for progress tracking.
        processed_samples_ = 0;

        // Discover subjects and initialize dataset with specified input mode.
        const auto discovered =
            discoverSubjects(config_.dataset_root, config_.subject_regex_pattern);

        // Instantiate dataset based on configured type.
        // Dataset is shared_ptr so it can be easily passed to DataLoader
        // and persist across the experiment.
        switch (config_.dataset_type)
        {
            case Experiment03DatasetType::Protocol:
            {
                auto proto_dataset = make_shared<Protocol101117Dataset>(discovered);
                proto_dataset->set_input_mode(config_.input_mode);
                dataset_ = proto_dataset;
                break;
            }
            case Experiment03DatasetType::EegWindow:
            {
                dataset_ = make_shared<EEGWindowDataset>(discovered, config_.eeg_window_spec);
                break;
            }
            case Experiment03DatasetType::AudioWindow:
            {
                dataset_ = make_shared<AudioWindowDataset>(discovered, config_.audio_window_spec);
                break;
            }
            case Experiment03DatasetType::FusedWindow:
            {
                dataset_ = make_shared<FusedWindowDataset>(
                    discovered, config_.eeg_window_spec, config_.audio_window_spec);
                break;
            }
        }

        if (!is_autoencoder_compatible(config_.dataset_type, config_.autoencoder_type))
        {
            cout << "Warning: selected dataset type '"
                 << dataset_type_to_string(config_.dataset_type)
                 << "' does not match selected autoencoder '"
                 << autoencoder_type_to_string(config_.autoencoder_type)
                 << "'. Execution will continue using observed input feature size.\n";
        }

        // DataLoader is reused across epochs; prefetcher is re-created per epoch.
        loader_ =
            std::make_unique<DataLoader>(dataset_, config_.batch_size, config_.sampler_options);

        // Store total dataset size for progress tracking.
        dataset_total_samples_ = dataset_->size();

        // Print dataset summary before processing batches.
        // Protocol datasets have a specialized summary; windowing datasets print basic info.
        if (config_.dataset_type == Experiment03DatasetType::Protocol)
        {
            auto* proto = dynamic_cast<Protocol101117Dataset*>(dataset_.get());
            if (proto) printDatasetSummary(*proto, config_.dataset_root);
        }
        else
        {
            cout << "Dataset initialized with " << dataset_total_samples_ << " total samples."
                 << std::endl;
        }

        // Training: iterate over all epochs, recreating the prefetcher each epoch so the
        // DataLoader resets its iteration state for each pass over the data.
        MSELoss loss;
        std::unique_ptr<Adam> optimizer;

        for (int epoch = 0; epoch < config_.epochs; ++epoch)
        {
            cout << "\n=== Epoch " << (epoch + 1) << " / " << config_.epochs << " ===\n";

            // Reset per-epoch counters.
            processed_samples_ = 0;
            seen_batches_ = 0;

            // Re-create prefetcher so it drives the DataLoader through a fresh pass.
            prefetcher_ =
                std::make_unique<BatchPrefetcher>(*loader_, config_.max_batches, config_.lookahead);

            float epoch_loss_sum = 0.0F;
            int epoch_batches = 0;

            while (prefetcher_->hasNext())
            {
                auto maybe_batch = prefetcher_->next();
                if (!maybe_batch.has_value()) break;

                Batch batch = std::move(maybe_batch.value());

                // Lazy model + optimizer init on the first batch observed.
                if (!model_)
                {
                    model_ =
                        build_autoencoder_model(config_, static_cast<int>(batch.inputs.cols()));
                    optimizer = std::make_unique<Adam>(config_.learning_rate);
                    auto init_p = model_->params();
                    optimizer->attach(std::span<nn::Tensor*>(init_p.data(), init_p.size()));
                }

                // SNN models need membrane state reset between independent sequences (batches).
                if (is_snn_type(config_.autoencoder_type)) model_->reset_state();

                // === Training step ===
                auto params = model_->params();
                optimizer->zero_grad(std::span<nn::Tensor*>(params.data(), params.size()));

                // Forward pass (unsupervised reconstruction: target == input).
                auto reconstruction = model_->forward(batch.inputs, /*requires_grad=*/true);

                // Compute MSE reconstruction loss.
                loss.set_target(batch.inputs);
                auto L = loss.forward(reconstruction, /*requires_grad=*/true);
                auto dL = loss.backward(L);

                // Backward pass + parameter update.
                model_->backward(dL);
                optimizer->step(std::span<nn::Tensor*>(params.data(), params.size()));

                epoch_loss_sum += L.at(0, 0);
                ++epoch_batches;

                processed_samples_ += static_cast<std::size_t>(batch.inputs.rows());
                seen_batches_ = prefetcher_->seenBatches();

                printProgress(dataset_total_samples_,
                    config_.batch_size,
                    config_.max_batches,
                    seen_batches_,
                    processed_samples_,
                    false);
            }

            // Finalize progress line for this epoch.
            printProgress(dataset_total_samples_,
                config_.batch_size,
                config_.max_batches,
                prefetcher_->seenBatches(),
                processed_samples_,
                true);

            const float mean_loss =
                epoch_batches > 0 ? epoch_loss_sum / static_cast<float>(epoch_batches) : 0.0F;
            cout << "  mean reconstruction loss: " << mean_loss << "\n";
        }

        if (seen_batches_ == 0)
        {
            cout << "No batches produced. Check dataset files and row counts.\n";
        }

        cout << "Training complete.\n";
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
