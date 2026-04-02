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
#include <vector>

#include "AudioWindowAutoencoder.hpp"
#include "AudioWindowSpikingAutoencoder.hpp"
#include "AutoencoderConfig.hpp"
#include "DatasetBuilder.hpp"
#include "EegWindowAutoencoder.hpp"
#include "EegWindowSpikingAutoencoder.hpp"
#include "FusedWindowAutoencoder.hpp"
#include "FusedWindowSpikingAutoencoder.hpp"
#include "ProtocolAutoencoder.hpp"
#include "ProtocolSpikingAutoencoder.hpp"
#include "ResultsWriter.hpp"
#include "RunSummaryBuilder.hpp"
#include "experiment03.hpp"
#include "nn/dataLoaders/10.1117/dataset_info.hpp"
#include "nn/dataLoaders/10.1117/protocol/Protocol101117Dataset.hpp"
#include "nn/dataLoaders/10.1117/schema/METADATA.hpp"
#include "nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp"
#include "nn/dataLoaders/BatchPrefetcher.hpp"
#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/dataLoaders/SqliteBatchSource.hpp"
#include "nn/layers/MSELoss.hpp"
#include "nn/logging/Logger.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/tensor/EigenTensorBackend.hpp"
#include "nn/utility/progress.hpp"

using std::exception;
using std::make_shared;

using experiment03::DatasetBuilder;
using experiment03::Summary;
using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
using std::endl;
using std::make_unique;
using std::size_t;
using std::unique_ptr;

namespace
{
auto build_autoencoder_model(const Config& config, nn::Index input_features)
    -> std::unique_ptr<Module>
{
    AutoencoderConfig model_cfg{};
    model_cfg.input_features =
        config.autoencoder_input_features > 0 ? config.autoencoder_input_features : input_features;
    model_cfg.hidden_size = config.autoencoder_hidden_size;
    model_cfg.latent_size = config.autoencoder_latent_size;
    model_cfg.depth = config.autoencoder_depth;
    model_cfg.layer_sizes = config.autoencoder_layer_sizes;
    model_cfg.architecture = config.autoencoder_architecture;
    model_cfg.branch_hidden_size = config.autoencoder_branch_hidden_size;
    model_cfg.fusion_hidden_size = config.autoencoder_fusion_hidden_size;
    model_cfg.residual_blocks = config.autoencoder_residual_blocks;
    model_cfg.time_step = config.autoencoder_time_step;
    model_cfg.resistance = config.autoencoder_resistance;
    model_cfg.capacitance = config.autoencoder_capacitance;

    if (config.autoencoder_type == Experiment03AutoencoderType::FusedWindowAnn ||
        config.autoencoder_type == Experiment03AutoencoderType::FusedWindowSnn)
    {
        const int inferred_eeg_features =
            static_cast<int>(ImaginedSpeechSchema_10_1117.eeg_channels) *
            config.eeg_window_config.window_size;
        const int inferred_audio_features = config.audio_window_config.window_size;

        model_cfg.eeg_features = config.autoencoder_eeg_features > 0
                                     ? config.autoencoder_eeg_features
                                     : inferred_eeg_features;
        model_cfg.audio_features = config.autoencoder_audio_features > 0
                                       ? config.autoencoder_audio_features
                                       : inferred_audio_features;
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
            const int inferred_audio_features =
                static_cast<int>(ImaginedSpeechSchema_10_1117.audioSamples());
            const int inferred_eeg_features =
                static_cast<int>(ImaginedSpeechSchema_10_1117.eeg_channels) *
                inferred_audio_features;

            model_cfg.audio_features = config.autoencoder_audio_features > 0
                                           ? config.autoencoder_audio_features
                                           : inferred_audio_features;
            model_cfg.eeg_features = config.autoencoder_eeg_features > 0
                                         ? config.autoencoder_eeg_features
                                         : inferred_eeg_features;
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
            return make_unique<ProtocolAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::EegWindowAnn:
            return make_unique<EegWindowAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::AudioWindowAnn:
            return make_unique<AudioWindowAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::FusedWindowAnn:
            return make_unique<FusedWindowAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::ProtocolSnn:
            return make_unique<ProtocolSpikingAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::EegWindowSnn:
            return make_unique<EegWindowSpikingAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::AudioWindowSnn:
            return make_unique<AudioWindowSpikingAutoencoder>(model_cfg);
        case Experiment03AutoencoderType::FusedWindowSnn:
            return make_unique<FusedWindowSpikingAutoencoder>(model_cfg);
    }

    throw std::runtime_error("Unsupported autoencoder type");
}

} // namespace

Experiment03::Experiment03(const Config& config) : config_(config)
{
    // Dataset, DataLoader and BatchPrefetcher will be initialized in `run()`
    // after parsing CLI params and discovering subjects.
}

int Experiment03::run()
{
    std::vector<float> epoch_mean_losses;

    try
    {
        ////////////////////////////
        // Dataset initialization //
        ////////////////////////////

        // Discover subjects and initialize dataset with specified input mode.
        const auto discovered = discoverSubjects( //
            config_.dataset_root,                 //
            config_.subject_filter_regex          //
        );

        // Instantiate dataset using a small builder to keep selection logic
        // centralized and easier to test.
        dataset_ = DatasetBuilder()                 //
                       .with_discovered(discovered) //
                       .with_config(config_)        //
                       .build();

        if (!is_autoencoder_compatible(config_.dataset_type, config_.autoencoder_type)) [[unlikely]]
        {
            std::ostringstream _oss;
            _oss << "Warning: selected dataset type '"
                 << dataset_type_to_string(config_.dataset_type)
                 << "' does not match selected autoencoder '"
                 << autoencoder_type_to_string(config_.autoencoder_type)
                 << "'. Execution will continue using observed input feature size.";
            NN_LOG_WARN(_oss.str());
        }

        // DataLoader is reused across training_epochs; prefetcher is re-created per epoch.
        data_loader_ = make_unique<DataLoader>( //
            dataset_,                           //
            config_.batch_size,                 //
            config_.resolved_sampler_options    //
        );

        // Store total dataset size for progress tracking.
        dataset_total_samples_ = dataset_->size();

        // Print dataset summary before processing batches.
        // Protocol datasets have a specialized summary; windowing datasets print basic info.
        if (config_.dataset_type == Experiment03DatasetType::Protocol)
        {
            auto* protocol_dataset = dynamic_cast<Protocol101117Dataset*>(dataset_.get());
            if (protocol_dataset) printDatasetSummary(*protocol_dataset, config_.dataset_root);
        }
        else
        {
            std::ostringstream _oss;
            _oss << "Dataset initialized with " << dataset_total_samples_ << " total samples.";
            NN_LOG_INFO(_oss.str());
        }

        /////////////////////////////////
        // Training loop setup and run //
        /////////////////////////////////

        // Create loss function and optimizer (initialized later
        // once model is created and parameters are known).
        MSELoss loss;
        unique_ptr<Adam> optimizer;

        // Training: iterate over all training_epochs.
        // Recreating the prefetcher each epoch so the DataLoader
        // resets its iteration state for each pass over the data.
        for (size_t epoch = 0; epoch < config_.training_epochs; ++epoch)
        {
            // Track cumulative loss and batch count for mean
            // loss reporting at the end of the epoch.
            size_t epoch_batches = 0;
            float epoch_loss_sum = 0.0F;
            double last_batch_loss = 0.0;

            // Reset per-epoch counters.
            seen_batches_ = 0;
            processed_samples_ = 0;

            // Re-create prefetcher so it drives the DataLoader through a fresh pass.
            std::unique_ptr<IBatchSource> src;

            // Choose DB-backed or in-memory DataLoader-backed source.
            if (config_.use_sqlite)
            {
                // Map experiment dataset enum to SqliteBatchSource internal enum.
                nn::dataLoaders::SqliteDatasetType ds_type =
                    nn::dataLoaders::SqliteDatasetType::Protocol;
                switch (config_.dataset_type)
                {
                    case Experiment03DatasetType::Protocol:
                        ds_type = nn::dataLoaders::SqliteDatasetType::Protocol;
                        break;
                    case Experiment03DatasetType::EegWindow:
                        ds_type = nn::dataLoaders::SqliteDatasetType::EegWindow;
                        break;
                    case Experiment03DatasetType::AudioWindow:
                        ds_type = nn::dataLoaders::SqliteDatasetType::AudioWindow;
                        break;
                    case Experiment03DatasetType::FusedWindow:
                        ds_type = nn::dataLoaders::SqliteDatasetType::FusedWindow;
                        break;
                }
                src = std::make_unique<SqliteBatchSource>(config_.dataset_root,
                    config_.batch_size,
                    ds_type,
                    config_.eeg_window_config,
                    config_.audio_window_config,
                    config_.input_mode);
            }
            else
            {
                // `DataLoaderSource` removed; enforce SQLite-only source.
                nn::dataLoaders::SqliteDatasetType ds_type =
                    nn::dataLoaders::SqliteDatasetType::Protocol;
                switch (config_.dataset_type)
                {
                    case Experiment03DatasetType::Protocol:
                        ds_type = nn::dataLoaders::SqliteDatasetType::Protocol;
                        break;
                    case Experiment03DatasetType::EegWindow:
                        ds_type = nn::dataLoaders::SqliteDatasetType::EegWindow;
                        break;
                    case Experiment03DatasetType::AudioWindow:
                        ds_type = nn::dataLoaders::SqliteDatasetType::AudioWindow;
                        break;
                    case Experiment03DatasetType::FusedWindow:
                        ds_type = nn::dataLoaders::SqliteDatasetType::FusedWindow;
                        break;
                }
                src = std::make_unique<SqliteBatchSource>(config_.dataset_root,
                    config_.batch_size,
                    ds_type,
                    config_.eeg_window_config,
                    config_.audio_window_config,
                    config_.input_mode);
            }

            // Determine max batches for this epoch. A value of 0 in the
            // configuration means "process the entire dataset for the
            // epoch", so compute the number of batches from dataset size.
            std::size_t epoch_max_batches = config_.max_batches_per_epoch;
            if (epoch_max_batches == 0)
            {
                const std::size_t batches = dataset_total_samples_ / config_.batch_size;
                epoch_max_batches = batches + (dataset_total_samples_ % config_.batch_size ? 1 : 0);
            }

            prefetcher_ = make_unique<BatchPrefetcher>(                //
                std::move(src),                                        //
                epoch_max_batches,                                     //
                config_.prefetch_lookahead,                            //
                config_.prefetch_ram_cap_mb * std::size_t{1024 * 1024} //
            );

            // Iterate over batches produced by the prefetcher.
            while (prefetcher_->hasNext()) [[likely]]
            {
                // Get next batch from prefetcher;
                // break if no more batches are available.
                auto maybe_batch = prefetcher_->next();
                if (!maybe_batch.has_value()) [[unlikely]]
                    break;

                // Move batch out of optional;
                // batch is now owned by this scope.
                const Batch batch = std::move(maybe_batch.value());

                // Lazy model + optimizer init on the first batch observed.
                if (!model_) [[unlikely]]
                {
                    // Build model based on observed input feature size and config.
                    model_ = build_autoencoder_model( //
                        config_,                      //
                        batch.inputs.cols()           //
                    );

                    // Initialize optimizer with model parameters.
                    optimizer = make_unique<Adam>(config_.training_learning_rate);

                    // Attach model parameters to optimizer.
                    optimizer->attach(model_->params());
                }

                // SNN models need membrane state reset between independent sequences (batches).
                if (is_snn_type(config_.autoencoder_type)) model_->reset_state();

                // === Training step ===
                auto params = model_->params();
                optimizer->zero_grad(params);

                // Forward pass (unsupervised reconstruction: target == input).
                auto reconstruction = model_->forward(batch.inputs, /*requires_grad=*/true);

                // Compute MSE reconstruction loss.
                loss.set_target(batch.inputs);

                auto loss_value = loss.forward(reconstruction, /*requires_grad=*/true);
                auto derivative_loss_value = loss.backward(loss_value);

                // Backward pass + parameter update.
                model_->backward(derivative_loss_value);
                optimizer->step(params);

                epoch_loss_sum += loss_value.at(0, 0);
                last_batch_loss = static_cast<double>(loss_value.at(0, 0));
                ++epoch_batches;

                processed_samples_ += static_cast<size_t>(batch.inputs.rows());
                seen_batches_ = prefetcher_->seenBatches();

                printProgress(dataset_total_samples_,
                    config_.batch_size,
                    config_.max_batches_per_epoch,
                    seen_batches_,
                    processed_samples_,
                    false,
                    epoch + 1,
                    config_.training_epochs,
                    static_cast<double>(loss_value.at(0, 0)));
            }

            // Finalize progress line for this epoch.
            if (model_)
            {
                printProgress(dataset_total_samples_,
                    config_.batch_size,
                    config_.max_batches_per_epoch,
                    prefetcher_->seenBatches(),
                    processed_samples_,
                    true,
                    epoch + 1,
                    config_.training_epochs,
                    last_batch_loss,
                    model_->params());
            }

            if (prefetcher_)
            {
                auto d = prefetcher_->diagnostics();
                std::ostringstream _oss;
                _oss << "  Prefetcher ring size: " << d.ring_size
                     << " | seen batches: " << d.seen_batches;
                NN_LOG_INFO(_oss.str());
            }

            const float mean_loss =
                epoch_batches > 0 ? epoch_loss_sum / static_cast<float>(epoch_batches) : 0.0F;
            epoch_mean_losses.push_back(mean_loss);
            {
                std::ostringstream _oss;
                _oss << "  mean reconstruction loss: " << mean_loss;
                NN_LOG_INFO(_oss.str());
            }
        }

        if (seen_batches_ == 0)
        {
            NN_LOG_WARN("No batches produced. Check dataset files and row counts.");
        }

        std::string results_path;
        std::string results_error;
        auto summary = experiment03::build_run_summary(config_,
            0,
            dataset_total_samples_,
            processed_samples_,
            seen_batches_,
            epoch_mean_losses);

        if (experiment03::write_run_summary_json(summary, results_path, results_error))
        {
            NN_LOG_INFO(std::string("Run summary written to: ") + results_path);
        }
        else
        {
            NN_LOG_WARN(std::string("Warning: failed to write run summary: ") + results_error);
        }

        NN_LOG_INFO("Training complete.");
    }
    catch (const exception& e)
    {
        std::string results_path;
        std::string results_error;
        auto summary = experiment03::build_run_summary(config_,
            1,
            dataset_total_samples_,
            processed_samples_,
            seen_batches_,
            epoch_mean_losses,
            e.what());
        (void) experiment03::write_run_summary_json(summary, results_path, results_error);

        NN_LOG_ERROR(std::string("Error: ") + e.what());
        return 1;
    }

    return 0;
}
