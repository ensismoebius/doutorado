/**
 * @file src/experiments/03/lib/src/experiment03.cpp
 * @brief Core implementation of the Experiment03 driver.
 *
 * Contains experiment lifecycle management: dataset discovery, data loader and
 * prefetcher setup, model construction and the training loop. Public-facing
 * configuration lives in `lib/include/experiment03.hpp`.
 */

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

// Experiment-specific components
#include "DatasetBuilder.hpp"
#include "ResultsWriter.hpp"
#include "RunSummaryBuilder.hpp"
#include "TrialFoldSelector.hpp"
#include "experiment03.hpp"
#include "experiment03_helpers.hpp"

// Core libraries
#include "nn/dataLoaders/10.1117/datasets/raw/Dataset101117Printer.hpp"
#include "nn/dataLoaders/10.1117/datasets/windowed/WindowingDatasetPrinter.hpp"
#include "nn/dataLoaders/10.1117/schema/METADATA.hpp"
#include "nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp"
#include "nn/dataLoaders/BatchPrefetcher.hpp"
#include "nn/dataLoaders/IDatasetPrinter.hpp"
#include "nn/dataLoaders/SqliteBatchSource.hpp"
#include "nn/device/Device.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/logging/Logger.hpp"
#include "nn/optimizers/Optimizer.hpp"
#include "nn/optimizers/OptimizerFactory.hpp"
#include "nn/tensor/eigen/EigenTensorBackend.hpp"
#include "nn/tensor/opencl/OpenCLTensorBackend.hpp"
#include "nn/utility/batching.hpp"
#include "nn/utility/progress.hpp"

// Local using declarations for experiment components.
using experiment03::build_autoencoder_model;
using experiment03::build_run_summary;
using experiment03::DatasetBuilder;
using experiment03::Summary;
using experiment03::to_sqlite_dataset_type;
using experiment03::write_run_summary_json;

// Core nn namespaces
using nn::Device;
using nn::Index;
using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
using nn::dataLoaders::SqliteDatasetType;
using nn::optimizers::OptimizerFactory;

// Standard library namespaces
using std::endl;
using std::exception;
using std::make_shared;
using std::make_unique;
using std::ostringstream;
using std::runtime_error;
using std::size_t;
using std::string;
using std::unique_ptr;
using std::vector;

namespace
{
/// @brief Exit code indicating successful experiment completion.
constexpr int kExitSuccess = 0;
/// @brief Exit code indicating the experiment terminated with an error.
constexpr int kExitFailure = 1;

auto effective_fold_max_batches(size_t configured_max_batches, size_t trial_count, size_t n_splits)
    -> size_t
{
    if (configured_max_batches == 0)
    {
        return std::max<size_t>(1, trial_count);
    }

    const size_t splits = std::max<size_t>(1, n_splits);
    const size_t per_fold_budget = (configured_max_batches + splits - 1) / splits;
    return std::max<size_t>(1, std::min(per_fold_budget, trial_count));
}
} // namespace

// internal helpers moved to experiment03_helpers.hpp / .cpp

Experiment03::Experiment03(const Config& config) : config_(config)
{
    // Dataset, DataLoader and BatchPrefetcher will be initialized in `run()`
    // after parsing CLI params and discovering subjects.
}

int Experiment03::run()
{
    /// Mean losses for each epoch, used for final reporting in the run summary.
    vector<float> epoch_mean_losses;
    /// Per-fold validation losses collected for k-fold reporting.
    vector<float> fold_val_losses;

    try
    {
        /// Initialize device early to catch any configuration
        /// or runtime errors before starting the training loop.
        const auto device =
            Device::from_string(config_.device).with_profiling(config_.opencl_profiling_enabled);

        ////////////////////////////
        // Dataset initialization //
        ////////////////////////////

        /// Discover subjects and initialize dataset with specified input mode.
        const auto discovered = discoverSubjects( //
            config_.dataset_root_path,            //
            config_.dataset_subject_filter_regex  //
        );

        // Instantiate dataset using a small builder to keep selection logic
        // centralized and easier to test.
        dataset_ = DatasetBuilder()                 //
                       .with_discovered(discovered) //
                       .with_config(config_)        //
                       .build();

        if (!is_autoencoder_compatible(config_.dataset_type, config_.autoencoder_type)) [[unlikely]]
        {
            ostringstream _oss;
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
            config_.training_batch_size,        //
            config_.sampler_resolved_options    //
        );

        // Store total dataset size for progress tracking.
        dataset_total_samples_ = dataset_->size();

        // Create a printer for the dataset type to log dataset summary information.
        unique_ptr<IDatasetPrinter> printer;
        if (config_.dataset_type == Experiment03DatasetType::Protocol)
        {
            printer = make_unique<Dataset101117Printer>( //
                config_.dataset_root_path                //
            );                                           //
        }
        else
        {
            printer = make_unique<WindowingDatasetPrinter>(  //
                dataset_type_to_string(config_.dataset_type) //
            );                                               //
        }

        // Print dataset summary using the appropriate printer strategy.
        dataset_->print(*printer);

        /////////////////////////////////
        // Training loop setup and run //
        /////////////////////////////////

        // Create loss function and optimizer (optimizer may be initialized
        // eagerly here if we can construct the model from dataset metadata).
        MSELoss loss;
        unique_ptr<Optimizer> optimizer = OptimizerFactory::create( //
            config_.training_optimizer_type,                        //
            config_.training_learning_rate,                         //
            config_.training_optimizer_momentum,                    //
            config_.training_optimizer_adam_beta1,                  //
            config_.training_optimizer_adam_beta2,                  //
            config_.training_optimizer_adam_epsilon                 //
        );

        // Attempt to eagerly construct model and optimizer before entering
        // the epoch loop. Prefer explicit config override, otherwise infer
        // from the first dataset sample when available. If neither is
        // available, defer construction to the first observed batch.

        // Use the dataset's first item to infer input columns.
        config_.autoencoder_input_features = dataset_->get_item(0).inputs.cols();

        // If dataset first item is not available or has zero columns,
        // log a warning and stop.
        if (config_.autoencoder_input_features == 0)
        {
            NN_LOG_WARN("Could not infer input size from metadata");
            return kExitFailure;
        }

        // Build model based on observed input feature size and config.
        model_ = build_autoencoder_model(                          //
            config_,                                               //
            static_cast<Index>(config_.autoencoder_input_features) //
        );

        // Move model to requested device and verify gpu acceleration
        // activity early, if any.
        model_->to(device);

        // Attach model parameters to optimizer.
        optimizer->attach(model_->params());

        /////////////////////////////////////////////////////////////////////
        // K-Fold cross-validation training path.
        // Splits the dataset into kfold_n_splits folds; for each fold:
        //   1. Trains a fresh model on the training subset for training_epochs.
        //   2. Evaluates reconstruction loss on the held-out test subset.
        // Results are reported per-fold and included in the run summary.
        /////////////////////////////////////////////////////////////////////

        // Build fold selections from SQLite trial ids. This keeps fold
        // orchestration outside the training loop while preserving the
        // SqliteBatchSource + BatchPrefetcher fast path.
        const auto fold_selector =
            experiment03::TrialFoldSelector::from_sqlite(config_.dataset_root_path,
                config_.kfold_n_splits,
                config_.kfold_shuffle,
                config_.kfold_seed);

        seen_batches_ = 0;
        processed_samples_ = 0;

        for (size_t fold_idx = 0; fold_idx < fold_selector.fold_count(); ++fold_idx)
        {
            const auto selection = fold_selector.selection_for_fold(fold_idx);
            const auto& train_trial_ids = selection.train_trial_ids;
            const auto& val_trial_ids = selection.val_trial_ids;

            ostringstream fold_header;
            fold_header << "=== K-Fold " << (fold_idx + 1) << "/" << fold_selector.fold_count()
                        << "  train_trials=" << train_trial_ids.size()
                        << "  val_trials=" << val_trial_ids.size() << " ===";
            NN_LOG_INFO(fold_header.str());

            // Fresh model and optimizer for every fold to avoid weight leakage.
            model_ = build_autoencoder_model(                          //
                config_,                                               //
                static_cast<Index>(config_.autoencoder_input_features) //
            );
            model_->to(device);

            unique_ptr<Optimizer> fold_optimizer = OptimizerFactory::create( //
                config_.training_optimizer_type,                             //
                config_.training_learning_rate,                              //
                config_.training_optimizer_momentum,                         //
                config_.training_optimizer_adam_beta1,                       //
                config_.training_optimizer_adam_beta2,                       //
                config_.training_optimizer_adam_epsilon                      //
            );
            fold_optimizer->attach(model_->params());

            // Train for training_epochs on the training subset.
            for (size_t epoch = 0; epoch < config_.training_epochs; ++epoch)
            {
                unique_ptr<IBatchSource> train_src = make_unique<SqliteBatchSource>( //
                    config_.dataset_root_path,                                       //
                    config_.training_batch_size,                                     //
                    to_sqlite_dataset_type(config_.dataset_type),                    //
                    config_.window_eeg_config,                                       //
                    config_.window_audio_config,                                     //
                    config_.dataset_input_mode,                                      //
                    train_trial_ids                                                  //
                );

                const size_t train_epoch_max_batches =
                    effective_fold_max_batches(config_.training_max_batches_per_epoch,
                        train_trial_ids.size(),
                        config_.kfold_n_splits);

                prefetcher_ = make_unique<BatchPrefetcher>(                //
                    std::move(train_src),                                  //
                    train_epoch_max_batches,                               //
                    config_.prefetch_lookahead,                            //
                    config_.prefetch_ram_cap_mb * std::size_t{1024 * 1024} //
                );

                size_t epoch_batches = 0;
                float epoch_loss_sum = 0.0F;

                while (prefetcher_->hasNext())
                {
                    auto maybe_batch = prefetcher_->next();
                    if (!maybe_batch.has_value())
                    {
                        break;
                    }
                    const Batch batch = std::move(maybe_batch.value());

                    if (is_snn_type(config_.autoencoder_type))
                    {
                        model_->reset_state();
                    }

                    auto params = model_->params();
                    fold_optimizer->zero_grad(params);

                    auto reconstruction = model_->forward(batch.inputs, /*requires_grad=*/true);
                    loss.set_target(batch.inputs);
                    auto loss_value = loss.forward(reconstruction, /*requires_grad=*/true);
                    auto d_loss = loss.backward(loss_value);
                    model_->backward(d_loss);
                    fold_optimizer->step(params);

                    epoch_loss_sum += loss_value.at(0, 0);
                    ++epoch_batches;
                    processed_samples_ += static_cast<size_t>(batch.inputs.rows());
                    ++seen_batches_;
                }

                const float mean_loss =
                    epoch_batches > 0 ? epoch_loss_sum / static_cast<float>(epoch_batches) : 0.0F;
                epoch_mean_losses.push_back(mean_loss);

                ostringstream fold_epoch_log;
                fold_epoch_log << "  fold " << (fold_idx + 1) << " epoch " << (epoch + 1) << "/"
                               << config_.training_epochs
                               << "  mean reconstruction loss: " << mean_loss;
                NN_LOG_INFO(fold_epoch_log.str());
            }

            // Evaluate reconstruction loss on the held-out test subset (no backprop).
            unique_ptr<IBatchSource> val_src = make_unique<SqliteBatchSource>( //
                config_.dataset_root_path,                                     //
                config_.training_batch_size,                                   //
                to_sqlite_dataset_type(config_.dataset_type),                  //
                config_.window_eeg_config,                                     //
                config_.window_audio_config,                                   //
                config_.dataset_input_mode,                                    //
                val_trial_ids                                                  //
            );

            const size_t val_max_batches =
                effective_fold_max_batches(config_.training_max_batches_per_epoch,
                    val_trial_ids.size(),
                    config_.kfold_n_splits);

            prefetcher_ = make_unique<BatchPrefetcher>(                //
                std::move(val_src),                                    //
                val_max_batches,                                       //
                config_.prefetch_lookahead,                            //
                config_.prefetch_ram_cap_mb * std::size_t{1024 * 1024} //
            );

            float val_loss_sum = 0.0F;
            size_t val_batches = 0;

            while (prefetcher_->hasNext())
            {
                auto maybe_batch = prefetcher_->next();
                if (!maybe_batch.has_value())
                {
                    break;
                }
                const Batch batch = std::move(maybe_batch.value());

                if (is_snn_type(config_.autoencoder_type))
                {
                    model_->reset_state();
                }

                auto reconstruction = model_->forward(batch.inputs, /*requires_grad=*/false);
                loss.set_target(batch.inputs);
                auto loss_value = loss.forward(reconstruction, /*requires_grad=*/false);
                val_loss_sum += loss_value.at(0, 0);
                ++val_batches;
            }

            const float fold_val_loss =
                val_batches > 0 ? val_loss_sum / static_cast<float>(val_batches) : 0.0F;
            fold_val_losses.push_back(fold_val_loss);

            ostringstream fold_val_log;
            fold_val_log << "  fold " << (fold_idx + 1) << " val loss: " << fold_val_loss;
            NN_LOG_INFO(fold_val_log.str());
        }

        if (seen_batches_ == 0) [[unlikely]]
        {
            NN_LOG_WARN("No batches produced. Check dataset files and row counts.");
        }

        string results_path;
        string results_error;
        auto summary = build_run_summary( //
            config_,                      //
            kExitSuccess,                 //
            dataset_total_samples_,       //
            processed_samples_,           //
            seen_batches_,                //
            epoch_mean_losses,            //
            fold_val_losses               //
        );

        if (write_run_summary_json(summary, results_path, results_error)) [[likely]]
        {
            NN_LOG_INFO(string("Run summary written to: ") + results_path);
        }
        else
        {
            NN_LOG_WARN(string("Warning: failed to write run summary: ") + results_error);
        }

        NN_LOG_INFO("Training complete.");
    }
    catch (const exception& e)
    {
        string results_path;
        string results_error;
        auto summary = build_run_summary( //
            config_,                      //
            kExitFailure,                 //
            dataset_total_samples_,       //
            processed_samples_,           //
            seen_batches_,                //
            epoch_mean_losses,            //
            fold_val_losses,              //
            e.what()                      //
        );
        (void) write_run_summary_json(summary, results_path, results_error);

        NN_LOG_ERROR(string("Error: ") + e.what());
        return kExitFailure;
    }

    return kExitSuccess;
}
