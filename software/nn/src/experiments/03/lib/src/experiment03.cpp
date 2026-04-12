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
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
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
#include "nn/optimizers/Adam.hpp"
#include "nn/optimizers/Optimizer.hpp"
#include "nn/optimizers/OptimizerFactory.hpp"
#include "nn/optimizers/SGD.hpp"
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

auto effective_fold_max_batches(size_t configured_max_batches, size_t available_batches) -> size_t
{
    if (configured_max_batches == 0)
    {
        return std::max<size_t>(1, available_batches);
    }

    return std::max<size_t>(1, std::min(configured_max_batches, available_batches));
}

struct ReduceLROnPlateauState
{
    float best_val_loss = std::numeric_limits<float>::infinity();
    size_t bad_epochs = 0;
};

auto optimizer_learning_rate_ptr(Optimizer& optimizer) -> float*
{
    if (auto* adam = dynamic_cast<Adam*>(&optimizer))
    {
        return &adam->learning_rate;
    }
    if (auto* sgd = dynamic_cast<SGD*>(&optimizer))
    {
        return &sgd->learning_rate;
    }
    return nullptr;
}

auto apply_reduce_lr_on_plateau(
    Optimizer& optimizer, const Config& config, ReduceLROnPlateauState& state, float epoch_val_loss)
    -> float
{
    float* learning_rate = optimizer_learning_rate_ptr(optimizer);
    if (learning_rate == nullptr || !config.training_lr_plateau_enabled)
    {
        return learning_rate != nullptr ? *learning_rate : config.training_learning_rate;
    }

    const bool improved =
        epoch_val_loss <
        (state.best_val_loss - std::max(0.0F, config.training_lr_plateau_min_delta));
    if (improved)
    {
        state.best_val_loss = epoch_val_loss;
        state.bad_epochs = 0;
        return *learning_rate;
    }

    ++state.bad_epochs;
    if (state.bad_epochs < std::max<size_t>(1, config.training_lr_plateau_patience))
    {
        return *learning_rate;
    }

    const float factor = std::clamp(config.training_lr_plateau_factor, 0.0F, 1.0F);
    const float new_lr = std::max(1e-8F, (*learning_rate) * factor);
    if (new_lr < *learning_rate)
    {
        ostringstream lr_log;
        lr_log << "ReduceLROnPlateau: val loss plateau detected, lr " << *learning_rate << " -> "
               << new_lr;
        NN_LOG_INFO(lr_log.str());
        *learning_rate = new_lr;
    }
    state.bad_epochs = 0;
    return *learning_rate;
}

auto modality_val_losses_from_batch(const Batch& val_batch,
    const nn::Tensor& val_reconstruction,
    size_t eeg_features,
    size_t audio_features) -> std::pair<float, float>
{
    if (eeg_features == 0 || audio_features == 0)
    {
        return {0.0F, 0.0F};
    }
    const size_t total_features = eeg_features + audio_features;
    if (static_cast<size_t>(val_batch.inputs.cols()) < total_features ||
        static_cast<size_t>(val_reconstruction.cols()) < total_features)
    {
        return {0.0F, 0.0F};
    }

    const auto eeg_target =
        val_batch.inputs.block(0, 0, val_batch.inputs.rows(), static_cast<Index>(eeg_features));
    const auto eeg_pred =
        val_reconstruction.block(0, 0, val_reconstruction.rows(), static_cast<Index>(eeg_features));

    const auto audio_target = val_batch.inputs.block(0,
        static_cast<Index>(eeg_features),
        val_batch.inputs.rows(),
        static_cast<Index>(audio_features));
    const auto audio_pred = val_reconstruction.block(0,
        static_cast<Index>(eeg_features),
        val_reconstruction.rows(),
        static_cast<Index>(audio_features));

    return {eeg_pred.mean_squared_error(eeg_target), audio_pred.mean_squared_error(audio_target)};
}

class MAELoss final : public Module<nn::EigenTensorBackend>
{
   public:
    using Tensor = Module<nn::EigenTensorBackend>::Tensor;

    void set_target(const Tensor& target)
    {
        target_ = target;
        target_set_ = true;
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        if (!target_set_)
        {
            throw std::runtime_error("MAELoss: target has not been set. Call set_target() first.");
        }
        if (requires_grad)
        {
            last_input_ = input;
        }
        Tensor diff = input - target_;
        Tensor loss_tensor(1, 1);
        loss_tensor.at(0, 0) = diff.abs().mean();
        return loss_tensor;
    }

    auto backward(const Tensor& /*grad_output*/) -> Tensor override
    {
        Tensor grad(last_input_.rows(), last_input_.cols());
        const float factor = 1.0F / static_cast<float>(std::max<Index>(1, last_input_.size()));
        for (Index i = 0; i < last_input_.rows(); ++i)
        {
            for (Index j = 0; j < last_input_.cols(); ++j)
            {
                const float diff = last_input_.at(i, j) - target_.at(i, j);
                grad.at(i, j) = diff > 0.0F ? factor : (diff < 0.0F ? -factor : 0.0F);
            }
        }
        return grad;
    }

   private:
    Tensor last_input_;
    Tensor target_;
    bool target_set_ = false;
};

class ReconstructionLoss
{
   public:
    explicit ReconstructionLoss(const std::string& loss_type)
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

    auto set_target(const nn::Tensor& target) -> void
    {
        if (mse_) mse_->set_target(target);
        if (mae_) mae_->set_target(target);
    }

    auto forward(const nn::Tensor& input, bool requires_grad) -> nn::Tensor
    {
        if (mse_) return mse_->forward(input, requires_grad);
        return mae_->forward(input, requires_grad);
    }

    auto backward(const nn::Tensor& grad_output) -> nn::Tensor
    {
        if (mse_) return mse_->backward(grad_output);
        return mae_->backward(grad_output);
    }

   private:
    std::unique_ptr<MSELoss> mse_;
    std::unique_ptr<MAELoss> mae_;
};
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
    /// Per-fold, per-epoch validation losses: [fold_idx][epoch_idx].
    vector<vector<float>> fold_epoch_val_losses;
    /// Per-fold, per-epoch EEG validation reconstruction losses.
    vector<vector<float>> fold_epoch_val_eeg_losses;
    /// Per-fold, per-epoch Audio validation reconstruction losses.
    vector<vector<float>> fold_epoch_val_audio_losses;
    /// Mean validation loss per fold (mean over all epochs within that fold).
    vector<float> fold_mean_val_losses;
    /// Grand mean validation loss across all folds.
    float mean_val_loss = 0.0F;

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
        ReconstructionLoss loss(config_.training_loss_type);
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

        seen_batches_ = 0;
        processed_samples_ = 0;

        if (config_.kfold_enabled)
        {
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

            struct FoldRuntimePlan
            {
                experiment03::TrialFoldSelection selection;
                size_t train_epoch_max_batches = 0;
                size_t val_max_batches = 0;
            };

            vector<FoldRuntimePlan> fold_plans;
            fold_plans.reserve(fold_selector.fold_count());

            size_t global_training_total_batches = 0;
            for (size_t fold_idx = 0; fold_idx < fold_selector.fold_count(); ++fold_idx)
            {
                auto selection = fold_selector.selection_for_fold(fold_idx);
                const size_t train_epoch_max_batches = effective_fold_max_batches(
                    config_.training_max_batches_per_epoch, selection.train_trial_ids.size());
                const size_t val_max_batches = effective_fold_max_batches(
                    config_.training_max_batches_per_epoch, selection.val_trial_ids.size());

                global_training_total_batches += train_epoch_max_batches * config_.training_epochs;
                fold_plans.push_back(FoldRuntimePlan{
                    std::move(selection), train_epoch_max_batches, val_max_batches});
            }

            const size_t global_training_total_samples =
                global_training_total_batches * config_.training_batch_size;
            const size_t global_total_epochs = fold_selector.fold_count() * config_.training_epochs;

            size_t global_training_seen_batches = 0;
            size_t global_training_processed_samples = 0;

            // Pre-allocate one inner vector per fold to receive per-epoch val losses.
            fold_epoch_val_losses.resize(fold_selector.fold_count());
            fold_epoch_val_eeg_losses.resize(fold_selector.fold_count());
            fold_epoch_val_audio_losses.resize(fold_selector.fold_count());

            for (size_t fold_idx = 0; fold_idx < fold_selector.fold_count(); ++fold_idx)
            {
                const auto& selection = fold_plans[fold_idx].selection;
                const auto& train_trial_ids = selection.train_trial_ids;
                const auto& val_trial_ids = selection.val_trial_ids;
                const string fold_progress_context = string("Fold ") +
                                                     std::to_string(fold_idx + 1) + "/" +
                                                     std::to_string(fold_selector.fold_count());

                ostringstream fold_header;
                fold_header << "=== K-Fold " << (fold_idx + 1) << "/" << fold_selector.fold_count()
                            << "  train_trials=" << train_trial_ids.size()
                            << "  val_trials=" << val_trial_ids.size()
                            << "  train_epoch_max_batches="
                            << fold_plans[fold_idx].train_epoch_max_batches
                            << "  val_max_batches=" << fold_plans[fold_idx].val_max_batches
                            << " ===";
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
                ReduceLROnPlateauState fold_lr_state{};

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
                        fold_plans[fold_idx].train_epoch_max_batches;

                    prefetcher_ = make_unique<BatchPrefetcher>(                //
                        std::move(train_src),                                  //
                        train_epoch_max_batches,                               //
                        config_.prefetch_lookahead,                            //
                        config_.prefetch_ram_cap_mb * std::size_t{1024 * 1024} //
                    );

                    size_t epoch_batches = 0;
                    float epoch_loss_sum = 0.0F;
                    double last_batch_loss = 0.0;
                    const size_t global_epoch_index =
                        fold_idx * config_.training_epochs + epoch + 1;

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
                        last_batch_loss = static_cast<double>(loss_value.at(0, 0));
                        ++epoch_batches;
                        processed_samples_ += static_cast<size_t>(batch.inputs.rows());
                        ++seen_batches_;
                        global_training_processed_samples +=
                            static_cast<size_t>(batch.inputs.rows());
                        ++global_training_seen_batches;

                        printProgress(global_training_total_samples,
                            config_.training_batch_size,
                            global_training_total_batches,
                            global_training_seen_batches,
                            global_training_processed_samples,
                            false,
                            global_epoch_index,
                            global_total_epochs,
                            last_batch_loss,
                            std::span<nn::Tensor*>{},
                            fold_progress_context);
                    }

                    if (model_) [[likely]]
                    {
                        const bool global_done = (fold_idx + 1 == fold_selector.fold_count()) &&
                                                 (epoch + 1 == config_.training_epochs);

                        printProgress(global_training_total_samples,
                            config_.training_batch_size,
                            global_training_total_batches,
                            global_training_seen_batches,
                            global_training_processed_samples,
                            global_done,
                            global_epoch_index,
                            global_total_epochs,
                            last_batch_loss,
                            global_done ? model_->params() : std::span<nn::Tensor*>{},
                            fold_progress_context);
                    }

                    const float mean_train_loss =
                        epoch_batches > 0 ? epoch_loss_sum / static_cast<float>(epoch_batches)
                                          : 0.0F;
                    epoch_mean_losses.push_back(mean_train_loss);

                    // Validate this epoch on the held-out validation subset (no backprop).
                    unique_ptr<IBatchSource> val_src = make_unique<SqliteBatchSource>( //
                        config_.dataset_root_path,                                     //
                        config_.training_batch_size,                                   //
                        to_sqlite_dataset_type(config_.dataset_type),                  //
                        config_.window_eeg_config,                                     //
                        config_.window_audio_config,                                   //
                        config_.dataset_input_mode,                                    //
                        val_trial_ids                                                  //
                    );
                    auto val_prefetcher = make_unique<BatchPrefetcher>(        //
                        std::move(val_src),                                    //
                        fold_plans[fold_idx].val_max_batches,                  //
                        config_.prefetch_lookahead,                            //
                        config_.prefetch_ram_cap_mb * std::size_t{1024 * 1024} //
                    );
                    float val_loss_sum = 0.0F;
                    size_t val_batches = 0;
                    float val_eeg_loss_sum = 0.0F;
                    float val_audio_loss_sum = 0.0F;
                    const size_t eeg_features = static_cast<size_t>(
                        std::max(0, config_.effective_autoencoder_eeg_features()));
                    const size_t audio_features = static_cast<size_t>(
                        std::max(0, config_.effective_autoencoder_audio_features()));
                    while (val_prefetcher->hasNext())
                    {
                        auto maybe_val_batch = val_prefetcher->next();
                        if (!maybe_val_batch.has_value())
                        {
                            break;
                        }
                        const Batch val_batch = std::move(maybe_val_batch.value());
                        if (is_snn_type(config_.autoencoder_type))
                        {
                            model_->reset_state();
                        }
                        auto val_reconstruction =
                            model_->forward(val_batch.inputs, /*requires_grad=*/false);
                        loss.set_target(val_batch.inputs);
                        auto val_loss_value =
                            loss.forward(val_reconstruction, /*requires_grad=*/false);
                        val_loss_sum += val_loss_value.at(0, 0);
                        if (config_.validation_modality_diagnostics_enabled)
                        {
                            const auto [batch_eeg_loss, batch_audio_loss] =
                                modality_val_losses_from_batch(
                                    val_batch, val_reconstruction, eeg_features, audio_features);
                            val_eeg_loss_sum += batch_eeg_loss;
                            val_audio_loss_sum += batch_audio_loss;
                        }
                        ++val_batches;
                    }
                    const float epoch_val_loss =
                        val_batches > 0 ? val_loss_sum / static_cast<float>(val_batches) : 0.0F;
                    const float epoch_val_eeg_loss =
                        val_batches > 0 ? val_eeg_loss_sum / static_cast<float>(val_batches) : 0.0F;
                    const float epoch_val_audio_loss =
                        val_batches > 0 ? val_audio_loss_sum / static_cast<float>(val_batches)
                                        : 0.0F;
                    fold_epoch_val_losses[fold_idx].push_back(epoch_val_loss);
                    fold_epoch_val_eeg_losses[fold_idx].push_back(epoch_val_eeg_loss);
                    fold_epoch_val_audio_losses[fold_idx].push_back(epoch_val_audio_loss);

                    const float current_lr = apply_reduce_lr_on_plateau(
                        *fold_optimizer, config_, fold_lr_state, epoch_val_loss);

                    ostringstream fold_epoch_log;
                    fold_epoch_log << "  fold " << (fold_idx + 1) << "/"
                                   << fold_selector.fold_count() << " epoch " << (epoch + 1) << "/"
                                   << config_.training_epochs << "  train loss: " << mean_train_loss
                                   << "  val loss: " << epoch_val_loss;
                    if (config_.validation_modality_diagnostics_enabled)
                    {
                        fold_epoch_log << "  val_eeg: " << epoch_val_eeg_loss
                                       << "  val_audio: " << epoch_val_audio_loss;
                    }
                    fold_epoch_log << "  lr: " << current_lr;
                    NN_LOG_INFO(fold_epoch_log.str());
                }

                // Compute mean val loss for this fold from all per-epoch val losses.
                const auto& fold_val_epochs = fold_epoch_val_losses[fold_idx];
                const float fold_mean_val =
                    fold_val_epochs.empty()
                        ? 0.0F
                        : std::accumulate(fold_val_epochs.begin(), fold_val_epochs.end(), 0.0F) /
                              static_cast<float>(fold_val_epochs.size());
                fold_mean_val_losses.push_back(fold_mean_val);

                ostringstream fold_summary_log;
                fold_summary_log << "  fold " << (fold_idx + 1) << "/" << fold_selector.fold_count()
                                 << " complete: mean val loss = " << fold_mean_val;
                NN_LOG_INFO(fold_summary_log.str());
            }

            // Compute grand mean validation loss across all folds.
            mean_val_loss =
                fold_mean_val_losses.empty()
                    ? 0.0F
                    : std::accumulate(
                          fold_mean_val_losses.begin(), fold_mean_val_losses.end(), 0.0F) /
                          static_cast<float>(fold_mean_val_losses.size());

            ostringstream kfold_summary_log;
            kfold_summary_log << "K-Fold complete: grand mean val loss = " << mean_val_loss;
            NN_LOG_INFO(kfold_summary_log.str());
        }
        else
        {
            NN_LOG_INFO("K-Fold disabled: running single training loop.");

            // Reuse k-fold summary fields with a single logical fold in no-kfold mode
            // so downstream comparisons can consume the same JSON schema.
            fold_epoch_val_losses.assign(1, {});
            fold_epoch_val_eeg_losses.assign(1, {});
            fold_epoch_val_audio_losses.assign(1, {});
            ReduceLROnPlateauState lr_state{};

            const size_t epoch_max_batches =
                config_.training_max_batches_per_epoch == 0
                    ? std::max<size_t>(1,
                          (dataset_total_samples_ + config_.training_batch_size - 1) /
                              config_.training_batch_size)
                    : config_.training_max_batches_per_epoch;

            const size_t total_training_batches = epoch_max_batches * config_.training_epochs;
            const size_t total_training_samples =
                total_training_batches * config_.training_batch_size;

            for (size_t epoch = 0; epoch < config_.training_epochs; ++epoch)
            {
                unique_ptr<IBatchSource> src = make_unique<SqliteBatchSource>( //
                    config_.dataset_root_path,                                 //
                    config_.training_batch_size,                               //
                    to_sqlite_dataset_type(config_.dataset_type),              //
                    config_.window_eeg_config,                                 //
                    config_.window_audio_config,                               //
                    config_.dataset_input_mode                                 //
                );

                prefetcher_ = make_unique<BatchPrefetcher>(                //
                    std::move(src),                                        //
                    epoch_max_batches,                                     //
                    config_.prefetch_lookahead,                            //
                    config_.prefetch_ram_cap_mb * std::size_t{1024 * 1024} //
                );

                size_t epoch_batches = 0;
                float epoch_loss_sum = 0.0F;
                double last_batch_loss = 0.0;

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
                    optimizer->zero_grad(params);

                    auto reconstruction = model_->forward(batch.inputs, /*requires_grad=*/true);
                    loss.set_target(batch.inputs);
                    auto loss_value = loss.forward(reconstruction, /*requires_grad=*/true);
                    auto d_loss = loss.backward(loss_value);
                    model_->backward(d_loss);
                    optimizer->step(params);

                    epoch_loss_sum += loss_value.at(0, 0);
                    last_batch_loss = static_cast<double>(loss_value.at(0, 0));
                    ++epoch_batches;
                    processed_samples_ += static_cast<size_t>(batch.inputs.rows());
                    ++seen_batches_;

                    printProgress(total_training_samples,
                        config_.training_batch_size,
                        total_training_batches,
                        seen_batches_,
                        processed_samples_,
                        false,
                        epoch + 1,
                        config_.training_epochs,
                        last_batch_loss,
                        std::span<nn::Tensor*>{});
                }

                if (model_) [[likely]]
                {
                    const bool done = (epoch + 1 == config_.training_epochs);
                    printProgress(total_training_samples,
                        config_.training_batch_size,
                        total_training_batches,
                        seen_batches_,
                        processed_samples_,
                        done,
                        epoch + 1,
                        config_.training_epochs,
                        last_batch_loss,
                        done ? model_->params() : std::span<nn::Tensor*>{});
                }

                const float mean_train_loss =
                    epoch_batches > 0 ? epoch_loss_sum / static_cast<float>(epoch_batches) : 0.0F;
                epoch_mean_losses.push_back(mean_train_loss);

                // Validate this epoch using the same protocol and budget in no-kfold mode.
                unique_ptr<IBatchSource> val_src = make_unique<SqliteBatchSource>( //
                    config_.dataset_root_path,                                     //
                    config_.training_batch_size,                                   //
                    to_sqlite_dataset_type(config_.dataset_type),                  //
                    config_.window_eeg_config,                                     //
                    config_.window_audio_config,                                   //
                    config_.dataset_input_mode                                     //
                );
                auto val_prefetcher = make_unique<BatchPrefetcher>(        //
                    std::move(val_src),                                    //
                    epoch_max_batches,                                     //
                    config_.prefetch_lookahead,                            //
                    config_.prefetch_ram_cap_mb * std::size_t{1024 * 1024} //
                );
                float val_loss_sum = 0.0F;
                size_t val_batches = 0;
                float val_eeg_loss_sum = 0.0F;
                float val_audio_loss_sum = 0.0F;
                const size_t eeg_features =
                    static_cast<size_t>(std::max(0, config_.effective_autoencoder_eeg_features()));
                const size_t audio_features = static_cast<size_t>(
                    std::max(0, config_.effective_autoencoder_audio_features()));
                while (val_prefetcher->hasNext())
                {
                    auto maybe_val_batch = val_prefetcher->next();
                    if (!maybe_val_batch.has_value())
                    {
                        break;
                    }
                    const Batch val_batch = std::move(maybe_val_batch.value());
                    if (is_snn_type(config_.autoencoder_type))
                    {
                        model_->reset_state();
                    }
                    auto val_reconstruction =
                        model_->forward(val_batch.inputs, /*requires_grad=*/false);
                    loss.set_target(val_batch.inputs);
                    auto val_loss_value = loss.forward(val_reconstruction, /*requires_grad=*/false);
                    val_loss_sum += val_loss_value.at(0, 0);
                    if (config_.validation_modality_diagnostics_enabled)
                    {
                        const auto [batch_eeg_loss, batch_audio_loss] =
                            modality_val_losses_from_batch(
                                val_batch, val_reconstruction, eeg_features, audio_features);
                        val_eeg_loss_sum += batch_eeg_loss;
                        val_audio_loss_sum += batch_audio_loss;
                    }
                    ++val_batches;
                }
                const float epoch_val_loss =
                    val_batches > 0 ? val_loss_sum / static_cast<float>(val_batches) : 0.0F;
                const float epoch_val_eeg_loss =
                    val_batches > 0 ? val_eeg_loss_sum / static_cast<float>(val_batches) : 0.0F;
                const float epoch_val_audio_loss =
                    val_batches > 0 ? val_audio_loss_sum / static_cast<float>(val_batches) : 0.0F;
                fold_epoch_val_losses[0].push_back(epoch_val_loss);
                fold_epoch_val_eeg_losses[0].push_back(epoch_val_eeg_loss);
                fold_epoch_val_audio_losses[0].push_back(epoch_val_audio_loss);

                const float current_lr =
                    apply_reduce_lr_on_plateau(*optimizer, config_, lr_state, epoch_val_loss);

                ostringstream epoch_log;
                epoch_log << "epoch " << (epoch + 1) << "/" << config_.training_epochs
                          << " train loss: " << mean_train_loss << "  val loss: " << epoch_val_loss;
                if (config_.validation_modality_diagnostics_enabled)
                {
                    epoch_log << "  val_eeg: " << epoch_val_eeg_loss
                              << "  val_audio: " << epoch_val_audio_loss;
                }
                epoch_log << "  lr: " << current_lr;
                NN_LOG_INFO(epoch_log.str());
            }

            const auto& single_fold_val_epochs = fold_epoch_val_losses[0];
            const float single_fold_mean_val =
                single_fold_val_epochs.empty()
                    ? 0.0F
                    : std::accumulate(
                          single_fold_val_epochs.begin(), single_fold_val_epochs.end(), 0.0F) /
                          static_cast<float>(single_fold_val_epochs.size());
            fold_mean_val_losses.push_back(single_fold_mean_val);
            mean_val_loss = single_fold_mean_val;

            ostringstream single_summary_log;
            single_summary_log << "No-KFold complete: mean val loss = " << mean_val_loss;
            NN_LOG_INFO(single_summary_log.str());
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
            fold_epoch_val_losses,        //
            fold_epoch_val_eeg_losses,    //
            fold_epoch_val_audio_losses,  //
            fold_mean_val_losses,         //
            mean_val_loss,                //
            optimizer_learning_rate_ptr(*optimizer) != nullptr
                ? *optimizer_learning_rate_ptr(*optimizer)
                : config_.training_learning_rate //
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
        auto summary = build_run_summary(   //
            config_,                        //
            kExitFailure,                   //
            dataset_total_samples_,         //
            processed_samples_,             //
            seen_batches_,                  //
            epoch_mean_losses,              //
            fold_epoch_val_losses,          //
            fold_epoch_val_eeg_losses,      //
            fold_epoch_val_audio_losses,    //
            fold_mean_val_losses,           //
            mean_val_loss,                  //
            config_.training_learning_rate, //
            e.what()                        //
        );
        (void) write_run_summary_json(summary, results_path, results_error);

        NN_LOG_ERROR(string("Error: ") + e.what());
        return kExitFailure;
    }

    return kExitSuccess;
}
