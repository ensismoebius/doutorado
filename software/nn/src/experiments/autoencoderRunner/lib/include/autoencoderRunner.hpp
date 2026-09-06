/**
 * @file src/experiments/autoencoderRunner/lib/include/autoencoderRunner.hpp
 * @brief Public experiment driver API for AutoencoderRunner.
 *
 * Declares the `AutoencoderRunner` class which orchestrates dataset discovery,
 * data loading, model construction and training loop. Consumers should
 * configure the experiment via the `Config` structure in `cli.hpp`.
 */

#pragma once

#include <exception>

#include "AutoencoderRunnerConfig.hpp"
#include "Backend.hpp"
#include "TrialFoldSelector.hpp"
#include "autoencoderRunner_helpers.hpp"
#include "data_loaders/datasets/Dataset.hpp"
#include "data_loaders/runtime/BatchPrefetcher.hpp"
#include "data_loaders/runtime/DataLoader.hpp"
#include "device/Device.hpp"
#include "layers/base/Module.hpp"
#include "optimizers/Optimizer.hpp"
#include "utility/ITransform.hpp"

class AutoencoderRunner
{
   public:
    explicit AutoencoderRunner(const Config& config);
    // Run the experiment; returns 0 on success, non-zero on failure.
    int run();

   private:
    Config config_;
    // Dataset and pipeline components are stored as members so they
    // persist across `run()` and can be inspected or reused.
    // dataset_ uses base class pointer to support multiple dataset types.
    std::unique_ptr<BatchPrefetcher> prefetcher_;
    std::unique_ptr<DataLoader> data_loader_;
    std::shared_ptr<Dataset> dataset_;
    std::unique_ptr<Module<nn::Backend>> model_;

    std::size_t seen_batches_ = 0;
    std::size_t processed_samples_ = 0;
    std::size_t dataset_total_samples_ = 0;

    // Per-epoch validation loss summary, returned by validate_one_epoch().
    struct EpochValidationResult
    {
        float mean_loss = 0.0F;
        float mean_eeg_loss = 0.0F;
        float mean_audio_loss = 0.0F;
    };

    // Discovers subjects, builds `dataset_`/`data_loader_`, and prints the dataset summary.
    void initialize_dataset();

    // Infers the input feature size from the dataset, builds `model_`, moves it to `device`,
    // and attaches its parameters to `optimizer`. Returns false (without touching `model_`
    // further) if the input feature size could not be inferred, matching the caller's
    // pre-existing "return kExitFailure immediately, no run summary written" behavior.
    bool build_and_attach_model(const nn::Device& device, Optimizer& optimizer);

    // Builds the trial-fold selector for the test split and returns the held-out test trial ids.
    auto setup_fold_selector(std::vector<int>& test_trial_ids) const
        -> autoencoderRunner::TrialFoldSelector;

    // Runs one training epoch over `trial_ids` (or the full corpus when nullptr) and reports
    // progress via postProgressAsync using the supplied progress-bar bookkeeping. When
    // `also_update_member_counters` is true, `seen_batches_`/`processed_samples_` are updated
    // in addition to `progress_seen_batches`/`progress_processed_samples` (used by the k-fold
    // path, which tracks both per-fold-run-global and lifetime counters).
    auto train_one_epoch(Optimizer& optimizer,
        autoencoderRunner::ReconstructionLoss& loss,
        const std::shared_ptr<nn::transforms::ITransform>& input_transform,
        const std::vector<int>* trial_ids,
        std::size_t epoch_max_batches,
        std::size_t progress_total_samples,
        std::size_t progress_total_batches,
        std::size_t& progress_seen_batches,
        std::size_t& progress_processed_samples,
        std::size_t fold_number_1based,
        std::size_t fold_count,
        std::size_t epoch_index_0based,
        std::size_t total_epochs,
        const std::string& progress_context,
        bool also_update_member_counters) -> float;

    // Runs one validation epoch (no backprop) over `trial_ids` (or the full corpus when
    // nullptr) and returns the mean reconstruction/eeg/audio losses across its batches.
    auto validate_one_epoch(autoencoderRunner::ReconstructionLoss& loss,
        const std::shared_ptr<nn::transforms::ITransform>& input_transform,
        const std::vector<int>* trial_ids,
        std::size_t max_batches) -> EpochValidationResult;

    // Trains + validates a single k-fold cross-validation fold: fresh model/optimizer, fits
    // the input normalizer on the fold's training split, then runs `training_epochs` of
    // train_one_epoch()/validate_one_epoch(), appending losses into the output parameters.
    void train_and_validate_one_fold(const nn::Device& device,
        autoencoderRunner::ReconstructionLoss& loss,
        const autoencoderRunner::TrialFoldSelection& selection,
        std::size_t train_epoch_max_batches,
        std::size_t val_epoch_max_batches,
        std::size_t fold_number_1based,
        std::size_t fold_count,
        std::size_t global_training_total_samples,
        std::size_t global_training_total_batches,
        std::size_t& global_training_seen_batches,
        std::size_t& global_training_processed_samples,
        std::vector<float>& epoch_mean_losses,
        std::vector<float>& fold_epoch_val_losses_this_fold,
        std::vector<float>& fold_epoch_val_eeg_losses_this_fold,
        std::vector<float>& fold_epoch_val_audio_losses_this_fold,
        std::vector<float>& fold_mean_val_losses);

    // Runs the k-fold cross-validation training path: trains + validates a fresh model per
    // fold and appends the per-fold/per-epoch loss series into the output parameters.
    void run_kfold_training(const nn::Device& device,
        autoencoderRunner::ReconstructionLoss& loss,
        autoencoderRunner::TrialFoldSelector& fold_selector,
        std::vector<float>& epoch_mean_losses,
        std::vector<std::vector<float>>& fold_epoch_val_losses,
        std::vector<std::vector<float>>& fold_epoch_val_eeg_losses,
        std::vector<std::vector<float>>& fold_epoch_val_audio_losses,
        std::vector<float>& fold_mean_val_losses,
        float& mean_val_loss);

    // Runs the single (no k-fold) training path over the full corpus.
    void run_no_kfold_training(const nn::Device& device,
        autoencoderRunner::ReconstructionLoss& loss,
        Optimizer& optimizer,
        std::vector<float>& epoch_mean_losses,
        std::vector<std::vector<float>>& fold_epoch_val_losses,
        std::vector<std::vector<float>>& fold_epoch_val_eeg_losses,
        std::vector<std::vector<float>>& fold_epoch_val_audio_losses,
        std::vector<float>& fold_mean_val_losses,
        float& mean_val_loss);

    // Builds the success-path run summary, writes it to disk, and logs the outcome.
    void write_success_summary(Optimizer& optimizer,
        const std::vector<float>& epoch_mean_losses,
        const std::vector<std::vector<float>>& fold_epoch_val_losses,
        const std::vector<std::vector<float>>& fold_epoch_val_eeg_losses,
        const std::vector<std::vector<float>>& fold_epoch_val_audio_losses,
        const std::vector<float>& fold_mean_val_losses,
        float mean_val_loss,
        const std::vector<int>& test_trial_ids);

    // Builds the failure-path run summary (annotated with the exception message), writes it
    // to disk, and logs the error.
    void write_failure_summary(const std::exception& e,
        const std::vector<float>& epoch_mean_losses,
        const std::vector<std::vector<float>>& fold_epoch_val_losses,
        const std::vector<std::vector<float>>& fold_epoch_val_eeg_losses,
        const std::vector<std::vector<float>>& fold_epoch_val_audio_losses,
        const std::vector<float>& fold_mean_val_losses,
        float mean_val_loss,
        const std::vector<int>& test_trial_ids);
};
