/**
 * @file src/experiments/03/lib/include/ResultsWriter.hpp
 * @brief Resultswriter.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#pragma once

#include <string>
#include <vector>

namespace experiment03
{
struct Summary
{
    std::string profile_name;
    std::string dataset_type;
    std::string autoencoder_type;
    std::string optimizer_type;
    std::string loss_type;
    float optimizer_learning_rate = 0.0F;
    float optimizer_momentum = 0.0F;
    float optimizer_adam_beta1 = 0.0F;
    float optimizer_adam_beta2 = 0.0F;
    float optimizer_adam_epsilon = 0.0F;
    float optimizer_final_learning_rate = 0.0F;
    int exit_code = 0;
    std::size_t total_samples = 0;
    std::size_t processed_samples = 0;
    std::size_t seen_batches = 0;
    /// Per-epoch mean training losses across all folds (concatenated in fold order).
    std::vector<float> epoch_mean_losses;
    // K-fold cross-validation results.
    std::size_t kfold_n_splits = 0;
    /// Per-fold, per-epoch validation mean reconstruction losses: [fold_idx][epoch_idx].
    std::vector<std::vector<float>> fold_epoch_val_losses;
    /// Per-fold, per-epoch EEG-only validation reconstruction losses: [fold_idx][epoch_idx].
    std::vector<std::vector<float>> fold_epoch_val_eeg_losses;
    /// Per-fold, per-epoch Audio-only validation reconstruction losses: [fold_idx][epoch_idx].
    std::vector<std::vector<float>> fold_epoch_val_audio_losses;
    /// Mean validation loss for each fold (average over all epochs within that fold).
    std::vector<float> fold_mean_val_losses;
    /// Grand mean validation loss across all folds (average of fold_mean_val_losses).
    float mean_val_loss = 0.0F;
    std::string error_message;
};

// Write run summary to src/experiments/03/results or ./results when available.
// Returns true on success and fills out_path; otherwise returns false and sets out_error.
auto write_run_summary_json(const Summary& summary, std::string& out_path, std::string& out_error)
    -> bool;
} // namespace experiment03
