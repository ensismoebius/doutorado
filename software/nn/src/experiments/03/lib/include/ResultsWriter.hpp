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
    float optimizer_learning_rate = 0.0F;
    float optimizer_momentum = 0.0F;
    float optimizer_adam_beta1 = 0.0F;
    float optimizer_adam_beta2 = 0.0F;
    float optimizer_adam_epsilon = 0.0F;
    int exit_code = 0;
    std::size_t total_samples = 0;
    std::size_t processed_samples = 0;
    std::size_t seen_batches = 0;
    std::vector<float> epoch_mean_losses;
    // K-fold cross-validation results. Empty when kfold is disabled.
    bool kfold_enabled = false;
    std::size_t kfold_n_splits = 0;
    std::vector<float> fold_val_losses; // Per-fold validation mean reconstruction loss.
    std::string error_message;
};

// Write run summary to src/experiments/03/results or ./results when available.
// Returns true on success and fills out_path; otherwise returns false and sets out_error.
auto write_run_summary_json(const Summary& summary, std::string& out_path, std::string& out_error)
    -> bool;
} // namespace experiment03
