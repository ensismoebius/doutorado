/**
 * @file src/experiments/03/lib/include/RunSummaryBuilder.hpp
 * @brief Runsummarybuilder.
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

#include "Experiment03Config.hpp"
#include "ResultsWriter.hpp"

namespace experiment03
{
auto build_run_summary(const Config& config,
    int exit_code,
    size_t total_samples,
    size_t processed_samples,
    size_t seen_batches,
    const std::vector<float>& epoch_mean_losses,
    const std::vector<std::vector<float>>& fold_epoch_val_losses = {},
    const std::vector<std::vector<float>>& fold_epoch_val_eeg_losses = {},
    const std::vector<std::vector<float>>& fold_epoch_val_audio_losses = {},
    const std::vector<float>& fold_mean_val_losses = {},
    float mean_val_loss = 0.0F,
    float optimizer_final_learning_rate = 0.0F,
    const std::string& error_message = "") -> Summary;

} // namespace experiment03
