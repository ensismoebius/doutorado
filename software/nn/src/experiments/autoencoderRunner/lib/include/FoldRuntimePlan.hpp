/**
 * @file src/experiments/autoencoderRunner/lib/include/FoldRuntimePlan.hpp
 * @brief FoldRuntimePlan struct + fold runtime-plan helpers (extracted from
 *        autoencoderRunner.cpp). Global scope, matching the scope the
 *        original anonymous namespace occupied.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "TrialFoldSelection.hpp"
#include "TrialFoldSelector.hpp"
#include "cli.hpp"

inline auto effective_fold_max_batches(
    std::size_t configured_max_batches, std::size_t available_batches) -> std::size_t
{
    if (configured_max_batches == 0)
    {
        return std::max<std::size_t>(1, available_batches);
    }

    return std::max<std::size_t>(1, std::min(configured_max_batches, available_batches));
}

/// Per-fold batch budget, derived once before the k-fold training loop starts.
struct FoldRuntimePlan
{
    autoencoderRunner::TrialFoldSelection selection;
    std::size_t train_epoch_max_batches = 0;
    std::size_t val_max_batches = 0;
};

/// Computes each fold's training/validation selection and per-epoch batch budget, plus the
/// total number of training batches across all folds (used for global progress reporting).
inline auto build_fold_runtime_plans(autoencoderRunner::TrialFoldSelector& fold_selector,
    const Config& config) -> std::pair<std::vector<FoldRuntimePlan>, std::size_t>
{
    std::vector<FoldRuntimePlan> fold_plans;
    fold_plans.reserve(fold_selector.fold_count());

    std::size_t global_training_total_batches = 0;
    for (std::size_t fold_idx = 0; fold_idx < fold_selector.fold_count(); ++fold_idx)
    {
        auto selection = fold_selector.selection_for_fold(fold_idx);
        const std::size_t train_epoch_max_batches = effective_fold_max_batches(
            config.training_max_batches_per_epoch, selection.train_trial_ids.size());
        const std::size_t val_max_batches = effective_fold_max_batches(
            config.training_max_batches_per_epoch, selection.val_trial_ids.size());

        global_training_total_batches += train_epoch_max_batches * config.training_epochs;
        fold_plans.push_back(
            FoldRuntimePlan{std::move(selection), train_epoch_max_batches, val_max_batches});
    }

    return {std::move(fold_plans), global_training_total_batches};
}
