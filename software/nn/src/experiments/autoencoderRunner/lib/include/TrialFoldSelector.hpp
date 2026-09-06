/**
 * @file src/experiments/autoencoderRunner/lib/include/TrialFoldSelector.hpp
 * @brief Trial-id fold selector adapter for SQLite-backed k-fold training.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "TrialFoldSelection.hpp"
#include "statistics/kfold.hpp"

namespace autoencoderRunner
{

/// Bridges statistics::KFold (index splits) to SQLite trial ids.
class TrialFoldSelector
{
   public:
    TrialFoldSelector() = default;

    /// Create a selector by loading ordered trial ids from SQLite and splitting them.
    /// @param test_split Fraction of data to hold out as test set (0.0 = no test set).
    static auto from_sqlite(const std::string& dataset_root_path,
        std::size_t n_splits,
        bool shuffle,
        std::optional<unsigned int> seed,
        float test_split = 0.0f) -> TrialFoldSelector;

    /// Number of folds available in this selector.
    [[nodiscard]] auto fold_count() const noexcept -> std::size_t;

    /// Resolve train/validation trial ids for one fold index.
    [[nodiscard]] auto selection_for_fold(std::size_t fold_idx) const -> TrialFoldSelection;

    /// Get the test set trial ids (same for all folds).
    [[nodiscard]] auto test_trial_ids() const noexcept -> const std::vector<int>&;

   private:
    TrialFoldSelector(std::vector<int> trial_ids,
        std::vector<statistics::FoldSplit> folds,
        std::vector<int> test_trial_ids);

    std::vector<int> trial_ids_;
    std::vector<statistics::FoldSplit> folds_;
    std::vector<int> test_trial_ids_;
};
} // namespace autoencoderRunner
