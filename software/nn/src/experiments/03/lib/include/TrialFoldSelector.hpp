/**
 * @file src/experiments/03/lib/include/TrialFoldSelector.hpp
 * @brief Trial-id fold selector adapter for SQLite-backed k-fold training.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "nn/statistics/kfold.hpp"

namespace experiment03
{
struct TrialFoldSelection
{
    std::vector<int> train_trial_ids;
    std::vector<int> val_trial_ids;
};

/// Bridges statistics::KFold (index splits) to SQLite trial ids.
class TrialFoldSelector
{
   public:
    /// Create a selector by loading ordered trial ids from SQLite and splitting them.
    static auto from_sqlite(const std::string& dataset_root_path,
        std::size_t n_splits,
        bool shuffle,
        std::optional<unsigned int> seed) -> TrialFoldSelector;

    /// Number of folds available in this selector.
    [[nodiscard]] auto fold_count() const noexcept -> std::size_t;

    /// Resolve train/validation trial ids for one fold index.
    [[nodiscard]] auto selection_for_fold(std::size_t fold_idx) const -> TrialFoldSelection;

   private:
    TrialFoldSelector(std::vector<int> trial_ids, std::vector<statistics::FoldSplit> folds);

    std::vector<int> trial_ids_;
    std::vector<statistics::FoldSplit> folds_;
};
} // namespace experiment03
