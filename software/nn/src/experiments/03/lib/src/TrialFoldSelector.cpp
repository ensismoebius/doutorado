/**
 * @file src/experiments/03/lib/src/TrialFoldSelector.cpp
 * @brief Implementation of TrialFoldSelector.
 */

#include "TrialFoldSelector.hpp"

#include <sqlite3.h>

#include <filesystem>
#include <stdexcept>

#include "nn/dataLoaders/samplers/FoldSampler.hpp"

namespace experiment03
{
namespace
{
auto load_trial_ids_from_sqlite(const std::string& dataset_root_path) -> std::vector<int>
{
    const auto db_path = std::filesystem::path(dataset_root_path) / "database.sqlite";

    sqlite3* db = nullptr;
    if (sqlite3_open(db_path.string().c_str(), &db) != SQLITE_OK)
    {
        if (db)
        {
            sqlite3_close(db);
        }
        throw std::runtime_error("Failed to open SQLite DB: " + db_path.string());
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "SELECT DISTINCT t.id FROM trial t "
        "INNER JOIN audio_samples a ON a.trial_id = t.id "
        "INNER JOIN eeg_samples e ON e.trial_id = t.id "
        "ORDER BY t.id;";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
        const std::string err = sqlite3_errmsg(db);
        sqlite3_close(db);
        throw std::runtime_error("Failed to prepare trial-id query: " + err);
    }

    std::vector<int> trial_ids;
    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        trial_ids.push_back(sqlite3_column_int(stmt, 0));
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);

    if (trial_ids.empty())
    {
        throw std::runtime_error("No joined trial ids found in SQLite DB: " + db_path.string());
    }

    return trial_ids;
}

auto sample_indices(const statistics::FoldSplit& split, FoldPartition partition)
    -> std::vector<std::size_t>
{
    FoldSampler sampler(split, partition);
    std::vector<std::size_t> indices(sampler.index_count());
    sampler.sample_into(indices);
    return indices;
}

auto map_indices_to_trial_ids(
    const std::vector<std::size_t>& indices, const std::vector<int>& trial_ids) -> std::vector<int>
{
    std::vector<int> selected;
    selected.reserve(indices.size());
    for (const std::size_t idx : indices)
    {
        if (idx >= trial_ids.size())
        {
            throw std::runtime_error("Fold index out of range for trial-id mapping");
        }
        selected.push_back(trial_ids[idx]);
    }
    return selected;
}
} // namespace

TrialFoldSelector::TrialFoldSelector(
    std::vector<int> trial_ids, std::vector<statistics::FoldSplit> folds)
    : trial_ids_(std::move(trial_ids)), folds_(std::move(folds))
{
}

auto TrialFoldSelector::from_sqlite(const std::string& dataset_root_path,
    std::size_t n_splits,
    bool shuffle,
    std::optional<unsigned int> seed) -> TrialFoldSelector
{
    std::vector<int> trial_ids = load_trial_ids_from_sqlite(dataset_root_path);
    std::vector<statistics::FoldSplit> folds =
        statistics::KFold(n_splits, shuffle, seed.value_or(0U)).split(trial_ids.size());
    return TrialFoldSelector(std::move(trial_ids), std::move(folds));
}

auto TrialFoldSelector::fold_count() const noexcept -> std::size_t
{
    return folds_.size();
}

auto TrialFoldSelector::selection_for_fold(std::size_t fold_idx) const -> TrialFoldSelection
{
    if (fold_idx >= folds_.size())
    {
        throw std::runtime_error(
            "Fold index out of range in TrialFoldSelector::selection_for_fold");
    }

    const auto& split = folds_[fold_idx];
    const std::vector<std::size_t> train_indices = sample_indices(split, FoldPartition::Train);
    const std::vector<std::size_t> val_indices = sample_indices(split, FoldPartition::Validation);

    return TrialFoldSelection{
        .train_trial_ids = map_indices_to_trial_ids(train_indices, trial_ids_),
        .val_trial_ids = map_indices_to_trial_ids(val_indices, trial_ids_),
    };
}
} // namespace experiment03
