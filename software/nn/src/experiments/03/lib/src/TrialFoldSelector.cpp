/**
 * @file src/experiments/03/lib/src/TrialFoldSelector.cpp
 * @brief Implementation of TrialFoldSelector.
 */

#include "TrialFoldSelector.hpp"

#include <algorithm>
#include <random>
#include <sqlite3.h>

#include <filesystem>
#include <stdexcept>

#include "dataLoaders/samplers/FoldSampler.hpp"

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

auto split_train_val_test(std::vector<int> trial_ids,
    float test_split_ratio,
    bool shuffle,
    std::optional<unsigned int> seed) -> std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>
{
    if (test_split_ratio <= 0.0f || test_split_ratio >= 1.0f)
    {
        return {trial_ids, {}, {}};
    }

    std::vector<std::size_t> indices(trial_ids.size());
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle)
    {
        std::mt19937 rng(seed.value_or(std::random_device{}()));
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    const auto test_size = static_cast<size_t>(trial_ids.size() * test_split_ratio);
    const auto train_val_end = trial_ids.size() - test_size;

    std::vector<int> train_val_trial_ids;
    std::vector<int> test_trial_ids;

    train_val_trial_ids.reserve(train_val_end);
    test_trial_ids.reserve(test_size);

    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (i < train_val_end)
        {
            train_val_trial_ids.push_back(trial_ids[indices[i]]);
        }
        else
        {
            test_trial_ids.push_back(trial_ids[indices[i]]);
        }
    }

    return {train_val_trial_ids, test_trial_ids, {}};
}

} // namespace

TrialFoldSelector::TrialFoldSelector(
    std::vector<int> trial_ids, std::vector<statistics::FoldSplit> folds, std::vector<int> test_trial_ids)
    : trial_ids_(std::move(trial_ids)), folds_(std::move(folds)), test_trial_ids_(std::move(test_trial_ids))
{
}

auto TrialFoldSelector::from_sqlite(const std::string& dataset_root_path,
    std::size_t n_splits,
    bool shuffle,
    std::optional<unsigned int> seed,
    float test_split) -> TrialFoldSelector
{
    std::vector<int> all_trial_ids = load_trial_ids_from_sqlite(dataset_root_path);
    std::vector<int> test_trial_ids;
    std::vector<int> train_val_trial_ids;

    if (test_split > 0.0f && test_split < 1.0f)
    {
        std::tie(train_val_trial_ids, test_trial_ids, std::ignore) =
            split_train_val_test(all_trial_ids, test_split, shuffle, seed);
    }
    else
    {
        train_val_trial_ids = std::move(all_trial_ids);
    }

    std::vector<statistics::FoldSplit> folds;
    if (n_splits > 1 && !train_val_trial_ids.empty())
    {
        folds = statistics::KFold(n_splits, shuffle, seed.value_or(0U)).split(train_val_trial_ids.size());
    }

    return TrialFoldSelector(std::move(train_val_trial_ids), std::move(folds), std::move(test_trial_ids));
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
        .test_trial_ids = test_trial_ids_,
    };
}

auto TrialFoldSelector::test_trial_ids() const noexcept -> const std::vector<int>&
{
    return test_trial_ids_;
}
} // namespace experiment03
