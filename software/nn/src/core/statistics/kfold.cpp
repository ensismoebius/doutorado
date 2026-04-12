/**
 * @file kfold.cpp
 * @brief Implementation of KFold and StratifiedKFold splitters.
 */

#include "nn/statistics/kfold.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace
{

void validate_split_params(std::size_t n_splits, std::size_t n_samples)
{
    if (n_splits < 2U)
    {
        throw std::invalid_argument("n_splits must be >= 2");
    }
    if (n_samples < n_splits)
    {
        throw std::invalid_argument("n_samples must be >= n_splits");
    }
}

} // namespace

namespace statistics
{

KFold::KFold(std::size_t n_splits, bool shuffle, std::uint32_t random_seed)
    : n_splits_(n_splits), shuffle_(shuffle), random_seed_(random_seed)
{
}

auto KFold::split(std::size_t n_samples) const -> std::vector<FoldSplit>
{
    validate_split_params(n_splits_, n_samples);

    std::vector<std::size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0U);

    if (shuffle_)
    {
        std::mt19937 rng(random_seed_);
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    std::vector<std::size_t> fold_sizes(n_splits_, n_samples / n_splits_);
    const std::size_t remainder = n_samples % n_splits_;
    for (std::size_t i = 0; i < remainder; ++i)
    {
        ++fold_sizes[i];
    }

    std::vector<FoldSplit> splits;
    splits.reserve(n_splits_);

    std::size_t current = 0U;
    for (std::size_t fold_idx = 0; fold_idx < n_splits_; ++fold_idx)
    {
        const std::size_t fold_size = fold_sizes[fold_idx];
        const std::size_t fold_end = current + fold_size;

        FoldSplit fold;
        fold.test_indices.reserve(fold_size);
        fold.train_indices.reserve(n_samples - fold_size);

        for (std::size_t i = current; i < fold_end; ++i)
        {
            fold.test_indices.push_back(indices[i]);
        }
        for (std::size_t i = 0; i < current; ++i)
        {
            fold.train_indices.push_back(indices[i]);
        }
        for (std::size_t i = fold_end; i < n_samples; ++i)
        {
            fold.train_indices.push_back(indices[i]);
        }

        splits.push_back(std::move(fold));
        current = fold_end;
    }

    return splits;
}

StratifiedKFold::StratifiedKFold(std::size_t n_splits, bool shuffle, std::uint32_t random_seed)
    : n_splits_(n_splits), shuffle_(shuffle), random_seed_(random_seed)
{
}

auto StratifiedKFold::split(const std::vector<int>& labels) const -> std::vector<FoldSplit>
{
    const std::size_t n_samples = labels.size();
    validate_split_params(n_splits_, n_samples);

    std::unordered_map<int, std::vector<std::size_t>> class_to_indices;
    class_to_indices.reserve(n_samples);

    for (std::size_t idx = 0; idx < n_samples; ++idx)
    {
        class_to_indices[labels[idx]].push_back(idx);
    }

    std::vector<std::vector<std::size_t>> test_folds(n_splits_);
    std::mt19937 rng(random_seed_);

    for (auto& kv : class_to_indices)
    {
        auto& class_indices = kv.second;
        if (shuffle_)
        {
            std::shuffle(class_indices.begin(), class_indices.end(), rng);
        }

        for (std::size_t i = 0; i < class_indices.size(); ++i)
        {
            test_folds[i % n_splits_].push_back(class_indices[i]);
        }
    }

    std::vector<FoldSplit> splits;
    splits.reserve(n_splits_);

    for (std::size_t fold_idx = 0; fold_idx < n_splits_; ++fold_idx)
    {
        FoldSplit fold;
        fold.test_indices = test_folds[fold_idx];
        fold.train_indices.reserve(n_samples - fold.test_indices.size());

        std::vector<char> is_test(n_samples, 0);
        for (const std::size_t idx : fold.test_indices)
        {
            is_test[idx] = 1;
        }

        for (std::size_t idx = 0; idx < n_samples; ++idx)
        {
            if (!is_test[idx])
            {
                fold.train_indices.push_back(idx);
            }
        }

        splits.push_back(std::move(fold));
    }

    return splits;
}

} // namespace statistics
