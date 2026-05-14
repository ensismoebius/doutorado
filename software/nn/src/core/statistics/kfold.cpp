/**
 * @file kfold.cpp
 * @brief Implementation of KFold and StratifiedKFold splitters.
 */

#include "statistics/kfold.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
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

// ── SampleKFoldPolicy ────────────────────────────────────────────────────────

SampleKFoldPolicy::SampleKFoldPolicy(
    std::size_t n_splits, bool shuffle, std::uint32_t seed)
    : n_splits_(n_splits), shuffle_(shuffle), seed_(seed)
{
}

auto SampleKFoldPolicy::make_splits(
    std::size_t n_samples, const std::vector<int>& /*groups*/) const -> std::vector<FoldSplit>
{
    KFold kf(n_splits_, shuffle_, seed_);
    return kf.split(n_samples);
}

// ── GroupKFoldPolicy ─────────────────────────────────────────────────────────

GroupKFoldPolicy::GroupKFoldPolicy(
    std::size_t n_splits, bool shuffle, std::uint32_t seed)
    : n_splits_(n_splits), shuffle_(shuffle), seed_(seed)
{
}

auto GroupKFoldPolicy::make_splits(
    std::size_t n_samples, const std::vector<int>& groups) const -> std::vector<FoldSplit>
{
    if (groups.empty())
        throw std::invalid_argument("GroupKFoldPolicy: groups must not be empty");
    if (groups.size() != n_samples)
        throw std::invalid_argument("GroupKFoldPolicy: groups.size() != n_samples");

    // Collect unique group IDs in stable insertion order.
    std::vector<int> unique_groups;
    std::unordered_set<int> seen;
    for (int g : groups)
    {
        if (seen.insert(g).second)
            unique_groups.push_back(g);
    }

    if (unique_groups.size() < n_splits_)
        throw std::invalid_argument(
            "GroupKFoldPolicy: number of unique groups is less than n_splits");

    if (shuffle_)
    {
        std::mt19937 rng(seed_);
        std::shuffle(unique_groups.begin(), unique_groups.end(), rng);
    }

    // Assign groups to folds round-robin (sklearn GroupKFold behaviour).
    std::unordered_map<int, std::size_t> group_to_fold;
    group_to_fold.reserve(unique_groups.size());
    for (std::size_t i = 0; i < unique_groups.size(); ++i)
        group_to_fold[unique_groups[i]] = i % n_splits_;

    // Bucket sample indices by fold.
    std::vector<std::vector<std::size_t>> fold_buckets(n_splits_);
    for (std::size_t i = 0; i < n_samples; ++i)
        fold_buckets[group_to_fold.at(groups[i])].push_back(i);

    // Build FoldSplit for each fold: that fold's bucket = test; rest = train.
    std::vector<FoldSplit> splits;
    splits.reserve(n_splits_);
    for (std::size_t fi = 0; fi < n_splits_; ++fi)
    {
        FoldSplit fs;
        fs.test_indices = fold_buckets[fi];
        fs.train_indices.reserve(n_samples - fold_buckets[fi].size());
        for (std::size_t fj = 0; fj < n_splits_; ++fj)
        {
            if (fj != fi)
            {
                for (std::size_t idx : fold_buckets[fj])
                    fs.train_indices.push_back(idx);
            }
        }
        splits.push_back(std::move(fs));
    }
    return splits;
}

// ── NestedKFold ──────────────────────────────────────────────────────────────

NestedKFold::NestedKFold(std::size_t n_outer_splits,
    std::size_t n_inner_splits,
    bool shuffle,
    std::uint32_t random_seed)
    : n_outer_splits_(n_outer_splits),
      n_inner_splits_(n_inner_splits),
      shuffle_(shuffle),
      random_seed_(random_seed)
{
}

NestedKFold::NestedKFold(std::size_t n_outer_splits,
    std::size_t n_inner_splits,
    std::shared_ptr<ISplitPolicy> outer_policy,
    std::shared_ptr<ISplitPolicy> inner_policy)
    : n_outer_splits_(n_outer_splits),
      n_inner_splits_(n_inner_splits),
      outer_policy_(std::move(outer_policy)),
      inner_policy_(std::move(inner_policy))
{
    if (!outer_policy_ || !inner_policy_)
        throw std::invalid_argument("NestedKFold: policies must not be null");
}

auto NestedKFold::split(std::size_t n_samples) const -> std::vector<NestedFoldSplit>
{
    // Legacy sample-level path — preserves original seeded-inner-fold behaviour.
    validate_split_params(n_outer_splits_, n_samples);

    KFold outer_kf(n_outer_splits_, shuffle_, random_seed_);
    auto outer_folds = outer_kf.split(n_samples);

    std::vector<NestedFoldSplit> nested;
    nested.reserve(n_outer_splits_);

    for (auto& outer : outer_folds)
    {
        NestedFoldSplit nfs;
        nfs.test_indices = outer.test_indices;

        const std::size_t inner_n = outer.train_indices.size();
        validate_split_params(n_inner_splits_, inner_n);

        std::uint32_t inner_seed = random_seed_;
        if (!outer.test_indices.empty())
            inner_seed ^= static_cast<std::uint32_t>(outer.test_indices[0]) * 2654435761U;

        KFold inner_kf(n_inner_splits_, shuffle_, inner_seed);
        auto inner_splits_local = inner_kf.split(inner_n);

        for (auto& inner : inner_splits_local)
        {
            FoldSplit mapped;
            mapped.train_indices.reserve(inner.train_indices.size());
            mapped.test_indices.reserve(inner.test_indices.size());
            for (std::size_t li : inner.train_indices)
                mapped.train_indices.push_back(outer.train_indices[li]);
            for (std::size_t li : inner.test_indices)
                mapped.test_indices.push_back(outer.train_indices[li]);
            nfs.inner_splits.push_back(std::move(mapped));
        }
        nested.push_back(std::move(nfs));
    }
    return nested;
}

auto NestedKFold::split(
    std::size_t n_samples, const std::vector<int>& groups) const -> std::vector<NestedFoldSplit>
{
    if (!outer_policy_)
        throw std::logic_error(
            "NestedKFold::split(groups) requires the policy constructor");

    auto outer_folds = outer_policy_->make_splits(n_samples, groups);

    std::vector<NestedFoldSplit> nested;
    nested.reserve(n_outer_splits_);

    for (auto& outer : outer_folds)
    {
        NestedFoldSplit nfs;
        nfs.test_indices = outer.test_indices;

        const std::size_t inner_n = outer.train_indices.size();

        // Build group labels for just the outer training samples.
        std::vector<int> inner_groups;
        if (!groups.empty())
        {
            inner_groups.reserve(inner_n);
            for (std::size_t li = 0; li < inner_n; ++li)
                inner_groups.push_back(groups[outer.train_indices[li]]);
        }

        auto inner_folds_local = inner_policy_->make_splits(inner_n, inner_groups);

        // Map local inner indices → global sample indices.
        for (auto& inner : inner_folds_local)
        {
            FoldSplit mapped;
            mapped.train_indices.reserve(inner.train_indices.size());
            mapped.test_indices.reserve(inner.test_indices.size());
            for (std::size_t li : inner.train_indices)
                mapped.train_indices.push_back(outer.train_indices[li]);
            for (std::size_t li : inner.test_indices)
                mapped.test_indices.push_back(outer.train_indices[li]);
            nfs.inner_splits.push_back(std::move(mapped));
        }
        nested.push_back(std::move(nfs));
    }
    return nested;
}

} // namespace statistics
