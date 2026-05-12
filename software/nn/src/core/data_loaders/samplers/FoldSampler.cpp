/**
 * @file src/core/dataLoaders/samplers/FoldSampler.cpp
 * @brief Implementation of FoldSampler.
 */

#include "data_loaders/samplers/FoldSampler.hpp"

#include <algorithm>
#include <stdexcept>

FoldSampler::FoldSampler(const statistics::FoldSplit& split, FoldPartition partition)
{
    if (partition == FoldPartition::Train)
    {
        indices_ = split.train_indices;
    }
    else
    {
        indices_ = split.test_indices;
    }
}

auto FoldSampler::index_count() const noexcept -> std::size_t
{
    return indices_.size();
}

void FoldSampler::set_epoch(std::size_t /*epoch*/)
{
    // Fold indices are deterministic and fixed for this sampler instance.
}

void FoldSampler::sample_into(std::span<std::size_t> out)
{
    if (out.size() != indices_.size()) [[unlikely]]
    {
        throw std::invalid_argument("FoldSampler::sample_into: out.size() != index_count()");
    }

    std::copy(indices_.begin(), indices_.end(), out.begin());
}
