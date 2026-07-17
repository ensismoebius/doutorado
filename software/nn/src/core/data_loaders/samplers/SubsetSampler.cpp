/**
 * @file src/core/data_loaders/samplers/SubsetSampler.cpp
 * @brief Implementation of SubsetSampler.
 */

#include "data_loaders/samplers/SubsetSampler.hpp"

#include <algorithm>
#include <stdexcept>

SubsetSampler::SubsetSampler(std::vector<std::size_t> indices) : indices_(std::move(indices)) {}

auto SubsetSampler::index_count() const noexcept -> std::size_t
{
    return indices_.size();
}

void SubsetSampler::set_epoch(std::size_t /*epoch*/)
{
    // Subset indices are epoch-invariant; no action needed.
}

void SubsetSampler::sample_into(std::span<std::size_t> out)
{
    if (out.size() != indices_.size()) [[unlikely]]
    {
        throw std::invalid_argument("SubsetSampler::sample_into: out.size() != index_count()");
    }
    std::copy(indices_.begin(), indices_.end(), out.begin());
}
