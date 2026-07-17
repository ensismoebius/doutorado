/**
 * @file src/core/data_loaders/samplers/SequentialSampler.cpp
 * @brief Implementation for Sequentialsampler.
 *

 */

#include "data_loaders/samplers/SequentialSampler.hpp"

#include <numeric>
#include <stdexcept>

SequentialSampler::SequentialSampler(std::size_t dataset_size) : dataset_size_(dataset_size) {}

auto SequentialSampler::index_count() const noexcept -> std::size_t
{
    return dataset_size_;
}

void SequentialSampler::set_epoch(std::size_t epoch)
{
    (void) epoch;
}

void SequentialSampler::sample_into(std::span<std::size_t> out)
{
    if (out.size() != dataset_size_)
    {
        throw std::invalid_argument("SequentialSampler: output span size mismatch.");
    }
    std::iota(out.begin(), out.end(), static_cast<std::size_t>(0));
}
