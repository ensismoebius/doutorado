/**
 * @file src/core/data_loaders/samplers/RandomSampler.cpp
 * @brief Implementation for Randomsampler.
 *

 */

#include "data_loaders/samplers/RandomSampler.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

RandomSampler::RandomSampler(std::size_t dataset_size, std::optional<unsigned int> seed)
    : dataset_size_(dataset_size),
      base_seed_(seed),
      rng_(seed.has_value() ? *seed : std::mt19937::result_type(std::random_device{}()))
{
}

auto RandomSampler::index_count() const noexcept -> std::size_t
{
    return dataset_size_;
}

void RandomSampler::set_epoch(std::size_t epoch)
{
    if (base_seed_)
    {
        rng_.seed(*base_seed_ + static_cast<unsigned int>(epoch));
    }
}

void RandomSampler::sample_into(std::span<std::size_t> out)
{
    if (out.size() != dataset_size_)
    {
        throw std::invalid_argument("RandomSampler: output span size mismatch.");
    }
    std::iota(out.begin(), out.end(), static_cast<std::size_t>(0));
    std::shuffle(out.begin(), out.end(), rng_);
}
