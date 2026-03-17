#include "nn/dataLoaders/samplers/DistributedSampler.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

DistributedSampler::DistributedSampler(std::size_t dataset_size,
    std::size_t num_replicas,
    std::size_t rank,
    bool shuffle,
    bool drop_last,
    std::optional<unsigned int> seed)
    : dataset_size_(dataset_size),
      num_replicas_(num_replicas),
      rank_(rank),
      shuffle_(shuffle),
      drop_last_(drop_last),
      base_seed_(seed),
      rng_(seed.has_value() ? *seed : std::mt19937::result_type(std::random_device{}()))
{
    if (num_replicas_ == 0)
    {
        throw std::invalid_argument("DistributedSampler: num_replicas must be > 0.");
    }

    if (rank_ >= num_replicas_)
    {
        throw std::invalid_argument("DistributedSampler: rank must be < num_replicas.");
    }

    if (drop_last_)
    {
        num_samples_ = dataset_size_ / num_replicas_;
    }
    else
    {
        num_samples_ = (dataset_size_ + num_replicas_ - 1) / num_replicas_;
    }
    total_size_ = num_samples_ * num_replicas_;
}

auto DistributedSampler::index_count() const noexcept -> std::size_t
{
    return num_samples_;
}

void DistributedSampler::set_epoch(std::size_t epoch)
{
    if (base_seed_)
    {
        rng_.seed(*base_seed_ + static_cast<unsigned int>(epoch));
    }
}

void DistributedSampler::sample_into(std::span<std::size_t> out)
{
    if (out.size() != num_samples_)
    {
        throw std::invalid_argument("DistributedSampler: output span size mismatch.");
    }

    std::vector<std::size_t> global_indices(dataset_size_);
    std::iota(global_indices.begin(), global_indices.end(), static_cast<std::size_t>(0));

    if (shuffle_)
    {
        std::shuffle(global_indices.begin(), global_indices.end(), rng_);
    }

    if (!drop_last_)
    {
        if (global_indices.empty())
        {
            std::fill(out.begin(), out.end(), static_cast<std::size_t>(0));
            return;
        }

        while (global_indices.size() < total_size_)
        {
            const std::size_t needed = total_size_ - global_indices.size();
            const std::size_t chunk = std::min(needed, dataset_size_);
            global_indices.insert(global_indices.end(),
                global_indices.begin(),
                global_indices.begin() + static_cast<std::ptrdiff_t>(chunk));
        }
    }
    else
    {
        global_indices.resize(total_size_);
    }

    for (std::size_t i = 0; i < num_samples_; ++i)
    {
        out[i] = global_indices[rank_ + (i * num_replicas_)];
    }
}
