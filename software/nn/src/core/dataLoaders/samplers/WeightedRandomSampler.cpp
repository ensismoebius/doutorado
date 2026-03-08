#include "nn/dataLoaders/samplers/WeightedRandomSampler.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>

WeightedRandomSampler::WeightedRandomSampler(std::vector<double> weights, std::size_t num_samples,
                                             std::optional<unsigned int> seed)
    : weights_(std::move(weights)),
      num_samples_(num_samples),
      base_seed_(seed),
      rng_(seed.has_value() ? *seed : std::mt19937::result_type(std::random_device{}()))
{
    if (weights_.empty())
    {
        throw std::invalid_argument("WeightedRandomSampler: weights cannot be empty.");
    }

    if (num_samples_ == 0)
    {
        throw std::invalid_argument("WeightedRandomSampler: num_samples must be > 0.");
    }

    const bool all_non_negative =
        std::all_of(weights_.begin(), weights_.end(), [](double w) { return w >= 0.0; });
    const bool has_positive =
        std::any_of(weights_.begin(), weights_.end(), [](double w) { return w > 0.0; });

    if (!all_non_negative || !has_positive)
    {
        throw std::invalid_argument(
            "WeightedRandomSampler: weights must be non-negative and contain at least one "
            "positive value.");
    }
}

auto WeightedRandomSampler::index_count() const noexcept -> std::size_t
{
    return num_samples_;
}

void WeightedRandomSampler::set_epoch(std::size_t epoch)
{
    if (base_seed_)
    {
        rng_.seed(*base_seed_ + static_cast<unsigned int>(epoch));
    }
}

void WeightedRandomSampler::sample_into(std::span<std::size_t> out)
{
    if (out.size() != num_samples_)
    {
        throw std::invalid_argument("WeightedRandomSampler: output span size mismatch.");
    }

    std::discrete_distribution<std::size_t> distribution(weights_.begin(), weights_.end());
    for (std::size_t i = 0; i < num_samples_; ++i)
    {
        out[i] = distribution(rng_);
    }
}
