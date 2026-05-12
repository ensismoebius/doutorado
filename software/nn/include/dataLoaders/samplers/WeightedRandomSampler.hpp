/**
 * @file include/nn/dataLoaders/samplers/WeightedRandomSampler.hpp
 * @brief Weightedrandomsampler.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_DATALOADERS_WEIGHTED_RANDOM_SAMPLER_HPP
#define NN_DATALOADERS_WEIGHTED_RANDOM_SAMPLER_HPP

#include <cstddef>
#include <optional>
#include <random>
#include <span>
#include <vector>

#include "dataLoaders/samplers/ISampler.hpp"

class WeightedRandomSampler final : public ISampler
{
   public:
    WeightedRandomSampler(std::vector<double> weights,
        std::size_t num_samples,
        std::optional<unsigned int> seed = std::nullopt);

    [[nodiscard]] auto index_count() const noexcept -> std::size_t override;
    void set_epoch(std::size_t epoch) override;
    void sample_into(std::span<std::size_t> out) override;

   private:
    std::vector<double> weights_;
    std::size_t num_samples_;
    std::optional<unsigned int> base_seed_;
    std::mt19937 rng_;
};

#endif // NN_DATALOADERS_WEIGHTED_RANDOM_SAMPLER_HPP
