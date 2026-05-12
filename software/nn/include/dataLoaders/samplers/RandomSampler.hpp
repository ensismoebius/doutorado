/**
 * @file include/nn/dataLoaders/samplers/RandomSampler.hpp
 * @brief Randomsampler.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_DATALOADERS_RANDOM_SAMPLER_HPP
#define NN_DATALOADERS_RANDOM_SAMPLER_HPP

#include <cstddef>
#include <optional>
#include <random>
#include <span>

#include "dataLoaders/samplers/ISampler.hpp"

class RandomSampler final : public ISampler
{
   public:
    explicit RandomSampler(
        std::size_t dataset_size, std::optional<unsigned int> seed = std::nullopt);

    [[nodiscard]] auto index_count() const noexcept -> std::size_t override;
    void set_epoch(std::size_t epoch) override;
    void sample_into(std::span<std::size_t> out) override;

   private:
    std::size_t dataset_size_;
    std::optional<unsigned int> base_seed_;
    std::mt19937 rng_;
};

#endif // NN_DATALOADERS_RANDOM_SAMPLER_HPP
