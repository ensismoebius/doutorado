/**
 * @file include/nn/dataLoaders/samplers/DistributedSampler.hpp
 * @brief Distributedsampler.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_DATALOADERS_DISTRIBUTED_SAMPLER_HPP
#define NN_DATALOADERS_DISTRIBUTED_SAMPLER_HPP

#include <cstddef>
#include <optional>
#include <random>
#include <span>

#include "nn/dataLoaders/samplers/ISampler.hpp"

class DistributedSampler final : public ISampler
{
   public:
    DistributedSampler(           //
        std::size_t dataset_size, //
        std::size_t num_replicas, //
        std::size_t rank,         //
        bool shuffle = true,
        bool drop_last = false,                         //
        std::optional<unsigned int> seed = std::nullopt //
    );

    [[nodiscard]] auto index_count() const noexcept -> std::size_t override;

    void set_epoch(std::size_t epoch) override;

    void sample_into(std::span<std::size_t> out) override;

   private:
    std::size_t dataset_size_;
    std::size_t num_replicas_;
    std::size_t rank_;
    bool shuffle_;
    bool drop_last_;
    std::optional<unsigned int> base_seed_;
    std::size_t num_samples_;
    std::size_t total_size_;
    std::mt19937 rng_;
};

#endif // NN_DATALOADERS_DISTRIBUTED_SAMPLER_HPP
