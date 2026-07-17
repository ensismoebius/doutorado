/**
 * @file include/data_loaders/samplers/SequentialSampler.hpp
 * @brief Sequentialsampler.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_DATALOADERS_SEQUENTIAL_SAMPLER_HPP
#define NN_DATALOADERS_SEQUENTIAL_SAMPLER_HPP

#include <cstddef>
#include <span>

#include "data_loaders/samplers/ISampler.hpp"

class SequentialSampler final : public ISampler
{
   public:
    explicit SequentialSampler(std::size_t dataset_size);

    [[nodiscard]] auto index_count() const noexcept -> std::size_t override;
    void set_epoch(std::size_t epoch) override;
    void sample_into(std::span<std::size_t> out) override;

   private:
    std::size_t dataset_size_;
};

#endif // NN_DATALOADERS_SEQUENTIAL_SAMPLER_HPP
