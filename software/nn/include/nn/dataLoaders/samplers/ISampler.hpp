#ifndef NN_DATALOADERS_ISAMPLER_HPP
#define NN_DATALOADERS_ISAMPLER_HPP

#include <cstddef>
#include <span>

/**
 * @file ISampler.hpp
 * @brief Sampling interface for DataLoader index generation.
 *
 * Samplers produce dataset indices only (not data).
 */
class ISampler
{
   public:
    virtual ~ISampler() = default;

    // Number of indices yielded per epoch.
    [[nodiscard]] virtual auto index_count() const noexcept -> std::size_t = 0;

    // Optional epoch hook (e.g., reseed deterministic shuffles).
    virtual void set_epoch(std::size_t epoch) = 0;

    // Fill `out` with sampled indices for the current epoch.
    virtual void sample_into(std::span<std::size_t> out) = 0;
};

#endif // NN_DATALOADERS_ISAMPLER_HPP
