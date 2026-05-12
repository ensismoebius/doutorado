/**
 * @file include/nn/dataLoaders/samplers/SubsetSampler.hpp
 * @brief Sampler that serves a fixed, pre-determined subset of dataset indices.
 *
 * Intended for k-fold cross-validation: each fold provides its own train or
 * test index vector; `SubsetSampler` wraps that vector so it can be passed
 * directly to `DataLoader`'s injected-sampler constructor.
 *
 * Contract:
 * - `index_count()` returns the size of the provided index vector.
 * - `set_epoch()` is a no-op; the subset indices are constant across epochs.
 * - `sample_into(out)` copies the fixed indices into the caller's span.
 * - `out` must have capacity equal to `index_count()`; behaviour is undefined
 *   if sizes differ.
 */

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "data_loaders/samplers/ISampler.hpp"

/// Serves a fixed list of dataset indices, regardless of epoch.
/// Designed for use with k-fold splits so that a DataLoader trains or
/// evaluates on exactly the requested subset.
class SubsetSampler final : public ISampler
{
   public:
    /// Construct from a pre-computed list of dataset indices.
    /// The indices are copied into the sampler and remain constant.
    explicit SubsetSampler(std::vector<std::size_t> indices);

    /// Returns the number of indices in the subset.
    [[nodiscard]] auto index_count() const noexcept -> std::size_t override;

    /// No-op: subset indices do not change between epochs.
    void set_epoch(std::size_t epoch) override;

    /// Copies the fixed indices into `out`. `out.size()` must equal `index_count()`.
    void sample_into(std::span<std::size_t> out) override;

   private:
    /// The fixed subset of dataset indices served by this sampler.
    std::vector<std::size_t> indices_;
};
