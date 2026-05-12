/**
 * @file include/nn/dataLoaders/samplers/FoldSampler.hpp
 * @brief Fold-aware sampler that emits train or validation indices for one fold.
 */

#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "data_loaders/samplers/ISampler.hpp"
#include "statistics/kfold.hpp"

/// Selects whether the sampler should expose train or validation indices.
enum class FoldPartition
{
    Train,
    Validation,
};

/// Wraps one `statistics::FoldSplit` as an `ISampler` implementation.
class FoldSampler final : public ISampler
{
   public:
    /// Build a sampler for either train or validation indices from one fold split.
    FoldSampler(const statistics::FoldSplit& split, FoldPartition partition);

    /// Returns count of selected indices for the chosen partition.
    [[nodiscard]] auto index_count() const noexcept -> std::size_t override;

    /// No-op: fold membership is epoch-invariant.
    void set_epoch(std::size_t epoch) override;

    /// Copies selected fold indices into `out`; sizes must match.
    void sample_into(std::span<std::size_t> out) override;

   private:
    std::vector<std::size_t> indices_;
};
