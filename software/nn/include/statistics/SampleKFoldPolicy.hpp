/**
 * @file SampleKFoldPolicy.hpp
 * @brief SampleKFoldPolicy strategy (extracted from kfold.hpp).
 */

#ifndef NN_STATISTICS_SAMPLE_KFOLD_POLICY_HPP
#define NN_STATISTICS_SAMPLE_KFOLD_POLICY_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "statistics/kfold.hpp"

namespace statistics
{

/**
 * @brief Sample-level k-fold policy (wraps KFold, ignores groups).
 *
 * Default policy used by the legacy NestedKFold constructor.  Equivalent to
 * splitting by raw sample index — no group integrity guarantee.
 */
class SampleKFoldPolicy : public ISplitPolicy
{
   public:
    explicit SampleKFoldPolicy(std::size_t n_splits, bool shuffle = false, std::uint32_t seed = 0U);

    [[nodiscard]] auto make_splits(std::size_t n_samples, const std::vector<int>& /*groups*/) const
        -> std::vector<FoldSplit> override;

   private:
    std::size_t n_splits_;
    bool shuffle_;
    std::uint32_t seed_;
};

} // namespace statistics

#endif // NN_STATISTICS_SAMPLE_KFOLD_POLICY_HPP
