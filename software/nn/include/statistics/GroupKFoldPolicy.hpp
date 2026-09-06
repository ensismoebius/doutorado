/**
 * @file GroupKFoldPolicy.hpp
 * @brief GroupKFoldPolicy strategy (extracted from kfold.hpp).
 */

#ifndef NN_STATISTICS_GROUP_KFOLD_POLICY_HPP
#define NN_STATISTICS_GROUP_KFOLD_POLICY_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "statistics/kfold.hpp"

namespace statistics
{

/**
 * @brief Group-aware k-fold policy: all samples of the same group stay in
 *        the same fold.
 *
 * Use this for speaker authentication experiments to prevent the same speaker
 * from appearing in both train and test sets (data leakage).  Groups are
 * shuffled and assigned to folds round-robin (sklearn GroupKFold behaviour).
 *
 * Requires groups.size() == n_samples and n_unique_groups >= n_splits.
 */
class GroupKFoldPolicy : public ISplitPolicy
{
   public:
    explicit GroupKFoldPolicy(std::size_t n_splits, bool shuffle = false, std::uint32_t seed = 0U);

    [[nodiscard]] auto make_splits(std::size_t n_samples, const std::vector<int>& groups) const
        -> std::vector<FoldSplit> override;

   private:
    std::size_t n_splits_;
    bool shuffle_;
    std::uint32_t seed_;
};

} // namespace statistics

#endif // NN_STATISTICS_GROUP_KFOLD_POLICY_HPP
