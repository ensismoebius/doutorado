/**
 * @file kfold.hpp
 * @brief Deterministic K-fold and stratified K-fold index splitters.
 *
 * Core interface (`ISplitPolicy`), entry class (`KFold`) and its one result
 * struct (`FoldSplit`) live here. The independent strategy implementations
 * (`SampleKFoldPolicy`, `GroupKFoldPolicy`, `StratifiedKFold`) and the nested
 * cross-validation pair (`NestedFoldSplit` + `NestedKFold`) each live in their
 * own file, included below so existing includers of kfold.hpp keep working.
 */

#ifndef NN_STATISTICS_KFOLD_HPP
#define NN_STATISTICS_KFOLD_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace statistics
{

/**
 * @brief One fold split with train and test index sets.
 */
struct FoldSplit
{
    std::vector<std::size_t> train_indices;
    std::vector<std::size_t> test_indices;
};

// ── Split policy interface ────────────────────────────────────────────────────

/**
 * @brief Pluggable strategy for dividing a dataset into k folds.
 *
 * Implement this interface to provide custom splitting logic (grouped by
 * speaker, stratified by label, time-ordered, etc.) and pass it to the
 * policy-based NestedKFold constructor.
 *
 * @param n_samples Total number of samples.
 * @param groups    One integer group-ID per sample (size == n_samples) or
 *                  empty when the policy does not need group information.
 * @return Vector of exactly n_splits FoldSplit objects whose indices cover
 *         [0, n_samples) without overlap.
 */
struct ISplitPolicy
{
    virtual ~ISplitPolicy() = default;
    [[nodiscard]] virtual auto make_splits(
        std::size_t n_samples, const std::vector<int>& groups) const -> std::vector<FoldSplit> = 0;
};

/**
 * @brief K-fold splitter similar to sklearn/pytorch data split behavior.
 */
class KFold
{
   public:
    /**
     * @brief Build a K-fold splitter.
     * @param n_splits Number of folds. Must be >= 2.
     * @param shuffle Whether to shuffle indices before splitting.
     * @param random_seed Seed used when shuffle is true.
     */
    explicit KFold(std::size_t n_splits, bool shuffle = false, std::uint32_t random_seed = 0U);

    /**
     * @brief Create all fold splits for a dataset size.
     * @param n_samples Total number of samples.
     * @return Vector with n_splits fold splits.
     */
    [[nodiscard]] auto split(std::size_t n_samples) const -> std::vector<FoldSplit>;

   private:
    std::size_t n_splits_;
    bool shuffle_;
    std::uint32_t random_seed_;
};

} // namespace statistics

#include "statistics/GroupKFoldPolicy.hpp"
#include "statistics/NestedKFold.hpp"
#include "statistics/SampleKFoldPolicy.hpp"
#include "statistics/StratifiedKFold.hpp"

#endif // NN_STATISTICS_KFOLD_HPP
