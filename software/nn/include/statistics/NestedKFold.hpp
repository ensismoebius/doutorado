/**
 * @file NestedKFold.hpp
 * @brief NestedFoldSplit + NestedKFold (extracted from kfold.hpp).
 */

#ifndef NN_STATISTICS_NESTED_KFOLD_HPP
#define NN_STATISTICS_NESTED_KFOLD_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "statistics/kfold.hpp"

namespace statistics
{

/**
 * @brief One outer fold for nested cross-validation.
 *
 * Contains a held-out test set plus all inner train/val splits used for
 * hyperparameter selection within this outer fold.
 */
struct NestedFoldSplit
{
    std::vector<std::size_t> test_indices; ///< Outer test set — never seen during HPO
    std::vector<FoldSplit> inner_splits;   ///< Inner k-fold splits for HPO (train/val)
};

/**
 * @brief Nested K-fold splitter for unbiased hyperparameter evaluation.
 *
 * Implements the two-loop cross-validation protocol recommended for biomedical
 * ML (PMC guide 2023):
 * - **Outer loop** (n_outer_splits folds): provides an unbiased estimate of
 *   generalisation performance on completely held-out test data.
 * - **Inner loop** (n_inner_splits folds per outer fold): used to select
 *   hyperparameters on the remaining data without touching the outer test set.
 *
 * Usage:
 * @code
 * statistics::NestedKFold nkf(5, 5, true, 42);
 * auto folds = nkf.split(n_samples);
 * for (auto& outer : folds) {
 *     // outer.test_indices — held-out test
 *     for (auto& inner : outer.inner_splits) {
 *         // inner.train_indices — HPO training
 *         // inner.test_indices  — HPO validation
 *     }
 * }
 * @endcode
 *
 * Reference: [41] A. Leal et al., "A guide to cross-validation for AI in medical
 * imaging," Radiology: AI, 2023.
 */
class NestedKFold
{
   public:
    /**
     * @brief Legacy constructor — sample-level splits, no group integrity.
     *
     * Preserved for backward compatibility.  Uses SampleKFoldPolicy internally.
     * Call split(n_samples) with this constructor.
     *
     * @param n_outer_splits Folds for the outer (test) loop. Must be >= 2.
     * @param n_inner_splits Folds for each inner (HPO) loop. Must be >= 2.
     * @param shuffle        Shuffle indices before splitting (both loops).
     * @param random_seed    RNG seed used when shuffle is true.
     */
    explicit NestedKFold(std::size_t n_outer_splits,
        std::size_t n_inner_splits,
        bool shuffle = false,
        std::uint32_t random_seed = 0U);

    /**
     * @brief Policy constructor — pluggable split strategy for outer and inner loops.
     *
     * Use this when group integrity is required (e.g. speaker-grouped splits).
     * Call split(n_samples, groups) with this constructor.
     *
     * @param n_outer_splits Folds for the outer (test) loop. Must be >= 2.
     * @param n_inner_splits Folds for each inner (HPO) loop. Must be >= 2.
     * @param outer_policy   Strategy used for the outer (test) fold assignment.
     * @param inner_policy   Strategy used for the inner (val) fold assignment.
     */
    NestedKFold(std::size_t n_outer_splits,
        std::size_t n_inner_splits,
        std::shared_ptr<ISplitPolicy> outer_policy,
        std::shared_ptr<ISplitPolicy> inner_policy);

    /**
     * @brief Generate all nested splits for a dataset of size n_samples.
     *
     * Uses the legacy sample-level path (requires legacy constructor).
     * @return Vector with n_outer_splits nested fold splits.
     */
    [[nodiscard]] auto split(std::size_t n_samples) const -> std::vector<NestedFoldSplit>;

    /**
     * @brief Generate all nested splits with group integrity.
     *
     * Each group's samples are kept together in the same fold.
     * Requires the policy constructor; throws std::logic_error otherwise.
     *
     * @param n_samples Total samples. Must equal groups.size().
     * @param groups    One integer group-ID per sample (e.g. speaker ID).
     * @return Vector with n_outer_splits nested fold splits.
     */
    [[nodiscard]] auto split(std::size_t n_samples, const std::vector<int>& groups) const
        -> std::vector<NestedFoldSplit>;

   private:
    std::size_t n_outer_splits_;
    std::size_t n_inner_splits_;
    // Legacy path
    bool shuffle_ = false;
    std::uint32_t random_seed_ = 0U;
    // Policy path (null = use legacy path)
    std::shared_ptr<ISplitPolicy> outer_policy_;
    std::shared_ptr<ISplitPolicy> inner_policy_;
};

} // namespace statistics

#endif // NN_STATISTICS_NESTED_KFOLD_HPP
