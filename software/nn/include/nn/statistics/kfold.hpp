/**
 * @file kfold.hpp
 * @brief Deterministic K-fold and stratified K-fold index splitters.
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

/**
 * @brief Stratified K-fold splitter for classification labels.
 */
class StratifiedKFold
{
   public:
    /**
     * @brief Build a stratified K-fold splitter.
     * @param n_splits Number of folds. Must be >= 2.
     * @param shuffle Whether to shuffle labels inside each class bucket.
     * @param random_seed Seed used when shuffle is true.
     */
    explicit StratifiedKFold(
        std::size_t n_splits, bool shuffle = false, std::uint32_t random_seed = 0U);

    /**
     * @brief Create stratified fold splits from class labels.
     * @param labels Integer class label for each sample.
     * @return Vector with n_splits fold splits.
     */
    [[nodiscard]] auto split(const std::vector<int>& labels) const -> std::vector<FoldSplit>;

   private:
    std::size_t n_splits_;
    bool shuffle_;
    std::uint32_t random_seed_;
};

/**
 * @brief One outer fold for nested cross-validation.
 *
 * Contains a held-out test set plus all inner train/val splits used for
 * hyperparameter selection within this outer fold.
 */
struct NestedFoldSplit
{
    std::vector<std::size_t> test_indices;  ///< Outer test set — never seen during HPO
    std::vector<FoldSplit> inner_splits;     ///< Inner k-fold splits for HPO (train/val)
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
     * @brief Generate all nested splits for a dataset of size n_samples.
     * @return Vector with n_outer_splits nested fold splits.
     */
    [[nodiscard]] auto split(std::size_t n_samples) const -> std::vector<NestedFoldSplit>;

   private:
    std::size_t n_outer_splits_;
    std::size_t n_inner_splits_;
    bool shuffle_;
    std::uint32_t random_seed_;
};

} // namespace statistics

#endif // NN_STATISTICS_KFOLD_HPP
