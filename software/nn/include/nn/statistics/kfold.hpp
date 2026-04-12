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

} // namespace statistics

#endif // NN_STATISTICS_KFOLD_HPP
