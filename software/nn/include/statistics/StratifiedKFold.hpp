/**
 * @file StratifiedKFold.hpp
 * @brief StratifiedKFold splitter (extracted from kfold.hpp).
 */

#ifndef NN_STATISTICS_STRATIFIED_KFOLD_HPP
#define NN_STATISTICS_STRATIFIED_KFOLD_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "statistics/kfold.hpp"

namespace statistics
{

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

#endif // NN_STATISTICS_STRATIFIED_KFOLD_HPP
