#ifndef SRC_CORE_STATISTICS_MULTICLASSMETRICS_H_
#define SRC_CORE_STATISTICS_MULTICLASSMETRICS_H_

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "statistics/kfold.hpp"

namespace statistics
{

/**
 * @file multi_class_metrics.hpp
 * @brief Multi-class classification metrics and a simple k-fold helper.
 *
 * `compute_classification_metrics()` returns common aggregate metrics from
 * integer class labels.
 *
 * Averaging conventions:
 * - `precision`, `recall`, and `f1_score` are macro-averaged (each class gets
 *   equal weight, independent of class frequency).
 * - `balanced_accuracy` is the mean per-class recall.
 * - `mcc` is only meaningful for binary classification in this implementation.
 *
 * `k_fold_cross_validation()` is a convenience helper for small experiments:
 * - It shuffles indices with a fixed RNG seed.
 * - It uses contiguous fold slices; if `n_samples % k != 0`, the remainder is
 *   currently ignored (the last few samples never appear in a test fold).
 */

struct ClassificationMetrics
{
    double accuracy = 0.0;
    double precision = 0.0; // macro-averaged
    double recall = 0.0;    // macro-averaged
    double f1_score = 0.0;  // macro-averaged
    double balanced_accuracy = 0.0;
    double mcc = 0.0; // Matthews correlation coefficient (for binary)
};

/**
 * Compute multi-class classification metrics
 * @param true_labels vector of true class labels
 * @param pred_labels vector of predicted class labels
 * @return ClassificationMetrics struct with computed metrics
 */
ClassificationMetrics compute_classification_metrics(
    const std::vector<int>& true_labels, const std::vector<int>& pred_labels);

/**
 * Perform k-fold cross validation
 * @param features feature matrix (n_samples x n_features)
 * @param labels label vector (n_samples)
 * @param k number of folds
 * @param random_seed seed for shuffling
 * @param fold_function function to run on each fold (train_features, train_labels, test_features,
 * test_labels) -> result
 * @return vector of results from each fold
 */
template <typename T, typename Func>
std::vector<T> k_fold_cross_validation(const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels,
    int k,
    int random_seed,
    Func fold_function)
{
    if (features.empty() || labels.empty())
    {
        throw std::runtime_error("features and labels must be non-empty");
    }
    if (features.size() != labels.size())
    {
        throw std::invalid_argument("features and labels must have the same size");
    }
    if (k <= 0)
    {
        throw std::invalid_argument("k must be > 0");
    }

    KFold splitter(static_cast<std::size_t>(k), true, static_cast<std::uint32_t>(random_seed));
    const std::vector<FoldSplit> folds = splitter.split(features.size());

    std::vector<T> results;
    results.reserve(folds.size());

    for (const auto& fold : folds)
    {
        std::vector<std::vector<double>> train_features, test_features;
        std::vector<int> train_labels, test_labels;

        train_features.reserve(fold.train_indices.size());
        train_labels.reserve(fold.train_indices.size());
        test_features.reserve(fold.test_indices.size());
        test_labels.reserve(fold.test_indices.size());

        for (const std::size_t idx : fold.train_indices)
        {
            train_features.push_back(features[idx]);
            train_labels.push_back(labels[idx]);
        }

        for (const std::size_t idx : fold.test_indices)
        {
            test_features.push_back(features[idx]);
            test_labels.push_back(labels[idx]);
        }

        T result = fold_function(train_features, train_labels, test_features, test_labels);
        results.push_back(result);
    }

    return results;
}

} // namespace statistics

#endif /* SRC_CORE_STATISTICS_MULTICLASSMETRICS_H_ */