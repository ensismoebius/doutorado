#ifndef SRC_CORE_STATISTICS_MULTICLASSMETRICS_H_
#define SRC_CORE_STATISTICS_MULTICLASSMETRICS_H_

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

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
ClassificationMetrics compute_classification_metrics(const std::vector<int>& true_labels,
                                                     const std::vector<int>& pred_labels);

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
                                       const std::vector<int>& labels, int k, int random_seed,
                                       Func fold_function)
{
    int n_samples = features.size();
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle indices
    std::mt19937 rng(random_seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    int fold_size = n_samples / k;
    std::vector<T> results;

    for (int fold = 0; fold < k; ++fold)
    {
        // Split data
        std::vector<std::vector<double>> train_features, test_features;
        std::vector<int> train_labels, test_labels;

        for (int i = 0; i < n_samples; ++i)
        {
            int idx = indices[i];
            if (i >= fold * fold_size && i < (fold + 1) * fold_size)
            {
                test_features.push_back(features[idx]);
                test_labels.push_back(labels[idx]);
            }
            else
            {
                train_features.push_back(features[idx]);
                train_labels.push_back(labels[idx]);
            }
        }

        // Run fold function
        T result = fold_function(train_features, train_labels, test_features, test_labels);
        results.push_back(result);
    }

    return results;
}

} // namespace statistics

#endif /* SRC_CORE_STATISTICS_MULTICLASSMETRICS_H_ */