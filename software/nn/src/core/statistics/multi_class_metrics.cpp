/**
 * @file multi_class_metrics.cpp
 * @brief Multi-class classification metrics implementation (snake_case translation unit).
 */

/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * Multi-class classification metrics implementation
 */

#include "nn/statistics/multi_class_metrics.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "nn/tensor/Tensor.hpp"

namespace statistics
{

ClassificationMetrics compute_classification_metrics(const std::vector<int>& true_labels,
                                                     const std::vector<int>& pred_labels)
{
    if (true_labels.size() != pred_labels.size())
    {
        throw std::runtime_error("Label vectors must have same size");
    }

    int n_samples = true_labels.size();
    int n_classes = 0;
    for (int label : true_labels)
    {
        n_classes = std::max(n_classes, label + 1);
    }

    // Confusion matrix - using float for now (consider IntTensor for integer precision)
    nn::Tensor cm(n_classes, n_classes);
    for (int i = 0; i < n_samples; ++i)
    {
        cm.at(true_labels[i], pred_labels[i]) += 1.0f;
    }

    // Per-class metrics
    std::vector<double> precisions, recalls, f1s;
    int total_correct = 0;

    for (int c = 0; c < n_classes; ++c)
    {
        int tp = static_cast<int>(cm.at(c, c));
        int fp = 0, fn = 0;
        for (int j = 0; j < n_classes; ++j)
        {
            if (j != c)
            {
                fp += static_cast<int>(cm.at(j, c));
                fn += static_cast<int>(cm.at(c, j));
            }
        }

        double precision = tp + fp > 0 ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall = tp + fn > 0 ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

        precisions.push_back(precision);
        recalls.push_back(recall);
        f1s.push_back(f1);
        total_correct += tp;
    }

    ClassificationMetrics metrics;
    metrics.accuracy = static_cast<double>(total_correct) / n_samples;
    metrics.precision = std::accumulate(precisions.begin(), precisions.end(), 0.0) / n_classes;
    metrics.recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / n_classes;
    metrics.f1_score = std::accumulate(f1s.begin(), f1s.end(), 0.0) / n_classes;
    metrics.balanced_accuracy =
        metrics.recall; // For multi-class, balanced accuracy is average recall

    // Simplified MCC calculation (for binary case, extend for multi-class)
    if (n_classes == 2)
    {
        int tp = cm.at(0, 0), tn = cm.at(1, 1), fp = cm.at(1, 0), fn = cm.at(0, 1);
        double numerator = static_cast<double>(tp * tn - fp * fn);
        double denominator =
            std::sqrt(static_cast<double>((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)));
        metrics.mcc = denominator > 0 ? numerator / denominator : 0.0;
    }

    return metrics;
}

} // namespace statistics