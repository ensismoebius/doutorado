/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * Multi-class classification metrics implementation
 */

#include "multiClassMetrics.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

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

    // Confusion matrix
    Eigen::MatrixXi cm = Eigen::MatrixXi::Zero(n_classes, n_classes);
    for (int i = 0; i < n_samples; ++i)
    {
        cm(true_labels[i], pred_labels[i])++;
    }

    // Per-class metrics
    std::vector<double> precisions, recalls, f1s;
    int total_correct = 0;

    for (int c = 0; c < n_classes; ++c)
    {
        int tp = cm(c, c);
        int fp = 0, fn = 0;
        for (int j = 0; j < n_classes; ++j)
        {
            if (j != c)
            {
                fp += cm(j, c);
                fn += cm(c, j);
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
        int tp = cm(0, 0), tn = cm(1, 1), fp = cm(1, 0), fn = cm(0, 1);
        double numerator = static_cast<double>(tp * tn - fp * fn);
        double denominator =
            std::sqrt(static_cast<double>((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)));
        metrics.mcc = denominator > 0 ? numerator / denominator : 0.0;
    }

    return metrics;
}

} // namespace statistics