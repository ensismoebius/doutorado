#ifndef SRC_CORE_STATISTICS_BINARYCLASSIFICATIONMETRICS_H_
#define SRC_CORE_STATISTICS_BINARYCLASSIFICATIONMETRICS_H_

#include <cstddef>
#include <vector>

namespace statistics
{

/**
 * @file binary_classification_metrics.hpp
 * @brief Precision/recall/F1 for the positive class (label == 1) only.
 *
 * NOT the same number as compute_classification_metrics() in multi_class_metrics.hpp:
 * that one macro-averages over every class (each class weighted equally), which for a
 * binary problem differs from "precision of the positive class" whenever the two
 * classes' per-class precision/recall differ. Use this one when the positive class is
 * what actually matters (e.g. "reconstruction error below threshold" as the positive
 * outcome); use compute_classification_metrics() for a balanced multi-class summary.
 */
inline void binary_precision_recall_f1(const std::vector<int>& y_true,
    const std::vector<int>& y_pred,
    float& precision,
    float& recall,
    float& f1)
{
    int tp = 0;
    int fp = 0;
    int fn = 0;
    for (std::size_t i = 0; i < y_true.size() && i < y_pred.size(); ++i)
    {
        if (y_true[i] == 1 && y_pred[i] == 1) ++tp;
        if (y_true[i] == 0 && y_pred[i] == 1) ++fp;
        if (y_true[i] == 1 && y_pred[i] == 0) ++fn;
    }

    precision = (tp + fp) > 0 ? static_cast<float>(tp) / static_cast<float>(tp + fp) : 0.0f;
    recall = (tp + fn) > 0 ? static_cast<float>(tp) / static_cast<float>(tp + fn) : 0.0f;
    f1 = (precision + recall) > 0.0f ? (2.0f * precision * recall) / (precision + recall) : 0.0f;
}

} // namespace statistics

#endif /* SRC_CORE_STATISTICS_BINARYCLASSIFICATIONMETRICS_H_ */
