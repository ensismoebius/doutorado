#ifndef NN_EXPERIMENTS_02_EXPERIMENT02EVALUATION_HPP
#define NN_EXPERIMENTS_02_EXPERIMENT02EVALUATION_HPP

#include <vector>

#include "nn/statistics/multi_class_metrics.hpp"

struct ParaconsistentMetrics
{
    double alpha = 0.0;
    double beta = 0.0;
    double G1 = 0.0;
    double G2 = 0.0;
};

using ClassificationMetrics = statistics::ClassificationMetrics;

struct FoldResult
{
    ClassificationMetrics metrics;
    ParaconsistentMetrics para_metrics;
    double fold_time_sec;
};

auto compute_paraconsistent_metrics(const std::vector<std::vector<double>>& features,
                                    const std::vector<int>& labels) -> ParaconsistentMetrics;

auto compute_classification_metrics(const std::vector<int>& true_labels,
                                    const std::vector<int>& pred_labels) -> ClassificationMetrics;

#endif // NN_EXPERIMENTS_02_EXPERIMENT02EVALUATION_HPP
