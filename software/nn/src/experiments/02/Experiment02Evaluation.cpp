/**
 * @file src/experiments/02/Experiment02Evaluation.cpp
 * @brief Implementation for Experiment02evaluation.
 *

 */

#include "Experiment02Evaluation.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "paraconsistent/paraconsistent.h"

auto compute_paraconsistent_metrics(const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels) -> ParaconsistentMetrics
{
    if (features.empty() || labels.empty() || features.size() != labels.size() ||
        features.front().empty())
    {
        return ParaconsistentMetrics{};
    }

    std::map<std::string, std::vector<std::vector<double>>> class_features;
    for (std::size_t i = 0; i < features.size(); ++i)
    {
        std::string class_key = std::to_string(labels[i]);
        class_features[class_key].push_back(features[i]);
    }

    const unsigned int n_classes = static_cast<unsigned int>(class_features.size());
    unsigned int n_samples_per_class = class_features.begin()->second.size();
    for (const auto& [_, class_samples] : class_features)
    {
        n_samples_per_class = std::min<unsigned int>(
            n_samples_per_class, static_cast<unsigned int>(class_samples.size()));
    }
    const unsigned int feature_dim = static_cast<unsigned int>(features.front().size());

    if (n_samples_per_class == 0 || feature_dim == 0)
    {
        return ParaconsistentMetrics{};
    }

    double alpha = calculate_alpha(n_classes, n_samples_per_class, feature_dim, class_features);
    double beta = calculate_beta(n_classes, n_samples_per_class, feature_dim, class_features);

    ParaconsistentMetrics metrics;
    metrics.alpha = alpha;
    metrics.beta = beta;
    metrics.G1 = calculate_certainty_degree_g1(alpha, beta);
    metrics.G2 = calculate_contradiction_degree_g2(alpha, beta);

    return metrics;
}

auto compute_classification_metrics(const std::vector<int>& true_labels,
    const std::vector<int>& pred_labels) -> ClassificationMetrics
{
    return statistics::compute_classification_metrics(true_labels, pred_labels);
}
