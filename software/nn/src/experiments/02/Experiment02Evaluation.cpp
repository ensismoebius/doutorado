#include "Experiment02Evaluation.hpp"

#include <map>
#include <string>
#include <vector>

#include "nn/paraconsistent/paraconsistent.h"

auto compute_paraconsistent_metrics(const std::vector<std::vector<double>>& features,
                                    const std::vector<int>& labels) -> ParaconsistentMetrics
{
    if (features.empty() || labels.empty())
    {
        return ParaconsistentMetrics{};
    }

    std::map<std::string, std::vector<std::vector<double>>> class_features;
    for (std::size_t i = 0; i < features.size(); ++i)
    {
        std::string class_key = std::to_string(labels[i]);
        class_features[class_key].push_back(features[i]);
    }

    if (class_features.empty())
    {
        return ParaconsistentMetrics{};
    }

    unsigned int n_classes = class_features.size();
    unsigned int n_samples_per_class = class_features.begin()->second.size();
    unsigned int feature_dim = features[0].size();

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
