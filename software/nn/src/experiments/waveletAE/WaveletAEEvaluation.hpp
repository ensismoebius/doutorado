/**
 * @file src/experiments/waveletAE/WaveletAEEvaluation.hpp
 * @brief WaveletAEevaluation.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_EXPERIMENTS_02_EXPERIMENT02EVALUATION_HPP
#define NN_EXPERIMENTS_02_EXPERIMENT02EVALUATION_HPP

#include <vector>

#include "FoldResult.hpp"
#include "ParaconsistentMetrics.hpp"
#include "statistics/multi_class_metrics.hpp"

auto compute_paraconsistent_metrics(const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels) -> ParaconsistentMetrics;

auto compute_classification_metrics(const std::vector<int>& true_labels,
    const std::vector<int>& pred_labels) -> ClassificationMetrics;

#endif // NN_EXPERIMENTS_02_EXPERIMENT02EVALUATION_HPP
