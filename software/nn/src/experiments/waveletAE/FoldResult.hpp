/**
 * @file src/experiments/waveletAE/FoldResult.hpp
 * @brief FoldResult struct (extracted from WaveletAEEvaluation.hpp).
 */

#ifndef NN_EXPERIMENTS_02_FOLDRESULT_HPP
#define NN_EXPERIMENTS_02_FOLDRESULT_HPP

#include "ParaconsistentMetrics.hpp"
#include "statistics/multi_class_metrics.hpp"

using ClassificationMetrics = statistics::ClassificationMetrics;

struct FoldResult
{
    ClassificationMetrics metrics;
    ParaconsistentMetrics para_metrics;
    double fold_time_sec;
};

#endif // NN_EXPERIMENTS_02_FOLDRESULT_HPP
