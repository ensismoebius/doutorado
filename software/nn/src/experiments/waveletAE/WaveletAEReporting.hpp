/**
 * @file src/experiments/waveletAE/WaveletAEReporting.hpp
 * @brief WaveletAEreporting.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_EXPERIMENTS_02_EXPERIMENT02REPORTING_HPP
#define NN_EXPERIMENTS_02_EXPERIMENT02REPORTING_HPP

#include <string>
#include <vector>

#include "WaveletAEConfig.hpp"
#include "WaveletAEEvaluation.hpp"

struct AggregatedFoldResults
{
    ClassificationMetrics classification;
    ParaconsistentMetrics paraconsistent;
    double total_time_sec = 0.0;
};

auto aggregate_fold_results(const std::vector<FoldResult>& fold_results) -> AggregatedFoldResults;

auto write_wavelet_results_csv(const std::string& csv_path,
    const ExperimentConfig& config,
    const std::string& wavelet_name,
    const AggregatedFoldResults& aggregated) -> void;

#endif // NN_EXPERIMENTS_02_EXPERIMENT02REPORTING_HPP
