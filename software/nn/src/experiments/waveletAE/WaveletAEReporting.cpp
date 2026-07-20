/**
 * @file src/experiments/waveletAE/WaveletAEReporting.cpp
 * @brief Implementation for WaveletAEreporting.
 *

 */

#include "WaveletAEReporting.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

auto aggregate_fold_results(const std::vector<FoldResult>& fold_results) -> AggregatedFoldResults
{
    if (fold_results.empty())
    {
        return AggregatedFoldResults{};
    }

    AggregatedFoldResults aggregated;
    for (const auto& result : fold_results)
    {
        aggregated.classification.accuracy += result.metrics.accuracy;
        aggregated.classification.precision += result.metrics.precision;
        aggregated.classification.recall += result.metrics.recall;
        aggregated.classification.f1_score += result.metrics.f1_score;
        aggregated.classification.mcc += result.metrics.mcc;

        aggregated.paraconsistent.alpha += result.para_metrics.alpha;
        aggregated.paraconsistent.beta += result.para_metrics.beta;
        aggregated.paraconsistent.G1 += result.para_metrics.G1;
        aggregated.paraconsistent.G2 += result.para_metrics.G2;

        aggregated.total_time_sec += result.fold_time_sec;
    }

    const double n_folds = static_cast<double>(fold_results.size());
    aggregated.classification.accuracy /= n_folds;
    aggregated.classification.precision /= n_folds;
    aggregated.classification.recall /= n_folds;
    aggregated.classification.f1_score /= n_folds;
    aggregated.classification.mcc /= n_folds;

    aggregated.paraconsistent.alpha /= n_folds;
    aggregated.paraconsistent.beta /= n_folds;
    aggregated.paraconsistent.G1 /= n_folds;
    aggregated.paraconsistent.G2 /= n_folds;

    return aggregated;
}

auto write_wavelet_results_csv(const std::string& csv_path,
    const ExperimentConfig& config,
    const std::string& wavelet_name,
    const AggregatedFoldResults& aggregated) -> void
{
    std::ofstream csv_file(csv_path);
    if (!csv_file)
    {
        throw std::runtime_error("Failed to open CSV output file: " + csv_path);
    }

    csv_file << "experiment_id,wavelet_name,decomposition_depth,alpha,beta,G1,G2,"
             << "accuracy,precision,recall,f1_score,mcc,total_time_sec\n";
    csv_file << config.id << "," << wavelet_name << "," << config.max_decomposition_depth << ","
             << aggregated.paraconsistent.alpha << "," << aggregated.paraconsistent.beta << ","
             << aggregated.paraconsistent.G1 << "," << aggregated.paraconsistent.G2 << ","
             << aggregated.classification.accuracy << "," << aggregated.classification.precision
             << "," << aggregated.classification.recall << "," << aggregated.classification.f1_score
             << "," << aggregated.classification.mcc << "," << aggregated.total_time_sec << "\n";
}
