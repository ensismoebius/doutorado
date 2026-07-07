#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "E05Classifiers.hpp"
#include "E05Config.hpp"
#include "E05Paraconsistent.hpp"

namespace e05
{

// Write per-fold accuracy/EER results to CSV.
// Path: results_dir/e05_{run_tag}_metrics.csv
void write_metrics_csv(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<ClassificationResult>& results);

// Write paraconsistent scores to CSV.
// Path: results_dir/e05_{run_tag}_paraconsistent.csv
void write_paraconsistent_csv(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<ParaconsistentScore>& scores);

// Write summary JSON with config, seed, aggregated stats, and the dataset
// composition actually used (n_subjects/n_stimuli/n_samples — after the
// paired audio+EEG drop, see load_dataset), so the run is self-describing.
// Path: results_dir/e05_{run_tag}_summary.json
void write_summary_json(const std::string& results_dir,
    const std::string& run_tag,
    const E05Config& cfg,
    const std::vector<ClassificationResult>& results,
    const std::vector<ParaconsistentScore>& scores,
    int n_subjects,
    int n_stimuli,
    size_t n_samples);

// Write pgfplots DAT file for thesis figures.
// Path: results_dir/e05_{run_tag}_comparison.dat
void write_comparison_dat(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<ClassificationResult>& results);

// Ensure directory exists (creates recursively).
void ensure_dir(const std::string& path);

} // namespace e05
