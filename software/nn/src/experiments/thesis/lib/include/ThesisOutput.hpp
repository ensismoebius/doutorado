#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "ThesisClassifiers.hpp"
#include "ThesisConfig.hpp"
#include "ThesisParaconsistent.hpp"

namespace thesis
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
// Also records the Guayaquil-style run diagnostics: per-run timing (train_ms/infer_ms),
// model complexity (param_count), and the config_hash for provenance/determinism.
// Path: results_dir/e05_{run_tag}_summary.json
void write_summary_json(const std::string& results_dir,
    const std::string& run_tag,
    const ThesisConfig& cfg,
    const std::vector<ClassificationResult>& results,
    const std::vector<ParaconsistentScore>& scores,
    int n_subjects,
    int n_stimuli,
    size_t n_samples,
    std::size_t config_hash = 0);

// Write pgfplots DAT file for thesis figures.
// Path: results_dir/e05_{run_tag}_comparison.dat
void write_comparison_dat(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<ClassificationResult>& results);

// Write per-epoch learning curves for every fold (the Guayaquil epoch-history analog).
// One row per (feature_set, fold, epoch): train/val loss, epoch time, and — for
// spiking classifiers — mean firing rate + synaptic-op count. Skipped (no file)
// when no result carries any epoch history.
// Path: results_dir/e05_{run_tag}_learning_curves.dat
void write_learning_curves_dat(const std::string& results_dir,
    const std::string& run_tag,
    const std::vector<ClassificationResult>& results);

// Ensure directory exists (creates recursively).
void ensure_dir(const std::string& path);

} // namespace thesis
