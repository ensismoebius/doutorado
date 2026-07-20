#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ThesisConfig.hpp"
#include "ThesisDataset.hpp"
#include "core/training/EpochResult.hpp"
#include "statistics/eer_scorer.hpp"

namespace thesis
{

// Per-fold classification result.
struct FoldResult
{
    int fold = 0;
    double accuracy = 0.0;
    double eer = 0.0;
    double loss = 0.0;
    double f1 = 0.0;
    double precision = 0.0;
    double recall = 0.0;
    double specificity = 0.0; // macro TN/(TN+FP)
    double auc = 0.0;
    std::string model_path;

    // Per-run stats (mirrors what the Guayaquil/Guayaquil pipeline records per run).
    double train_ms = 0.0; // wall-clock to produce this fold's model
    double infer_ms = 0.0; // wall-clock to score the held-out test fold
    // Learning curve of the fold's final model: per-epoch train/val loss, epoch
    // time, and (SNN) mean spike rate + synaptic-operation count. Empty when the
    // trainer produced no epochs (e.g. a cache hit).
    std::vector<nn::training::EpochResult> history;
};

// Full classification result over nested CV.
struct ClassificationResult
{
    std::string feature_set_label;
    std::string classifier_type;
    std::string text_mode;
    std::vector<FoldResult> outer_folds;
    double mean_accuracy = 0.0;
    double std_accuracy = 0.0;
    double ci95_accuracy = 0.0;
    double mean_f1 = 0.0;
    double std_f1 = 0.0;
    double mean_precision = 0.0;
    double mean_recall = 0.0;
    double mean_specificity = 0.0;
    double std_specificity = 0.0;
    double mean_eer = 0.0;
    double std_eer = 0.0;
    double ci95_eer = 0.0;
    double mean_auc = 0.0;
    double std_auc = 0.0;

    // Run-level cost / complexity stats (the Guayaquil-style "info about the run").
    std::size_t param_count = 0;  // trainable parameters in the classifier
    double mean_train_ms = 0.0;   // mean per-fold training wall-clock
    double mean_infer_ms = 0.0;   // mean per-fold inference wall-clock
    double mean_spike_rate = 0.0; // SNN: mean firing rate at last epoch (NaN→skipped)
    long long final_sops = 0LL;   // SNN: synaptic ops/forward pass at last epoch
};

// Train and evaluate the configured classifier using nested k-fold CV.
// feature_vectors: one vector per sample (aligned with view.samples).
// eer_scorer:      pluggable EER strategy; nullptr → GenuineImpostorEERScorer (SOTA default).
// global_bar_id / global_completed: optional ProgressManager bar updated after each outer fold.
auto run_classifier(const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const std::string& feature_label,
    const ThesisConfig& cfg,
    const statistics::IEERScorer* eer_scorer = nullptr,
    uint32_t global_bar_id = 0,
    int* global_completed = nullptr) -> ClassificationResult;

// Compute mean and std over fold accuracies.
void compute_aggregate_stats(ClassificationResult& result);

} // namespace thesis
