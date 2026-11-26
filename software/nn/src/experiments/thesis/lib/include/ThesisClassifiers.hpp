#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ClassificationResult.hpp"
#include "FoldResult.hpp"
#include "ThesisConfig.hpp"
#include "ThesisDataset.hpp"
#include "core/training/EpochResult.hpp"
#include "statistics/eer_scorer.hpp"

namespace thesis
{

/// Simulated dysphonia at test time: features extracted from the degraded voice, aligned
/// with view.samples. They replace the clean features of the TEST speakers only, scaled with
/// the scaler fitted on the clean training rows (the system was trained on healthy voices).
struct DegradedTestFeatures
{
    const std::vector<std::vector<double>>* vectors = nullptr;
    /// degradation.scope "probes": each test speaker's enrolment rows stay clean.
    bool enrol_clean = false;
    /// How many rows per speaker the scorer takes as enrolment: the FIRST ones in test
    /// order (GenuineImpostorEERScorer). Must equal the scorer's count.
    std::size_t enrol_per_speaker = 0;
};

// Train and evaluate the configured classifier using nested k-fold CV.
// feature_vectors: one vector per sample (aligned with view.samples).
// eer_scorer:      the EER/AUC strategy, required (there is no default scorer).
// global_bar_id / global_completed: optional ProgressManager bar updated after each outer fold.
auto run_classifier(const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const std::string& feature_label,
    const ThesisConfig& cfg,
    const statistics::IEERScorer& eer_scorer,
    uint32_t global_bar_id = 0,
    int* global_completed = nullptr,
    const DegradedTestFeatures* degraded = nullptr) -> ClassificationResult;

/// How one split of run_split is run and kept.
struct SplitRun
{
    int fold = 0;        // the split's index: messages, progress and the saved model's name
    int total_folds = 0; // the number of splits it belongs to
    /// false: nothing is written (an inner fold of the per-fold protocol, whose model only
    /// scores a hyperparameter point); true: the model goes to the run's models/ directory.
    bool save_model = false;
    const DegradedTestFeatures* degraded = nullptr; // simulated dysphonia of the test rows
    uint32_t global_bar_id = 0;                     // 0 = no bar
};

/// One train/test split of the classifier cfg declares — the per-fold protocol's building block
/// (Thesis-Per-Fold-Protocol.md §6): the feature scaler and the model are fitted on train_rows
/// only, and test_rows are scored once (EER, AUC). The rows are view.samples indices; a split
/// without train or test rows throws. Reads no CV key of cfg (k_folds, nested_cv).
auto run_split(const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const std::string& feature_label,
    const ThesisConfig& cfg,
    const statistics::IEERScorer& eer_scorer,
    const std::vector<std::size_t>& train_rows,
    const std::vector<std::size_t>& test_rows,
    const SplitRun& split) -> FoldResult;

/// Trainable parameters of the classifier cfg declares, on these features' width (0 for the
/// cosine control).
auto classifier_param_count(const ThesisDatasetView& view,
    const std::vector<std::vector<double>>& feature_vectors,
    const ThesisConfig& cfg,
    const statistics::IEERScorer& eer_scorer) -> std::size_t;

// Compute mean and std over fold accuracies.
void compute_aggregate_stats(ClassificationResult& result);

} // namespace thesis
