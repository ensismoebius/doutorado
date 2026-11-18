/**
 * @file src/experiments/thesis/lib/include/ThesisPerFold.hpp
 * @brief The per-fold protocol: extractor and classifier hyperparameters chosen inside each
 *        outer fold, on its training speakers only (.wiki/Experiments/Thesis-Per-Fold-Protocol.md).
 *
 * Why it exists: the fixed-configuration run takes the extractor ranked by D_pen on all 15
 * speakers, so the 3 test speakers of a fold helped elect the extractor they are scored with,
 * and it never tunes the classifier. Here one process does, per (repetition, outer fold):
 *
 *   outer fold f ── 12 training speakers ──┬─ every extractor candidate (PCA/AE fitted here)
 *                                          ├─ D_pen on these rows -> extractor winner
 *                                          ├─ inner GroupKFold(inner_k) -> grid winner
 *                                          └─ refit the winner on all 12
 *                └─ 3 test speakers ──────── scored ONCE
 *
 * Rows, with T = the text split (make_text_split) of the repetition's seed:
 *   training rows  = training speakers ∩ T.train   (extractor fit, D_pen, refit)
 *   test rows      = test speakers ∩ T.test        (the single evaluation)
 *   inner train    = inner-train speakers ∩ T.train
 *   inner validate = inner-validation speakers ∩ T.test (the outer test, one level down)
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ClassificationResult.hpp"
#include "FeatureDiskCache.hpp"
#include "FoldResult.hpp"
#include "ThesisConfig.hpp"
#include "ThesisDataset.hpp"
#include "ThesisFeatureExtraction.hpp"
#include "statistics/eer_scorer.hpp"

namespace thesis
{

/// The rows of one outer fold.
struct OuterFoldRows
{
    std::vector<std::size_t> speaker_rows; // every row of the training speakers, ascending
    std::vector<std::size_t> train;        // speaker_rows ∩ text-split train
    std::vector<std::size_t> test;         // the test speakers' rows ∩ text-split test
};

/// One inner fold (hyperparameter search) inside an outer fold.
struct InnerFoldRows
{
    std::vector<std::size_t> train;
    std::vector<std::size_t> validation;
};

/// The outer folds of the repetition cfg describes: GroupKFoldPolicy(k_folds, shuffle, seed) by
/// speaker, intersected with the text split exactly as the fixed-configuration flat CV does —
/// the same split, so the two protocols are compared fold by fold. A fold without train or
/// test rows throws.
auto outer_fold_rows(const std::vector<ThesisSample>& samples, const ThesisConfig& cfg)
    -> std::vector<OuterFoldRows>;

/// The inner folds of one outer fold: GroupKFoldPolicy(inner_k, shuffle, seed ^ 0xDEADBEEF) over
/// its training speakers only. Validation rows are the validation speakers' text-split TEST
/// rows, as the outer test is; a fold without train or validation rows throws.
auto inner_fold_rows(
    const std::vector<ThesisSample>& samples, const OuterFoldRows& outer, const ThesisConfig& cfg)
    -> std::vector<InnerFoldRows>;

/// D_pen of every candidate on `rows` only, in candidate order. A non-finite score throws.
auto rank_extractors(const std::vector<ThesisSample>& samples,
    const std::vector<std::size_t>& rows,
    const std::vector<FeatureSet>& candidates) -> std::vector<ExtractorRank>;

/// The lowest D_pen; the first in profile order on a tie.
auto extractor_winner(const std::vector<ExtractorRank>& ranks) -> std::size_t;

/// The grid, nested widths > residual blocks > learning rates > weight decays (that order is
/// the last tie-break). Scores and parameter counts are left for the caller.
auto grid_points(const ThesisConfig::Selection& selection) -> std::vector<GridPoint>;

/// cfg with one grid point's classifier and training keys declared.
auto grid_config(const ThesisConfig& cfg, const GridPoint& point) -> ThesisConfig;

/// The lowest mean inner EER; ties -> fewer trainable parameters, then grid order.
auto grid_winner(const std::vector<GridPoint>& grid) -> std::size_t;

/// Handcrafted candidates' features depend on no other sample and on no seed: extracted once
/// per process and shared by every fold and repetition. Keyed by (candidate, modality).
/// With `disk` set, the ranked signal's candidates of the clean view also go through the
/// on-disk cache shared by every process (FeatureDiskCache.hpp). One cache object serves one
/// clean view (and one degraded view): the process loads them once.
struct CandidateFeatureCache
{
    std::map<std::pair<std::size_t, std::string>, FeatureSet> clean;
    std::map<std::pair<std::size_t, std::string>, FeatureSet> degraded;
    const FeatureDiskCache* disk = nullptr;
    std::optional<std::uint64_t> clean_fingerprint; // view_fingerprint of the clean view
};

/// Simulated dysphonia of the test speakers (degradation.enabled): the same samples with the
/// degraded voice, and how the scorer enrols.
struct PerFoldDegradation
{
    const ThesisDatasetView* view = nullptr;
    bool enrol_clean = false;
    std::size_t enrol_per_speaker = 0;
};

/// One repetition of a selection.enabled profile: every outer fold as described above. The
/// result carries every fold's choices (FoldResult::selection); its run-level param_count is
/// nullopt (each fold's winner has its own).
auto run_per_fold(const ThesisDatasetView& view,
    const ThesisConfig& cfg,
    const statistics::IEERScorer& scorer,
    const PerFoldDegradation* degradation,
    CandidateFeatureCache& cache,
    uint32_t global_bar_id = 0) -> ClassificationResult;

/// The warm step (thesis --warm-feature-cache): every handcrafted candidate of the ranked
/// signal extracted from `view` into cache.disk, which must be in Write mode; no fold runs.
/// Returns how many candidates went through the disk cache.
auto warm_feature_cache(
    const ThesisDatasetView& view, const ThesisConfig& cfg, CandidateFeatureCache& cache)
    -> std::size_t;

/// The label of a per-fold result ("per-fold-voice", "per-fold-eeg", "per-fold-fused-late").
auto per_fold_label(const ThesisConfig& cfg) -> std::string;

} // namespace thesis
