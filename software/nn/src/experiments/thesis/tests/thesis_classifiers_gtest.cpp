// Unit tests for ThesisClassifiers: compute_aggregate_stats, run_classifier with
// synthetic data (2 classes, small feature dim, 2-fold CV).
// No SQLite, no real audio — everything synthetic.

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../lib/include/ThesisClassifiers.hpp"
#include "../lib/include/ThesisConfig.hpp"
#include "../lib/include/ThesisDataset.hpp"
#include "statistics/ClassificationEERScorer.hpp"
#include "statistics/eer_scorer.hpp"

using namespace thesis;

// ─── compute_aggregate_stats ─────────────────────────────────────────────────

TEST(ThesisAggregateStats, EmptyFoldsNoOp)
{
    ClassificationResult r;
    compute_aggregate_stats(r);
    EXPECT_DOUBLE_EQ(r.mean_accuracy, 0.0);
    EXPECT_DOUBLE_EQ(r.mean_eer, 0.0);
    EXPECT_DOUBLE_EQ(r.std_accuracy, 0.0);
}

TEST(ThesisAggregateStats, SingleFold)
{
    ClassificationResult r;
    r.outer_folds.push_back({0, 0.8, 0.2, 0.0});
    compute_aggregate_stats(r);
    EXPECT_DOUBLE_EQ(r.mean_accuracy, 0.8);
    EXPECT_DOUBLE_EQ(r.mean_eer, 0.2);
    EXPECT_NEAR(r.std_accuracy, 0.0, 1e-12);
}

TEST(ThesisAggregateStats, TwoFoldsMean)
{
    ClassificationResult r;
    r.outer_folds.push_back({0, 0.6, 0.1, 0.0});
    r.outer_folds.push_back({1, 0.8, 0.3, 0.0});
    compute_aggregate_stats(r);
    EXPECT_NEAR(r.mean_accuracy, 0.7, 1e-12);
    EXPECT_NEAR(r.mean_eer, 0.2, 1e-12);
}

TEST(ThesisAggregateStats, TwoFoldsStd)
{
    ClassificationResult r;
    r.outer_folds.push_back({0, 0.6, 0.0, 0.0});
    r.outer_folds.push_back({1, 0.8, 0.0, 0.0});
    compute_aggregate_stats(r);
    // mean = 0.7; sample variance = ((0.6-0.7)^2 + (0.8-0.7)^2) / (2-1) = 0.02
    // sample SD = sqrt(0.02) ≈ 0.141421356 (audit M-3: sample SD, not population).
    EXPECT_NEAR(r.std_accuracy, 0.14142135623730953, 1e-12);
}

TEST(ThesisAggregateStats, AllSameMeanAndZeroStd)
{
    ClassificationResult r;
    for (int i = 0; i < 5; ++i) r.outer_folds.push_back({i, 0.9, 0.05, 0.0});
    compute_aggregate_stats(r);
    EXPECT_NEAR(r.mean_accuracy, 0.9, 1e-12);
    EXPECT_NEAR(r.std_accuracy, 0.0, 1e-12);
}

// ─── run_classifier (synthetic) ──────────────────────────────────────────────

namespace
{
// Build a minimal ThesisConfig for a quick run: 2-fold CV, 5 epochs, batch=4.
ThesisConfig make_fast_cfg()
{
    ThesisConfig cfg;
    cfg.experiment.run_tag = "test";
    cfg.experiment.seed = 42;

    cfg.dataset.modality = "eeg";
    cfg.dataset.root = "/dev/null"; // never accessed in unit test
    cfg.dataset.results_dir = "/tmp/";

    cfg.feature_extraction.strategy = "handcrafted";

    cfg.classifier.type = "rnn";
    cfg.classifier.text_mode = "independent";

    cfg.training.epochs = 5;
    cfg.training.learning_rate = 1e-3f;
    cfg.training.samples_per_batch = 4;
    cfg.training.early_stop_patience = 0; // disabled
    cfg.training.k_folds = 2;

    cfg.paraconsistent.enabled = false;
    return cfg;
}

// Build a synthetic ThesisDatasetView: n_subjects classes, samples_per_subject
// samples each.  Audio tensor is a 1×D row of zeros (non-empty so feature
// extraction doesn't fall back to the 256-zero placeholder in ThesisDataset, but
// here we bypass extraction entirely and supply feature_vectors directly).
ThesisDatasetView make_view(int n_subjects, int samples_per_subject)
{
    ThesisDatasetView view;
    view.n_subjects = n_subjects;
    view.n_stimuli = 1;

    int id = 1;
    for (int s = 0; s < n_subjects; ++s, ++id)
    {
        for (int k = 0; k < samples_per_subject; ++k)
        {
            ThesisSample sam;
            sam.subject_id = id;
            sam.stimulus = k % 5; // vowel index 0-4
            sam.text_phrase = "a";
            // eeg/audio left default (0-dim scalar) — feature_vectors supplied externally
            view.samples.push_back(sam);
        }
    }
    return view;
}

// Build linearly separable feature vectors: class c gets feature vector [c, 0, …].
std::vector<std::vector<double>> make_features(const ThesisDatasetView& view, int feat_dim = 8)
{
    int id_idx = 0;
    int prev = -1;
    std::vector<std::vector<double>> fvs;
    fvs.reserve(view.samples.size());
    for (const auto& s : view.samples)
    {
        if (s.subject_id != prev)
        {
            ++id_idx;
            prev = s.subject_id;
        }
        std::vector<double> fv(static_cast<size_t>(feat_dim), 0.0);
        fv[0] = static_cast<double>(id_idx);
        fvs.push_back(fv);
    }
    return fvs;
}
} // namespace

TEST(ThesisRunClassifier, ThrowsOnEmptyFeatures)
{
    auto view = make_view(2, 4);
    ThesisConfig cfg = make_fast_cfg();
    EXPECT_THROW(run_classifier(view, {}, "test", cfg), std::invalid_argument);
}

TEST(ThesisRunClassifier, ThrowsOnSizeMismatch)
{
    auto view = make_view(2, 4);
    ThesisConfig cfg = make_fast_cfg();
    std::vector<std::vector<double>> fvs(3, std::vector<double>(8, 0.0));
    EXPECT_THROW(run_classifier(view, fvs, "test", cfg), std::invalid_argument);
}

// GroupKFoldPolicy with k=2 needs >= 4 subjects:
//   round-robin outer: {1,3}→fold0, {2,4}→fold1
//   inner training per outer fold has exactly 2 groups → inner 2-fold valid.
//
// Use ClassificationEERScorer in unit tests: GenuineImpostorEERScorer requires
// each test speaker to have been seen during training for non-NaN EER, which
// grouped CV deliberately prevents (that is the correct real-world behaviour,
// but makes unit-test assertions fragile).
TEST(ThesisRunClassifier, ReturnsFoldCountMatchingKFolds)
{
    const int n_subjects = 4;
    const int sps = 6;
    auto view = make_view(n_subjects, sps);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth", cfg, &scorer);

    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
}

TEST(ThesisRunClassifier, VerificationOnlyAccuracyIsNaN)
{
    // Verification-only protocol (audit C-1): closed-set accuracy is not reported
    // (speaker-disjoint folds), so mean_accuracy aggregates to NaN. The run must
    // still produce one result per outer fold.
    auto view = make_view(4, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth", cfg, &scorer);

    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
    EXPECT_TRUE(std::isnan(result.mean_accuracy));
}

TEST(ThesisRunClassifier, FeatureLabelPropagated)
{
    auto view = make_view(4, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "my-label", cfg, &scorer);
    EXPECT_EQ(result.feature_set_label, "my-label");
}

TEST(ThesisRunClassifier, FoldIndicesAreSequential)
{
    auto view = make_view(4, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth", cfg, &scorer);
    for (int i = 0; i < static_cast<int>(result.outer_folds.size()); ++i)
        EXPECT_EQ(result.outer_folds[static_cast<size_t>(i)].fold, i);
}

TEST(ThesisRunClassifier, DsnnPathRuns)
{
    auto view = make_view(4, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    cfg.classifier.type = "dsnn";
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth-dsnn", cfg, &scorer);
    EXPECT_EQ(result.classifier_type, "dsnn");
    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
    // Verification-only: closed-set accuracy not reported (audit C-1).
    EXPECT_TRUE(std::isnan(result.mean_accuracy));
}

// Regularization enabled (decoupled L2 weight decay + firing-rate band penalty)
// on the dsnn path must train and produce one result per outer fold without
// throwing — exercises Adam::weight_decay and ThesisDsnnClassifier::add_firing_rate_grad.
TEST(ThesisRunClassifier, DsnnWithRegularizationRuns)
{
    auto view = make_view(4, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    cfg.classifier.type = "dsnn";
    cfg.training.weight_decay = 1e-3f;
    cfg.training.firing_rate_reg_lambda = 0.05f;
    cfg.training.firing_rate_min = 0.05f;
    cfg.training.firing_rate_max = 0.80f;
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth-dsnn-reg", cfg, &scorer);
    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
}

// Weight decay on the rnn path must also train cleanly (firing-rate reg is inert
// for the non-spiking ResNet classifier).
TEST(ThesisRunClassifier, RnnWithWeightDecayRuns)
{
    auto view = make_view(4, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    cfg.training.weight_decay = 1e-3f;
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth-rnn-wd", cfg, &scorer);
    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
}

// tdBN enabled on the dsnn path must train and produce one result per outer fold
// without throwing — exercises the tdBN forward/backward wired into the DSNN.
TEST(ThesisRunClassifier, DsnnWithTdbnRuns)
{
    auto view = make_view(4, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    cfg.classifier.type = "dsnn";
    cfg.training.batch_normalization = "threshold-dependent";
    cfg.training.tdbn_alpha = 1.0f;
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth-dsnn-tdbn", cfg, &scorer);
    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
    EXPECT_FALSE(std::isnan(result.mean_eer)); // EER is the verification-protocol metric
}

// Comparative benchmark (Part 4): train the synthetic Thesis DSNN three ways —
// (a) no normalization, (b) tdBN with α·V_th = 1 (equivalent to conventional BN
// at V_th=1, since the project has no standalone BatchNorm layer), and
// (c) tdBN with α=2 (threshold-scaled). Records EER/AUC and asserts every variant
// trains stably (finite metrics, full fold count). Closed-set accuracy/F1/etc. are
// NaN by design under the verification-only protocol, so EER/AUC are reported.
TEST(ThesisRunClassifier, TdbnComparativeBenchmark)
{
    auto view = make_view(4, 6);
    auto fvs = make_features(view);
    statistics::ClassificationEERScorer scorer;

    struct Variant
    {
        const char* name;
        const char* bn;
        float alpha;
    };
    const Variant variants[] = {
        {"none", "none", 1.0f},
        {"bn(aV=1)", "threshold-dependent", 1.0f},
        {"tdBN(a=2)", "threshold-dependent", 2.0f},
    };

    std::cout << "\n[tdBN benchmark] variant     folds  mean_EER  mean_AUC\n";
    for (const auto& v : variants)
    {
        ThesisConfig cfg = make_fast_cfg();
        cfg.classifier.type = "dsnn";
        cfg.training.batch_normalization = v.bn;
        cfg.training.tdbn_alpha = v.alpha;

        auto result = run_classifier(view, fvs, std::string("synth-bench-") + v.name, cfg, &scorer);

        EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds) << v.name;
        EXPECT_FALSE(std::isnan(result.mean_eer)) << v.name;
        EXPECT_GE(result.mean_eer, 0.0);
        EXPECT_LE(result.mean_eer, 1.0);

        std::cout << "[tdBN benchmark] " << v.name << "\t" << result.outer_folds.size() << "\t"
                  << result.mean_eer << "\t" << result.mean_auc << "\n";
    }
}
