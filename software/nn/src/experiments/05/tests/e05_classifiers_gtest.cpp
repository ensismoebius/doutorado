// Unit tests for E05Classifiers: compute_aggregate_stats, run_classifier with
// synthetic data (2 classes, small feature dim, 2-fold CV).
// No SQLite, no real audio — everything synthetic.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../lib/include/E05Classifiers.hpp"
#include "../lib/include/E05Config.hpp"
#include "../lib/include/E05Dataset.hpp"
#include "statistics/eer_scorer.hpp"

using namespace e05;

// ─── compute_aggregate_stats ─────────────────────────────────────────────────

TEST(E05AggregateStats, EmptyFoldsNoOp)
{
    ClassificationResult r;
    compute_aggregate_stats(r);
    EXPECT_DOUBLE_EQ(r.mean_accuracy, 0.0);
    EXPECT_DOUBLE_EQ(r.mean_eer, 0.0);
    EXPECT_DOUBLE_EQ(r.std_accuracy, 0.0);
}

TEST(E05AggregateStats, SingleFold)
{
    ClassificationResult r;
    r.outer_folds.push_back({0, 0.8, 0.2, 0.0});
    compute_aggregate_stats(r);
    EXPECT_DOUBLE_EQ(r.mean_accuracy, 0.8);
    EXPECT_DOUBLE_EQ(r.mean_eer, 0.2);
    EXPECT_NEAR(r.std_accuracy, 0.0, 1e-12);
}

TEST(E05AggregateStats, TwoFoldsMean)
{
    ClassificationResult r;
    r.outer_folds.push_back({0, 0.6, 0.1, 0.0});
    r.outer_folds.push_back({1, 0.8, 0.3, 0.0});
    compute_aggregate_stats(r);
    EXPECT_NEAR(r.mean_accuracy, 0.7, 1e-12);
    EXPECT_NEAR(r.mean_eer, 0.2, 1e-12);
}

TEST(E05AggregateStats, TwoFoldsStd)
{
    ClassificationResult r;
    r.outer_folds.push_back({0, 0.6, 0.0, 0.0});
    r.outer_folds.push_back({1, 0.8, 0.0, 0.0});
    compute_aggregate_stats(r);
    // mean = 0.7; sample variance = ((0.6-0.7)^2 + (0.8-0.7)^2) / (2-1) = 0.02
    // sample SD = sqrt(0.02) ≈ 0.141421356 (audit M-3: sample SD, not population).
    EXPECT_NEAR(r.std_accuracy, 0.14142135623730953, 1e-12);
}

TEST(E05AggregateStats, AllSameMeanAndZeroStd)
{
    ClassificationResult r;
    for (int i = 0; i < 5; ++i)
        r.outer_folds.push_back({i, 0.9, 0.05, 0.0});
    compute_aggregate_stats(r);
    EXPECT_NEAR(r.mean_accuracy, 0.9, 1e-12);
    EXPECT_NEAR(r.std_accuracy, 0.0, 1e-12);
}

// ─── run_classifier (synthetic) ──────────────────────────────────────────────

namespace
{
// Build a minimal E05Config for a quick run: 2-fold CV, 5 epochs, batch=4.
E05Config make_fast_cfg()
{
    E05Config cfg;
    cfg.experiment.run_tag = "test";
    cfg.experiment.seed    = 42;

    cfg.dataset.modality    = "eeg";
    cfg.dataset.root        = "/dev/null"; // never accessed in unit test
    cfg.dataset.results_dir = "/tmp/";

    cfg.feature_extraction.strategy = "handcrafted";

    cfg.classifier.type      = "rnn";
    cfg.classifier.text_mode = "independent";

    cfg.training.epochs              = 5;
    cfg.training.learning_rate       = 1e-3;
    cfg.training.samples_per_batch   = 4;
    cfg.training.early_stop_patience = 0; // disabled
    cfg.training.k_folds             = 2;

    cfg.paraconsistent.enabled = false;
    return cfg;
}

// Build a synthetic E05DatasetView: n_subjects classes, samples_per_subject
// samples each.  Audio tensor is a 1×D row of zeros (non-empty so feature
// extraction doesn't fall back to the 256-zero placeholder in E05Dataset, but
// here we bypass extraction entirely and supply feature_vectors directly).
E05DatasetView make_view(int n_subjects, int samples_per_subject)
{
    E05DatasetView view;
    view.n_subjects = n_subjects;
    view.n_stimuli  = 1;

    int id = 1;
    for (int s = 0; s < n_subjects; ++s, ++id)
    {
        for (int k = 0; k < samples_per_subject; ++k)
        {
            E05Sample sam;
            sam.subject_id  = id;
            sam.stimulus    = k % 5; // vowel index 0-4
            sam.text_phrase = "a";
            // eeg/audio left default (0-dim scalar) — feature_vectors supplied externally
            view.samples.push_back(sam);
        }
    }
    return view;
}

// Build linearly separable feature vectors: class c gets feature vector [c, 0, …].
std::vector<std::vector<double>> make_features(const E05DatasetView& view,
    int feat_dim = 8)
{
    int id_idx = 0;
    int prev   = -1;
    std::vector<std::vector<double>> fvs;
    fvs.reserve(view.samples.size());
    for (const auto& s : view.samples)
    {
        if (s.subject_id != prev) { ++id_idx; prev = s.subject_id; }
        std::vector<double> fv(static_cast<size_t>(feat_dim), 0.0);
        fv[0] = static_cast<double>(id_idx);
        fvs.push_back(fv);
    }
    return fvs;
}
} // namespace

TEST(E05RunClassifier, ThrowsOnEmptyFeatures)
{
    auto view = make_view(2, 4);
    E05Config cfg = make_fast_cfg();
    EXPECT_THROW(run_classifier(view, {}, "test", cfg), std::invalid_argument);
}

TEST(E05RunClassifier, ThrowsOnSizeMismatch)
{
    auto view = make_view(2, 4);
    E05Config cfg = make_fast_cfg();
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
TEST(E05RunClassifier, ReturnsFoldCountMatchingKFolds)
{
    const int n_subjects = 4;
    const int sps        = 6;
    auto view            = make_view(n_subjects, sps);
    auto fvs             = make_features(view);
    E05Config cfg        = make_fast_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth", cfg, &scorer);

    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
}

TEST(E05RunClassifier, VerificationOnlyAccuracyIsNaN)
{
    // Verification-only protocol (audit C-1): closed-set accuracy is not reported
    // (speaker-disjoint folds), so mean_accuracy aggregates to NaN. The run must
    // still produce one result per outer fold.
    auto view     = make_view(4, 6);
    auto fvs      = make_features(view);
    E05Config cfg = make_fast_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth", cfg, &scorer);

    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
    EXPECT_TRUE(std::isnan(result.mean_accuracy));
}

TEST(E05RunClassifier, FeatureLabelPropagated)
{
    auto view     = make_view(4, 6);
    auto fvs      = make_features(view);
    E05Config cfg = make_fast_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "my-label", cfg, &scorer);
    EXPECT_EQ(result.feature_set_label, "my-label");
}

TEST(E05RunClassifier, FoldIndicesAreSequential)
{
    auto view     = make_view(4, 6);
    auto fvs      = make_features(view);
    E05Config cfg = make_fast_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth", cfg, &scorer);
    for (int i = 0; i < static_cast<int>(result.outer_folds.size()); ++i)
        EXPECT_EQ(result.outer_folds[static_cast<size_t>(i)].fold, i);
}

TEST(E05RunClassifier, DsnnPathRuns)
{
    auto view     = make_view(4, 6);
    auto fvs      = make_features(view);
    E05Config cfg = make_fast_cfg();
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
// throwing — exercises Adam::weight_decay and E05DsnnClassifier::add_firing_rate_grad.
TEST(E05RunClassifier, DsnnWithRegularizationRuns)
{
    auto view     = make_view(4, 6);
    auto fvs      = make_features(view);
    E05Config cfg = make_fast_cfg();
    cfg.classifier.type                  = "dsnn";
    cfg.training.weight_decay            = 1e-3f;
    cfg.training.firing_rate_reg_lambda  = 0.05f;
    cfg.training.firing_rate_min         = 0.05f;
    cfg.training.firing_rate_max         = 0.80f;
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth-dsnn-reg", cfg, &scorer);
    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
}

// Weight decay on the rnn path must also train cleanly (firing-rate reg is inert
// for the non-spiking ResNet classifier).
TEST(E05RunClassifier, RnnWithWeightDecayRuns)
{
    auto view     = make_view(4, 6);
    auto fvs      = make_features(view);
    E05Config cfg = make_fast_cfg();
    cfg.training.weight_decay = 1e-3f;
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth-rnn-wd", cfg, &scorer);
    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
}
