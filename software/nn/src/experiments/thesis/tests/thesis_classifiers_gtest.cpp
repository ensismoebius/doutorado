// Unit tests for ThesisClassifiers: compute_aggregate_stats, run_classifier with
// synthetic data (2 classes, small feature dim, 2-fold CV).
// No SQLite, no real audio — everything synthetic.

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../lib/include/IndexIntersection.hpp"
#include "../lib/include/ThesisClassifiers.hpp"
#include "../lib/include/ThesisConfig.hpp"
#include "../lib/include/ThesisDataset.hpp"
#include "ThesisTestConfig.hpp"
#include "statistics/ClassificationEERScorer.hpp"
#include "statistics/eer_scorer.hpp"

using namespace thesis;

// ─── intersect_indices ───────────────────────────────────────────────────────

// A fold's train indices are the other folds' buckets concatenated: ascending within a
// bucket, not across them. std::set_intersection on that input (the old implementation)
// returns {8, 9} here and drops 0, 1, 2, 4 and 5 without a word.
TEST(ThesisIndexIntersection, UnsortedInputKeepsEveryCommonIndex)
{
    const std::vector<std::size_t> concatenated_buckets = {8, 9, 0, 1, 2, 4, 5};
    const std::vector<std::size_t> text_split = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<std::size_t> expected = {0, 1, 2, 4, 5, 8, 9};
    EXPECT_EQ(intersect_indices(concatenated_buckets, text_split), expected);
    EXPECT_EQ(intersect_indices(text_split, concatenated_buckets), expected);
}

TEST(ThesisIndexIntersection, DisjointIsEmpty)
{
    EXPECT_TRUE(intersect_indices({5, 1, 3}, {4, 2, 0}).empty());
}

// ─── compute_aggregate_stats ─────────────────────────────────────────────────

namespace
{
// A trained fold: compute_aggregate_stats refuses a fold without a training history
// (or, for a network, without a measured energy).
// The one epoch carries the default (NaN) firing rate of a non-spiking classifier.
FoldResult fold(int i, double accuracy, double eer)
{
    FoldResult f{i, accuracy, eer, 0.0};
    f.history.emplace_back();
    f.energy = InferenceEnergy{}; // a network fold always measures its test pass
    return f;
}
} // namespace

// No folds used to be a no-op that left every statistic at 0, as if aggregated.
TEST(ThesisAggregateStats, EmptyFoldsThrow)
{
    ClassificationResult r;
    EXPECT_THROW(compute_aggregate_stats(r), std::logic_error);
}

// One fold has no SD; it used to be reported as 0.
TEST(ThesisAggregateStats, SingleFoldThrows)
{
    ClassificationResult r;
    r.outer_folds.push_back(fold(0, 0.8, 0.2));
    EXPECT_THROW(compute_aggregate_stats(r), std::logic_error);
}

// A metric missing in some folds used to be averaged over the rest.
TEST(ThesisAggregateStats, MetricNanInSomeFoldsThrows)
{
    ClassificationResult r;
    r.outer_folds.push_back(fold(0, 0.6, 0.1));
    r.outer_folds.push_back(fold(1, 0.8, std::numeric_limits<double>::quiet_NaN()));
    EXPECT_THROW(compute_aggregate_stats(r), std::logic_error);
}

// NaN in every fold is a metric not measured (closed-set under verification-only):
// mean, SD and CI are all NaN, never a spread of 0.
TEST(ThesisAggregateStats, MetricNanInEveryFoldIsNotApplicable)
{
    ClassificationResult r;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    r.outer_folds.push_back(fold(0, nan, 0.1));
    r.outer_folds.push_back(fold(1, nan, 0.3));
    compute_aggregate_stats(r);
    EXPECT_TRUE(std::isnan(r.mean_accuracy));
    EXPECT_TRUE(std::isnan(r.std_accuracy));
    EXPECT_TRUE(std::isnan(r.ci95_accuracy));
    EXPECT_NEAR(r.mean_eer, 0.2, 1e-12);
    EXPECT_TRUE(std::isnan(r.mean_spike_rate)); // ann: no firing rate in any fold
}

TEST(ThesisAggregateStats, FoldWithoutHistoryThrows)
{
    ClassificationResult r;
    r.outer_folds.push_back(fold(0, 0.6, 0.1));
    r.outer_folds.push_back({1, 0.8, 0.3, 0.0});
    EXPECT_THROW(compute_aggregate_stats(r), std::logic_error);
}

TEST(ThesisAggregateStats, TwoFoldsMean)
{
    ClassificationResult r;
    r.outer_folds.push_back(fold(0, 0.6, 0.1));
    r.outer_folds.push_back(fold(1, 0.8, 0.3));
    compute_aggregate_stats(r);
    EXPECT_NEAR(r.mean_accuracy, 0.7, 1e-12);
    EXPECT_NEAR(r.mean_eer, 0.2, 1e-12);
}

TEST(ThesisAggregateStats, TwoFoldsStd)
{
    ClassificationResult r;
    r.outer_folds.push_back(fold(0, 0.6, 0.0));
    r.outer_folds.push_back(fold(1, 0.8, 0.0));
    compute_aggregate_stats(r);
    // mean = 0.7; sample variance = ((0.6-0.7)^2 + (0.8-0.7)^2) / (2-1) = 0.02
    // sample SD = sqrt(0.02) ≈ 0.141421356 (audit M-3: sample SD, not population).
    EXPECT_NEAR(r.std_accuracy, 0.14142135623730953, 1e-12);
}

TEST(ThesisAggregateStats, AllSameMeanAndZeroStd)
{
    ClassificationResult r;
    for (int i = 0; i < 5; ++i) r.outer_folds.push_back(fold(i, 0.9, 0.05));
    compute_aggregate_stats(r);
    EXPECT_NEAR(r.mean_accuracy, 0.9, 1e-12);
    EXPECT_NEAR(r.std_accuracy, 0.0, 1e-12);
}

// ─── run_classifier (synthetic) ──────────────────────────────────────────────

namespace
{
// Build a minimal ThesisConfig for a quick run: 2-fold nested CV, 5 epochs, batch=4.
// Every key the classifier reads is declared — including the ones this test used to
// inherit from struct defaults (adam, nested_cv=true, standardize=true, and a layer_spec
// that was EMPTY and silently became 128 hidden / depth 2 in parse_layer_spec).
ThesisConfig make_fast_cfg()
{
    ThesisConfig cfg = test::phase00("eeg");
    test::set_handcrafted(cfg);
    cfg.dataset.root = "/dev/null"; // never accessed in unit test
    cfg.dataset.results_dir = "/tmp/";
    cfg.paraconsistent = ThesisConfig::Paraconsistent{};
    cfg.paraconsistent.enabled = false;

    test::set_classifier(cfg, "ann");
    cfg.classifier.layer_spec =
        std::vector<std::string>{"linear:128:relu", "residual:2", "linear:N_speakers:identity"};
    cfg.classifier.text_mode = "independent";

    test::set_training(cfg, 5, 1e-3f, 4);
    cfg.training.k_folds = 2;
    cfg.training.nested_cv = true;
    cfg.training.early_stop_patience = 0; // disabled
    cfg.validate();
    return cfg;
}

// The dsnn classifier reads its own keys: firing-rate regularization (off) and batch
// normalization (none). A test that turns either on declares the dependent keys.
ThesisConfig make_dsnn_cfg()
{
    ThesisConfig cfg = make_fast_cfg();
    cfg.classifier.type = "dsnn";
    cfg.training.firing_rate_reg_lambda = 0.0f;
    cfg.training.batch_normalization = "none";
    return cfg;
}

// Build a synthetic ThesisDatasetView: n_subjects classes, samples_per_subject
// samples each. Feature extraction is bypassed: feature_vectors are supplied directly.
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
            // Four phrases, so text_mode=independent can hold two out: with one phrase for
            // every sample its train half was empty, which the nested path used to hide by
            // ignoring the text split for training.
            static const char* const kPhrases[] = {"a", "e", "i", "o"};
            sam.text_phrase = kPhrases[k % 4];
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
    EXPECT_THROW(run_classifier(view, {}, "test", cfg, statistics::ClassificationEERScorer{}),
        std::invalid_argument);
}

TEST(ThesisRunClassifier, ThrowsOnSizeMismatch)
{
    auto view = make_view(2, 4);
    ThesisConfig cfg = make_fast_cfg();
    std::vector<std::vector<double>> fvs(3, std::vector<double>(8, 0.0));
    EXPECT_THROW(run_classifier(view, fvs, "test", cfg, statistics::ClassificationEERScorer{}),
        std::invalid_argument);
}

// GroupKFoldPolicy with k=2 on 8 subjects: each outer fold trains on 4 speakers, so
// each inner validation fold holds 2 — the fewest a verification score needs (genuine
// AND impostor trials). With the 4 subjects these tests used before, an inner fold held
// 1 speaker; its AUC was NaN and model selection quietly fell back to the first candidate.
TEST(ThesisRunClassifier, ReturnsFoldCountMatchingKFolds)
{
    const int n_subjects = 8;
    const int sps = 6;
    auto view = make_view(n_subjects, sps);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth", cfg, scorer);

    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
}

TEST(ThesisRunClassifier, VerificationOnlyAccuracyIsNaN)
{
    // Verification-only protocol (audit C-1): closed-set accuracy is not reported
    // (speaker-disjoint folds), so mean_accuracy aggregates to NaN. The run must
    // still produce one result per outer fold.
    auto view = make_view(8, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth", cfg, scorer);

    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
    EXPECT_TRUE(std::isnan(result.mean_accuracy));
}

TEST(ThesisRunClassifier, FeatureLabelPropagated)
{
    auto view = make_view(8, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "my-label", cfg, scorer);
    EXPECT_EQ(result.feature_set_label, "my-label");
}

TEST(ThesisRunClassifier, FoldIndicesAreSequential)
{
    auto view = make_view(8, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth", cfg, scorer);
    for (int i = 0; i < static_cast<int>(result.outer_folds.size()); ++i)
        EXPECT_EQ(result.outer_folds[static_cast<size_t>(i)].fold, i);
}

TEST(ThesisRunClassifier, DsnnPathRuns)
{
    auto view = make_view(8, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_dsnn_cfg();
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth-dsnn", cfg, scorer);
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
    auto view = make_view(8, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_dsnn_cfg();
    cfg.training.weight_decay = 1e-3f;
    cfg.training.firing_rate_reg_lambda = 0.05f;
    cfg.training.firing_rate_min = 0.05f;
    cfg.training.firing_rate_max = 0.80f;
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth-dsnn-reg", cfg, scorer);
    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
}

// Weight decay on the ann path must also train cleanly (firing-rate reg is inert
// for the non-spiking ResNet classifier).
TEST(ThesisRunClassifier, RnnWithWeightDecayRuns)
{
    auto view = make_view(8, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_fast_cfg();
    cfg.training.weight_decay = 1e-3f;
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth-ann-wd", cfg, scorer);
    EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds);
}

// tdBN enabled on the dsnn path must train and produce one result per outer fold
// without throwing — exercises the tdBN forward/backward wired into the DSNN.
TEST(ThesisRunClassifier, DsnnWithTdbnRuns)
{
    auto view = make_view(8, 6);
    auto fvs = make_features(view);
    ThesisConfig cfg = make_dsnn_cfg();
    cfg.training.batch_normalization = "threshold-dependent";
    cfg.training.tdbn_alpha = 1.0f;
    statistics::ClassificationEERScorer scorer;

    auto result = run_classifier(view, fvs, "synth-dsnn-tdbn", cfg, scorer);
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
    auto view = make_view(8, 6);
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
        ThesisConfig cfg = make_dsnn_cfg();
        cfg.training.batch_normalization = v.bn;
        if (std::string(v.bn) == "threshold-dependent") cfg.training.tdbn_alpha = v.alpha;
        cfg.validate();

        auto result = run_classifier(view, fvs, std::string("synth-bench-") + v.name, cfg, scorer);

        EXPECT_EQ(static_cast<int>(result.outer_folds.size()), cfg.training.k_folds) << v.name;
        EXPECT_FALSE(std::isnan(result.mean_eer)) << v.name;
        EXPECT_GE(result.mean_eer, 0.0);
        EXPECT_LE(result.mean_eer, 1.0);

        std::cout << "[tdBN benchmark] " << v.name << "\t" << result.outer_folds.size() << "\t"
                  << result.mean_eer << "\t" << result.mean_auc << "\n";
    }
}
