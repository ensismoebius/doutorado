/**
 * @file src/experiments/thesis/tests/thesis_per_fold_gtest.cpp
 * @brief The per-fold protocol (ThesisPerFold.hpp, .wiki/Experiments/Thesis-Per-Fold-Protocol.md).
 *
 * The property the protocol exists for is not "the code takes a row list" but this: nothing a
 * test speaker carries can change what its fold chooses. TestSpeakersCannotMoveTheChoices
 * replaces the voice of a fold's test speakers and requires the fold's extractor ranking and
 * every inner EER of its grid to stay bit-identical. A path that ranked or fitted on every row
 * would fail it.
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "../lib/include/IndexIntersection.hpp"
#include "../lib/include/ThesisOutput.hpp"
#include "../lib/include/ThesisPerFold.hpp"
#include "ThesisTestConfig.hpp"
#include "nlohmann/json.hpp"
#include "statistics/GenuineImpostorEERScorer.hpp"
#include "statistics/GroupKFoldPolicy.hpp"

using namespace thesis;

namespace
{

constexpr int kSpeakers = 8;
constexpr int kPerSpeaker = 6;

nn::Tensor voice(int n, double cycles_per_sample, std::uint32_t noise_seed)
{
    nn::Tensor t(static_cast<nn::Index>(n), 1);
    for (int i = 0; i < n; ++i)
        t.at(i, 0) = test::synthetic_value(i, cycles_per_sample, noise_seed);
    return t;
}

// 8 speakers x 6 utterances, phrases "a"/"e"; each speaker has its own pitch, each utterance a
// small offset, so the spectra separate the speakers.
ThesisDatasetView make_view()
{
    ThesisDatasetView v;
    v.voice_sample_rate = 44100.0;
    v.eeg_sample_rate = 1024.0;
    v.n_subjects = kSpeakers;
    v.n_stimuli = 2;
    for (int s = 1; s <= kSpeakers; ++s)
        for (int k = 0; k < kPerSpeaker; ++k)
        {
            ThesisSample x;
            const double hz = 150.0 + 60.0 * s + 4.0 * k;
            x.audio = voice(1024, hz / 44100.0, static_cast<std::uint32_t>(100 * s + k));
            x.subject_id = s;
            x.stimulus = 1 + k % 2;
            x.text_phrase = k % 2 == 0 ? "a" : "e";
            v.samples.push_back(std::move(x));
        }
    return v;
}

ThesisConfig::ExtractorCandidate pca_candidate(int components)
{
    ThesisConfig::ExtractorCandidate c;
    c.feature_extraction.strategy = "pca";
    c.feature_extraction.pca.n_components = components;
    return c;
}

ThesisConfig::ExtractorCandidate handcrafted_candidate()
{
    ThesisConfig tmp = test::phase00("voice");
    test::set_handcrafted(tmp);
    ThesisConfig::ExtractorCandidate c;
    c.feature_extraction = tmp.feature_extraction;
    return c;
}

auto results_dir() -> std::string
{
    return (std::filesystem::temp_directory_path() / "thesis_per_fold_gtest").string();
}

// A per-fold profile built in code: 2 outer folds (4 training speakers), 2 inner folds.
ThesisConfig per_fold_cfg(const std::string& type)
{
    ThesisConfig c = test::phase00("voice");
    c.dataset.results_dir = results_dir();
    c.paraconsistent = ThesisConfig::Paraconsistent{};
    c.paraconsistent.enabled = false;
    c.classifier.enabled = true;
    c.classifier.type = type;
    c.classifier.text_mode = "dependent";
    c.training.k_folds = 2;
    c.training.standardize_features = true;
    c.degradation.enabled = false;
    c.selection.enabled = true;
    c.selection.rank_modality = "voice";
    c.selection.extractor_candidates = {
        pca_candidate(2), pca_candidate(4), handcrafted_candidate()};
    if (type != "cosine")
    {
        c.training.epochs = 2;
        c.training.optimizer_type = "adam";
        c.training.gradient_clip_norm = 0.0f;
        c.training.samples_per_batch = 8;
        c.selection.inner_k = 2;
        c.selection.hidden_widths = std::vector<int>{8, 16};
        c.selection.residual_blocks = std::vector<int>{0};
        c.selection.learning_rates = std::vector<float>{1e-3f};
        c.selection.weight_decays = std::vector<float>{0.0f};
    }
    c.validate();
    return c;
}

std::set<int> speakers_of(const ThesisDatasetView& v, const std::vector<std::size_t>& rows)
{
    std::set<int> s;
    for (auto r : rows) s.insert(v.samples.at(r).subject_id);
    return s;
}

} // namespace

TEST(PerFoldRows, OuterFoldsAreTheFlatCvFolds)
{
    const auto view = make_view();
    const auto cfg = per_fold_cfg("cosine");
    const auto folds = outer_fold_rows(view.samples, cfg);

    // The fixed-configuration flat CV's split, rebuilt here from its own parts.
    std::vector<int> groups;
    groups.reserve(view.samples.size());
    for (const auto& s : view.samples) groups.push_back(s.subject_id);
    const statistics::GroupKFoldPolicy policy(2, true, cfg.experiment.seed);
    const auto splits = policy.make_splits(view.samples.size(), groups);
    auto ts = make_text_split(view.samples, "dependent", cfg.experiment.seed);
    ASSERT_EQ(folds.size(), splits.size());
    for (std::size_t f = 0; f < folds.size(); ++f)
    {
        EXPECT_EQ(folds[f].train, intersect_indices(splits[f].train_indices, ts.train_indices));
        EXPECT_EQ(folds[f].test, intersect_indices(splits[f].test_indices, ts.test_indices));
        const auto train_spk = speakers_of(view, folds[f].speaker_rows);
        for (int s : speakers_of(view, folds[f].test)) EXPECT_EQ(train_spk.count(s), 0u);
        EXPECT_EQ(train_spk.size(), static_cast<std::size_t>(kSpeakers / 2));
    }
}

TEST(PerFoldRows, InnerFoldsStayInsideTheTrainingSpeakers)
{
    const auto view = make_view();
    const auto cfg = per_fold_cfg("ann");
    const auto ts = make_text_split(view.samples, "dependent", cfg.experiment.seed);
    for (const auto& outer : outer_fold_rows(view.samples, cfg))
    {
        const auto inner = inner_fold_rows(view.samples, outer, cfg);
        ASSERT_EQ(inner.size(), 2u);
        std::set<int> validated;
        for (const auto& in : inner)
        {
            for (auto r : in.train)
                EXPECT_TRUE(std::binary_search(outer.train.begin(), outer.train.end(), r));
            for (auto r : in.validation)
            {
                EXPECT_TRUE(
                    std::binary_search(outer.speaker_rows.begin(), outer.speaker_rows.end(), r));
                EXPECT_NE(std::find(ts.test_indices.begin(), ts.test_indices.end(), r),
                    ts.test_indices.end());
            }
            const auto tr = speakers_of(view, in.train);
            for (int s : speakers_of(view, in.validation))
            {
                EXPECT_EQ(tr.count(s), 0u) << "speaker " << s << " trains and validates";
                validated.insert(s);
            }
        }
        EXPECT_EQ(validated, speakers_of(view, outer.speaker_rows))
            << "every speaker validates once";
    }
}

TEST(PerFoldGrid, OrderConfigAndTieBreaks)
{
    ThesisConfig::Selection s;
    s.hidden_widths = std::vector<int>{64, 128, 256};
    s.residual_blocks = std::vector<int>{0, 1, 2};
    s.learning_rates = std::vector<float>{1e-3f, 3e-4f};
    s.weight_decays = std::vector<float>{0.0f, 1e-4f};
    const auto grid = grid_points(s);
    ASSERT_EQ(grid.size(), 36u);
    EXPECT_EQ(grid[0].hidden_width, 64);
    EXPECT_EQ(grid[1].weight_decay, 1e-4f); // weight decay varies fastest
    EXPECT_EQ(grid[2].learning_rate, 3e-4f);
    EXPECT_EQ(grid[4].residual_blocks, 1);
    EXPECT_EQ(grid[12].hidden_width, 128);

    const auto c = grid_config(per_fold_cfg("ann"), grid[35]);
    EXPECT_EQ(c.classifier.layer_spec.get(),
        (std::vector<std::string>{"linear:256:relu", "residual:3", "linear:N_speakers:identity"}));
    EXPECT_EQ(c.training.learning_rate.get(), 3e-4f);
    EXPECT_EQ(c.training.weight_decay.get(), 1e-4f);

    std::vector<GridPoint> g(3);
    g[0].mean_inner_eer = 0.2;
    g[0].param_count = 50;
    g[1].mean_inner_eer = 0.1;
    g[1].param_count = 90;
    g[2].mean_inner_eer = 0.1;
    g[2].param_count = 40;
    EXPECT_EQ(grid_winner(g), 2u); // tie on EER -> fewer parameters
    g[2].param_count = 90;
    EXPECT_EQ(grid_winner(g), 1u); // full tie -> grid order
}

TEST(PerFoldExtractor, RankingReadsOnlyTheGivenRowsAndTiesGoFirst)
{
    const auto view = make_view();
    FeatureSet a{"a", {}};
    for (const auto& s : view.samples)
        a.vectors.push_back({static_cast<double>(s.subject_id), static_cast<double>(s.stimulus)});
    FeatureSet b = a;
    b.label = "b";
    const std::vector<std::size_t> rows = {0, 1, 2, 6, 7, 8, 12, 13, 14};
    const auto before = rank_extractors(view.samples, rows, {a, b});
    EXPECT_EQ(before[0].d_penalized, before[1].d_penalized);
    EXPECT_EQ(extractor_winner(before), 0u); // tie -> first in profile order
    b.vectors[40] = {1e6, -1e6};             // a row outside `rows`
    const auto after = rank_extractors(view.samples, rows, {a, b});
    EXPECT_EQ(after[1].d_penalized, before[1].d_penalized);
}

TEST(PerFoldRun, CosineChoosesAnExtractorPerFoldAndTunesNothing)
{
    const auto view = make_view();
    const auto cfg = per_fold_cfg("cosine");
    const statistics::GenuineImpostorEERScorer scorer(1);
    CandidateFeatureCache cache;
    const auto r = run_per_fold(view, cfg, scorer, nullptr, cache);
    EXPECT_EQ(r.feature_set_label, "per-fold-voice");
    EXPECT_FALSE(r.param_count.has_value());
    ASSERT_EQ(r.outer_folds.size(), 2u);
    for (const auto& f : r.outer_folds)
    {
        if (!f.selection.has_value())
        {
            GTEST_FAIL() << "an outer fold has no selection record";
        }
        EXPECT_EQ(f.selection->extractors.size(), 3u);
        EXPECT_TRUE(f.selection->grid.empty());
        EXPECT_EQ(f.selection->param_count, 0u);
        EXPECT_TRUE(std::isfinite(f.eer));
    }
    EXPECT_EQ(cache.clean.size(), 1u); // the handcrafted candidate, extracted once
    const auto again = run_per_fold(view, cfg, scorer, nullptr, cache);
    EXPECT_EQ(again.mean_eer, r.mean_eer);
}

TEST(PerFoldRun, TestSpeakersCannotMoveTheChoices)
{
    auto view = make_view();
    const auto cfg = per_fold_cfg("ann");
    const statistics::GenuineImpostorEERScorer scorer(1);
    CandidateFeatureCache cache_a;
    const auto a = run_per_fold(view, cfg, scorer, nullptr, cache_a);

    // Another voice for every row of fold 0's test speakers, every phrase.
    const auto folds = outer_fold_rows(view.samples, cfg);
    const auto test_spk = speakers_of(view, folds[0].test);
    // Distinct per utterance: these speakers train in fold 1, whose PCA needs real variance.
    for (std::size_t i = 0; i < view.samples.size(); ++i)
    {
        auto& s = view.samples[i];
        if (test_spk.count(s.subject_id) != 0)
            s.audio = voice(1024,
                (2000.0 + 37.0 * static_cast<double>(i)) / 44100.0,
                static_cast<std::uint32_t>(9000 + i));
    }
    CandidateFeatureCache cache_b;
    const auto b = run_per_fold(view, cfg, scorer, nullptr, cache_b);

    ASSERT_TRUE(a.outer_folds[0].selection.has_value());
    ASSERT_TRUE(b.outer_folds[0].selection.has_value());
    const FoldSelection sa = a.outer_folds[0].selection.value_or(FoldSelection{});
    const FoldSelection sb = b.outer_folds[0].selection.value_or(FoldSelection{});
    ASSERT_EQ(sa.extractors.size(), sb.extractors.size());
    for (std::size_t i = 0; i < sa.extractors.size(); ++i)
        EXPECT_EQ(sa.extractors[i].d_penalized, sb.extractors[i].d_penalized)
            << "candidate " << i << ": a test speaker moved the extractor ranking (leak)";
    ASSERT_EQ(sa.grid.size(), 2u);
    for (std::size_t g = 0; g < sa.grid.size(); ++g)
        EXPECT_EQ(sa.grid[g].inner_eer, sb.grid[g].inner_eer)
            << "grid point " << g << ": a test speaker moved the hyperparameter search (leak)";
    EXPECT_EQ(sa.extractor_index, sb.extractor_index);
    EXPECT_EQ(sa.grid_index, sb.grid_index);
    EXPECT_GT(sa.param_count, 0u);
    // ...and the test speakers are still scored: their EER is free to move.
    EXPECT_TRUE(std::isfinite(b.outer_folds[0].eer));
    std::filesystem::remove_all(results_dir());
}

TEST(PerFoldRun, DegradedTwinRefitsTheCleanExtractor)
{
    const auto view = make_view();
    ThesisConfig cfg = per_fold_cfg("cosine");
    // PCA only: the winner is a fitted extractor, so the degraded path refits it.
    cfg.selection.extractor_candidates = {pca_candidate(2), pca_candidate(4)};
    const statistics::GenuineImpostorEERScorer scorer(1);

    // A "degradation" that changes nothing must give the clean result exactly.
    CandidateFeatureCache c0;
    const auto clean = run_per_fold(view, cfg, scorer, nullptr, c0);
    const PerFoldDegradation same{&view, false, 1};
    CandidateFeatureCache c1;
    const auto twin = run_per_fold(view, cfg, scorer, &same, c1);
    ASSERT_EQ(clean.outer_folds.size(), twin.outer_folds.size());
    for (std::size_t f = 0; f < clean.outer_folds.size(); ++f)
        EXPECT_EQ(clean.outer_folds[f].eer, twin.outer_folds[f].eer);

    // A real change of every voice: the refit check passes (the model is the clean one) and
    // the choices, made on clean training rows, do not move.
    // One tone for everybody, noise drawn per utterance: no speaker is left in the voice.
    auto degraded = view;
    for (std::size_t i = 0; i < degraded.samples.size(); ++i)
        degraded.samples[i].audio =
            voice(1024, 3100.0 / 44100.0, static_cast<std::uint32_t>(7000 + i));
    const PerFoldDegradation real{&degraded, false, 1};
    CandidateFeatureCache c2;
    const auto hurt = run_per_fold(view, cfg, scorer, &real, c2);
    for (std::size_t f = 0; f < clean.outer_folds.size(); ++f)
    {
        const auto& h = hurt.outer_folds[f].selection;
        const auto& c = clean.outer_folds[f].selection;
        if (!h.has_value() || !c.has_value())
        {
            GTEST_FAIL() << "outer fold " << f << " has no selection record";
        }
        EXPECT_EQ(h->extractor_index, c->extractor_index);
    }
    EXPECT_GT(hurt.mean_eer, clean.mean_eer);
}

TEST(PerFoldRun, RefusesAProfileWithoutSelection)
{
    const auto view = make_view();
    ThesisConfig cfg = per_fold_cfg("cosine");
    cfg.selection.enabled = false;
    const statistics::GenuineImpostorEERScorer scorer(1);
    CandidateFeatureCache cache;
    EXPECT_THROW((void) run_per_fold(view, cfg, scorer, nullptr, cache), std::logic_error);
}

TEST(PerFoldRun, DegradedTwinRefitsTheCleanAutoencoder)
{
    // The spiking autoencoder with Poisson frames is the extractor most sensitive to how its
    // frames are seeded: the degraded path refits it and run_per_fold compares the training
    // rows bit for bit with the clean fit, throwing on any difference.
    const auto view = make_view();
    ThesisConfig cfg = per_fold_cfg("cosine");
    ThesisConfig tmp = test::phase00("voice");
    test::set_autoencoder(tmp,
        "snn-ae",
        {"linear:16:leaky", "linear:8:identity"},
        {"linear:16:leaky", "linear:output:identity"},
        "mse");
    test::set_snn(tmp, "poisson", 8, 0.2f, 0.0f, 0.0f, 0.0f);
    test::set_training(tmp, 2, 1e-3f, 8);
    ThesisConfig::ExtractorCandidate ae;
    ae.feature_extraction = tmp.feature_extraction;
    ae.training = tmp.training;
    cfg.selection.extractor_candidates = {ae};
    cfg.validate();

    auto degraded = view;
    for (std::size_t i = 0; i < degraded.samples.size(); ++i)
        degraded.samples[i].audio =
            voice(1024, 3100.0 / 44100.0, static_cast<std::uint32_t>(7000 + i));
    const statistics::GenuineImpostorEERScorer scorer(1);
    const PerFoldDegradation real{&degraded, false, 1};
    CandidateFeatureCache cache;
    ClassificationResult r;
    ASSERT_NO_THROW(r = run_per_fold(view, cfg, scorer, &real, cache));
    EXPECT_EQ(r.outer_folds.size(), 2u);
}

// ── Outputs (stage d) ──────────────────────────────────────────────────────────
namespace
{
std::vector<std::string> csv_lines(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("not written: " + path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(f, line);) lines.push_back(line);
    return lines;
}
} // namespace

TEST(PerFoldOutput, SummaryAndCsvsCarryEveryChoice)
{
    const auto view = make_view();
    ThesisConfig cfg = per_fold_cfg("ann");
    const std::string dir = results_dir() + "_out";
    cfg.dataset.results_dir = dir;
    cfg.experiment.run_tag = "pf";
    const statistics::GenuineImpostorEERScorer scorer(1);
    CandidateFeatureCache cache;
    const std::vector<ClassificationResult> results = {
        run_per_fold(view, cfg, scorer, nullptr, cache)};
    write_summary_json(dir, "pf", cfg, results, {}, kSpeakers, 2, view.samples.size(), 0u);
    write_selection_csvs(dir, "pf", results);

    nlohmann::json j;
    {
        std::ifstream f(dir + "/e05_pf_summary.json");
        ASSERT_TRUE(f.good());
        f >> j;
    }
    EXPECT_EQ(j["protocol"], "per-fold");
    EXPECT_FALSE(j.contains("strategy")) << "a per-fold run has no extractor of its own";
    const auto& r = j["results"][0];
    EXPECT_FALSE(r.contains("param_count")) << "each fold's winner has its own";
    ASSERT_EQ(r["fold_models"].size(), 2u);
    for (std::size_t f = 0; f < 2; ++f)
    {
        const auto& sel = r["fold_models"][f]["selection"];
        const FoldSelection& want = results[0].outer_folds[f].selection.value_or(FoldSelection{});
        EXPECT_EQ(sel["extractors"].size(), 3u);
        EXPECT_EQ(sel["extractor_index"], want.extractor_index);
        EXPECT_EQ(sel["extractor_label"], want.extractors.at(want.extractor_index).label);
        ASSERT_EQ(sel["grid"].size(), 2u);
        EXPECT_EQ(sel["grid_index"], want.grid_index);
        EXPECT_EQ(sel["grid"][0]["inner_eer"].size(), 2u);
        EXPECT_EQ(sel["param_count"], want.param_count);
        EXPECT_GT(sel["fold_ms"].get<double>(), 0.0);
    }

    // 2 folds x 3 candidates; 2 folds x 2 points x 2 inner folds; one "chosen" per fold.
    const auto ex = csv_lines(dir + "/e05_pf_selection_extractors.csv");
    const auto grid = csv_lines(dir + "/e05_pf_selection_grid.csv");
    EXPECT_EQ(ex.size(), 1u + 6u);
    EXPECT_EQ(grid.size(), 1u + 8u);
    const auto chosen = [](const std::vector<std::string>& lines)
    {
        return std::count_if(
            lines.begin() + 1, lines.end(), [](const std::string& l) { return l.back() == '1'; });
    };
    EXPECT_EQ(chosen(ex), 2);
    EXPECT_EQ(chosen(grid), 4); // the winner's row of each inner fold
    std::filesystem::remove_all(dir);
}

TEST(PerFoldOutput, FixedRunKeepsItsStrategyAndHasNoSelectionFiles)
{
    ThesisConfig cfg = test::phase00("voice");
    test::set_handcrafted(cfg);
    cfg.validate();
    const std::string dir = results_dir() + "_fixed";
    write_summary_json(dir, "fx", cfg, {}, {}, kSpeakers, 2, 48, 0u);
    nlohmann::json j;
    {
        std::ifstream f(dir + "/e05_fx_summary.json");
        ASSERT_TRUE(f.good());
        f >> j;
    }
    EXPECT_EQ(j["protocol"], "fixed");
    EXPECT_EQ(j["strategy"], "handcrafted");

    ClassificationResult r;
    r.outer_folds.resize(2);
    EXPECT_THROW(write_selection_csvs(dir, "fx", {r}), std::logic_error) << "no selection";
    r.outer_folds[1].selection = FoldSelection{};
    EXPECT_THROW(write_selection_csvs(dir, "fx", {r}), std::logic_error) << "a mix";
    std::filesystem::remove_all(dir);
}

// ── On-disk feature cache (FeatureDiskCache.hpp) ───────────────────────────────────
namespace
{
auto cache_dir(const std::string& name) -> std::string
{
    const auto d = std::filesystem::temp_directory_path() / ("thesis_fc_gtest_" + name);
    std::filesystem::remove_all(d);
    return d.string();
}

FeatureSet small_set(double base)
{
    return FeatureSet{"hc", {{base, 1.0}, {base + 2.0, 3.0}, {base + 4.0, 5.0}}};
}
} // namespace

TEST(FeatureDiskCache, FingerprintSeesEveryValueAndLabel)
{
    const auto view = make_view();
    const auto h = view_fingerprint(view);
    EXPECT_EQ(view_fingerprint(make_view()), h);
    auto a = view;
    a.samples[17].audio.at(500, 0) += 1e-6f;
    EXPECT_NE(view_fingerprint(a), h) << "one audio value";
    auto b = view;
    b.samples[3].text_phrase = "i";
    EXPECT_NE(view_fingerprint(b), h) << "a phrase";
    auto c = view;
    c.samples[3].subject_id = 99;
    EXPECT_NE(view_fingerprint(c), h) << "a speaker";
    auto d = view;
    d.voice_sample_rate = 16000.0;
    EXPECT_NE(view_fingerprint(d), h) << "the rate";
}

TEST(FeatureDiskCache, WritesOnceThenReadsTheSameBits)
{
    const auto dir = cache_dir("roundtrip");
    const FeatureDiskCache w(dir, "rev1", FeatureDiskCache::Mode::Write);
    int computed = 0;
    const auto compute = [&]()
    {
        ++computed;
        return small_set(0.1);
    };
    const auto first = w.load_or_compute("k", 3, compute);
    const auto second = w.load_or_compute("k", 3, compute);
    EXPECT_EQ(computed, 1);
    EXPECT_EQ(first.vectors, second.vectors);
    EXPECT_EQ(first.label, second.label);

    const FeatureDiskCache r(dir, "rev1", FeatureDiskCache::Mode::Read);
    EXPECT_EQ(r.load_or_compute("k", 3, compute).vectors, first.vectors);
    EXPECT_EQ(computed, 1) << "read mode reads";
    EXPECT_THROW((void) r.load_or_compute("other", 3, compute), std::runtime_error)
        << "a miss in read mode stops the run";
    EXPECT_THROW((void) r.load_or_compute("k", 4, compute), std::runtime_error) << "another height";

    // Another code revision is another key: nothing is reused across deploys.
    const FeatureDiskCache r2(dir, "rev2", FeatureDiskCache::Mode::Read);
    EXPECT_THROW((void) r2.load_or_compute("k", 3, compute), std::runtime_error);
    EXPECT_NE(r2.path_for("k"), r.path_for("k"));
    std::filesystem::remove_all(dir);
}

TEST(FeatureDiskCache, RefusesACorruptOrForeignFile)
{
    const auto dir = cache_dir("corrupt");
    const FeatureDiskCache w(dir, "rev1", FeatureDiskCache::Mode::Write);
    (void) w.load_or_compute("k", 3, []() { return small_set(0.1); });
    const auto path = w.path_for("k");

    // Another key's file under this key's name (a collision): refused, not read.
    std::filesystem::copy_file(path, w.path_for("j"));
    EXPECT_THROW(
        (void) w.load_or_compute("j", 3, []() { return small_set(9.0); }), std::runtime_error);

    // A truncated body.
    const auto size = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, size - 12);
    EXPECT_THROW(
        (void) w.load_or_compute("k", 3, []() { return small_set(0.1); }), std::runtime_error);
    std::filesystem::remove_all(dir);
}

TEST(FeatureDiskCache, EnvironmentIsStrict)
{
    const auto dir = cache_dir("env");
    ::unsetenv("THESIS_FEATURE_CACHE");
    ::unsetenv("THESIS_FEATURE_CACHE_REVISION");
    ::unsetenv("THESIS_FEATURE_CACHE_MODE");
    EXPECT_FALSE(FeatureDiskCache::from_env().has_value());
    ::setenv("THESIS_FEATURE_CACHE", dir.c_str(), 1);
    EXPECT_THROW((void) FeatureDiskCache::from_env(), std::invalid_argument) << "no revision";
    ::setenv("THESIS_FEATURE_CACHE_REVISION", "rev1", 1);
    EXPECT_THROW((void) FeatureDiskCache::from_env(), std::invalid_argument) << "no mode";
    ::setenv("THESIS_FEATURE_CACHE_MODE", "readwrite", 1);
    EXPECT_THROW((void) FeatureDiskCache::from_env(), std::invalid_argument) << "bad mode";
    ::setenv("THESIS_FEATURE_CACHE_MODE", "read", 1);
    const auto c = FeatureDiskCache::from_env();
    if (!c.has_value())
    {
        GTEST_FAIL() << "a complete environment gave no cache";
    }
    EXPECT_EQ(c->mode(), FeatureDiskCache::Mode::Read);
    ::unsetenv("THESIS_FEATURE_CACHE");
    ::unsetenv("THESIS_FEATURE_CACHE_REVISION");
    ::unsetenv("THESIS_FEATURE_CACHE_MODE");
    std::filesystem::remove_all(dir);
}

TEST(FeatureDiskCache, PerFoldRunIsTheSameWithAndWithoutIt)
{
    const auto view = make_view();
    const auto cfg = per_fold_cfg("cosine"); // 2 PCA + 1 handcrafted candidate
    const statistics::GenuineImpostorEERScorer scorer(1);
    CandidateFeatureCache plain;
    const auto without = run_per_fold(view, cfg, scorer, nullptr, plain);

    const auto dir = cache_dir("perfold");
    // The warm step writes the handcrafted candidate, and only that one.
    const FeatureDiskCache w(dir, "rev1", FeatureDiskCache::Mode::Write);
    CandidateFeatureCache warm;
    warm.disk = &w;
    EXPECT_EQ(warm_feature_cache(view, cfg, warm), 1u);

    // A new process (new in-memory cache) in read mode: same choices, same numbers.
    const FeatureDiskCache r(dir, "rev1", FeatureDiskCache::Mode::Read);
    CandidateFeatureCache reader;
    reader.disk = &r;
    const auto with = run_per_fold(view, cfg, scorer, nullptr, reader);
    ASSERT_EQ(with.outer_folds.size(), without.outer_folds.size());
    for (std::size_t f = 0; f < with.outer_folds.size(); ++f)
    {
        EXPECT_EQ(with.outer_folds[f].eer, without.outer_folds[f].eer);
        const auto a = with.outer_folds[f].selection.value_or(FoldSelection{});
        const auto b = without.outer_folds[f].selection.value_or(FoldSelection{});
        ASSERT_EQ(a.extractors.size(), b.extractors.size());
        for (std::size_t i = 0; i < a.extractors.size(); ++i)
            EXPECT_EQ(a.extractors[i].d_penalized, b.extractors[i].d_penalized);
    }

    // Read mode cannot warm, and another view is a miss.
    CandidateFeatureCache refused;
    refused.disk = &r;
    EXPECT_THROW((void) warm_feature_cache(view, cfg, refused), std::invalid_argument);
    auto other = view;
    other.samples[0].audio.at(10, 0) += 0.5f;
    CandidateFeatureCache miss;
    miss.disk = &r;
    EXPECT_THROW((void) run_per_fold(other, cfg, scorer, nullptr, miss), std::runtime_error);
    std::filesystem::remove_all(dir);
}
