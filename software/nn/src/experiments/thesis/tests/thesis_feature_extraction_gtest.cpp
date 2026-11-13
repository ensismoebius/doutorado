// Unit tests for ThesisFeatureExtraction: scalar descriptors and extract_handcrafted.
// No SQLite or ProgressManager dependency — all inputs are synthetic.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "../lib/include/ThesisConfig.hpp"
#include "../lib/include/ThesisFeatureExtraction.hpp"
#include "ThesisTestConfig.hpp"

using namespace thesis;

// ─── compute_energy ──────────────────────────────────────────────────────────

TEST(ThesisEnergy, EmptyReturnsZero)
{
    EXPECT_DOUBLE_EQ(compute_energy({}), 0.0);
}

TEST(ThesisEnergy, ConstantSignal)
{
    std::vector<double> sig(8, 3.0);
    EXPECT_DOUBLE_EQ(compute_energy(sig), 8.0 * 9.0);
}

TEST(ThesisEnergy, UnitImpulse)
{
    std::vector<double> sig(16, 0.0);
    sig[0] = 1.0;
    EXPECT_DOUBLE_EQ(compute_energy(sig), 1.0);
}

// ─── compute_zcr ─────────────────────────────────────────────────────────────

// Undefined descriptors throw (they returned 0, or NaN that the caller turned into 0,
// until 2026-11-12 — indistinguishable from a measured 0).
TEST(ThesisZCR, EmptyOrSingleIsUndefined)
{
    EXPECT_THROW((void) compute_zcr({}), std::domain_error);
    EXPECT_THROW((void) compute_zcr({1.0}), std::domain_error);
}

TEST(ThesisZCR, AlternatingSignal)
{
    // +-+-+- … → every consecutive pair crosses zero
    std::vector<double> sig = {1.0, -1.0, 1.0, -1.0, 1.0};
    // 4 crossings / 4 intervals = 1.0
    EXPECT_DOUBLE_EQ(compute_zcr(sig), 1.0);
}

TEST(ThesisZCR, ConstantPositiveNoCrossings)
{
    std::vector<double> sig(10, 1.0);
    EXPECT_DOUBLE_EQ(compute_zcr(sig), 0.0);
}

TEST(ThesisZCR, RateInRange)
{
    std::vector<double> sig(100);
    for (size_t i = 0; i < sig.size(); ++i) sig[i] = (i % 5 == 0) ? -1.0 : 1.0;
    double zcr = compute_zcr(sig);
    EXPECT_GE(zcr, 0.0);
    EXPECT_LE(zcr, 1.0);
}

// ─── compute_entropy ─────────────────────────────────────────────────────────

TEST(ThesisEntropy, EmptyIsUndefined)
{
    EXPECT_THROW((void) compute_entropy({}), std::domain_error);
}

TEST(ThesisEntropy, AllZeroesIsUndefined)
{
    EXPECT_THROW((void) compute_entropy({0.0, 0.0, 0.0}), std::domain_error);
}

TEST(ThesisEntropy, UniformSignalMaxEntropy)
{
    // |v|/total uniform → each p = 1/n → entropy = log2(n)
    std::vector<double> sig(8, 1.0);
    double ent = compute_entropy(sig);
    EXPECT_NEAR(ent, std::log2(8.0), 1e-10);
}

TEST(ThesisEntropy, ImpulseLowEntropy)
{
    // All energy in one sample → entropy ≈ 0
    std::vector<double> sig(16, 0.0);
    sig[0] = 1.0;
    EXPECT_NEAR(compute_entropy(sig), 0.0, 1e-10);
}

// ─── compute_teager ──────────────────────────────────────────────────────────

TEST(ThesisTeager, TooShortIsUndefined)
{
    EXPECT_THROW((void) compute_teager({1.0, 2.0}), std::domain_error);
}

TEST(ThesisTeager, ConstantSignalZero)
{
    // Teager of constant c: c^2 - c*c = 0
    std::vector<double> sig(10, 2.0);
    EXPECT_NEAR(compute_teager(sig), 0.0, 1e-12);
}

TEST(ThesisTeager, PureSinePositive)
{
    // Pure sine has positive mean Teager energy
    const int N = 64;
    std::vector<double> sig(N);
    for (int i = 0; i < N; ++i) sig[i] = std::sin(2.0 * M_PI * 4.0 * i / N);
    EXPECT_GT(compute_teager(sig), 0.0);
}

// ─── compute_jitter ──────────────────────────────────────────────────────────

TEST(ThesisJitter, TooFewPeaksIsUndefined)
{
    // Flat signal → no peaks
    std::vector<double> sig(32, 0.5);
    EXPECT_THROW((void) compute_jitter(sig, 44100.0), std::domain_error);
}

TEST(ThesisJitter, RegularSineNearZeroJitter)
{
    // Regular 100 Hz sine at 44100 Hz → very consistent periods → low jitter
    const int N = 4096;
    const double freq = 100.0;
    const double sr = 44100.0;
    std::vector<double> sig(N);
    for (int i = 0; i < N; ++i) sig[i] = std::sin(2.0 * M_PI * freq * i / sr);

    EXPECT_LT(compute_jitter(sig, sr), 0.05); // < 5% jitter for regular sine
}

// ─── compute_shimmer ─────────────────────────────────────────────────────────

TEST(ThesisShimmer, TooFewPeaksIsUndefined)
{
    std::vector<double> sig(32, 0.5);
    EXPECT_THROW((void) compute_shimmer(sig, 44100.0), std::domain_error);
}

TEST(ThesisShimmer, RegularSineNearZeroShimmer)
{
    const int N = 4096;
    const double freq = 100.0;
    const double sr = 44100.0;
    std::vector<double> sig(N);
    for (int i = 0; i < N; ++i) sig[i] = std::sin(2.0 * M_PI * freq * i / sr);

    EXPECT_LT(compute_shimmer(sig, sr), 0.05);
}

// ─── extract_handcrafted ─────────────────────────────────────────────────────

namespace
{
ThesisConfig::HandcraftedConfig make_hc_cfg(
    const std::vector<std::string>& descs, int dtwpt_level = 2)
{
    ThesisConfig::HandcraftedConfig cfg;
    cfg.transform = "dtwpt";
    cfg.descriptors = descs;
    cfg.dtwpt_level = dtwpt_level;
    cfg.scale = "mel";
    cfg.wavelet = "daub4";
    cfg.cepstral = false;
    return cfg;
}

// Power-of-2 length sine signal.
std::vector<double> make_sine(int N = 256, double freq = 10.0, double sr = 44100.0)
{
    std::vector<double> s(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) s[static_cast<size_t>(i)] = std::sin(2.0 * M_PI * freq * i / sr);
    return s;
}
} // namespace

TEST(ThesisHandcrafteExtract, EnergyOnly_DimMatchesBands)
{
    auto cfg = make_hc_cfg({"energy"}, 2);
    auto sig = make_sine();
    auto fv = extract_handcrafted(sig, cfg, 44100.0);
    // level-2 DTWPT → 4 sub-bands; at 44100 Hz MEL scale all 4 map to distinct bins
    EXPECT_EQ(fv.size(), 4u);
}

TEST(ThesisHandcrafteExtract, MultipleDescriptorsDimAdditive)
{
    auto cfg = make_hc_cfg({"energy", "zcr", "entropy"}, 2);
    auto sig = make_sine();
    auto fv = extract_handcrafted(sig, cfg, 44100.0);
    // 4 MEL bins × 3 descriptors = 12
    EXPECT_EQ(fv.size(), 12u);
}

TEST(ThesisHandcrafteExtract, AllDescriptors_FiniteOnABroadbandSignal)
{
    // Seeded white noise: every sub-band carries enough peaks for jitter and shimmer.
    auto cfg = make_hc_cfg({"energy", "zcr", "entropy", "teager", "jitter", "shimmer"}, 2);
    std::mt19937 rng(7u);
    std::normal_distribution<double> noise(0.0, 1.0);
    std::vector<double> sig(4096);
    for (double& x : sig) x = noise(rng);
    auto fv = extract_handcrafted(sig, cfg, 44100.0);
    // 4 MEL bins × 6 descriptors = 24
    EXPECT_EQ(fv.size(), 24u);
    for (size_t i = 0; i < fv.size(); ++i) EXPECT_TRUE(std::isfinite(fv[i])) << "index " << i;
}

TEST(ThesisHandcrafteExtract, UndefinedJitterThrowsInsteadOfZero)
{
    // 256 samples of a 100 Hz sine at 44.1 kHz is 0.58 of a period: no band has 3 peaks.
    // The extractor used to write jitter = shimmer = 0 here.
    auto cfg = make_hc_cfg({"energy", "zcr", "entropy", "teager", "jitter", "shimmer"}, 2);
    EXPECT_THROW((void) extract_handcrafted(make_sine(256, 100.0, 44100.0), cfg, 44100.0),
        std::domain_error);
}

TEST(ThesisHandcrafteExtract, Level3GivesMoreBands)
{
    auto cfg2 = make_hc_cfg({"energy"}, 2);
    auto cfg3 = make_hc_cfg({"energy"}, 3);
    auto sig = make_sine(512);
    EXPECT_LT(extract_handcrafted(sig, cfg2, 44100.0).size(),
        extract_handcrafted(sig, cfg3, 44100.0).size());
}

TEST(ThesisHandcrafteExtract, EnergyValuesNonNegative)
{
    auto cfg = make_hc_cfg({"energy"}, 2);
    auto sig = make_sine();
    for (double v : extract_handcrafted(sig, cfg, 44100.0)) EXPECT_GE(v, 0.0);
}

// ─── Category 2 cepstral (log + DCT) ─────────────────────────────────────────

TEST(ThesisCepstral, CepstralDiffersFromRawEnergy)
{
    // Category 1 (energy) vs Category 2 (cepstral): same dimension (one value per
    // band), different values.
    auto sig = make_sine(256, 100.0, 44100.0);
    auto cfg1 = make_hc_cfg({"energy"}, 2);
    auto cfg2 = make_hc_cfg({"energy"}, 2);
    cfg2.cepstral = true;
    auto f1 = extract_handcrafted(sig, cfg1, 44100.0);
    auto f2 = extract_handcrafted(sig, cfg2, 44100.0);
    ASSERT_EQ(f1.size(), f2.size());
    EXPECT_NE(f1, f2);
    for (double v : f2) EXPECT_TRUE(std::isfinite(v));
}

TEST(ThesisCepstral, EnergyReplacedOtherDescriptorsAppended)
{
    // scale=mel, level 2 → 4 bands. cepstral (4 coeffs, energy subsumed) + zcr(4).
    auto sig = make_sine(256, 100.0, 44100.0);
    auto cfg = make_hc_cfg({"energy", "zcr"}, 2);
    cfg.cepstral = true;
    auto f = extract_handcrafted(sig, cfg, 44100.0);
    EXPECT_EQ(f.size(), 8u);
}

// ─── pre-emphasis ────────────────────────────────────────────────────────────

TEST(ThesisPreEmphasis, MatchesThesisWorkedExample)
{
    // Thesis §Pré-ênfase: x=[1.0,0.9,0.6], alpha=0.97 → [1.0,-0.07,-0.273].
    std::vector<double> sig = {1.0, 0.9, 0.6};
    apply_preemphasis(sig, 0.97);
    ASSERT_EQ(sig.size(), 3u);
    EXPECT_NEAR(sig[0], 1.0, 1e-12);    // y[0] unchanged
    EXPECT_NEAR(sig[1], -0.07, 1e-12);  // 0.9 - 0.97*1.0
    EXPECT_NEAR(sig[2], -0.273, 1e-12); // 0.6 - 0.97*0.9
}

TEST(ThesisPreEmphasis, UsesOriginalNotFilteredPredecessor)
{
    // Back-to-front order: y[2] must use the ORIGINAL x[1], not the filtered one.
    std::vector<double> sig = {2.0, 2.0, 2.0};
    apply_preemphasis(sig, 0.5);
    EXPECT_NEAR(sig[0], 2.0, 1e-12);             // unchanged
    EXPECT_NEAR(sig[1], 2.0 - 0.5 * 2.0, 1e-12); // 1.0
    EXPECT_NEAR(sig[2], 2.0 - 0.5 * 2.0, 1e-12); // 1.0 (uses original x[1]=2.0)
}

TEST(ThesisPreEmphasis, AlphaZeroIsIdentity)
{
    std::vector<double> sig = {0.3, -0.5, 0.8, 1.0};
    const std::vector<double> orig = sig;
    apply_preemphasis(sig, 0.0);
    EXPECT_EQ(sig, orig);
}

TEST(ThesisPreEmphasis, ShortSignalsUnchanged)
{
    std::vector<double> empty;
    apply_preemphasis(empty, 0.97);
    EXPECT_TRUE(empty.empty());

    std::vector<double> one = {5.0};
    apply_preemphasis(one, 0.97);
    ASSERT_EQ(one.size(), 1u);
    EXPECT_DOUBLE_EQ(one[0], 5.0); // y[0] never touched
}

// ─── wavelet axis ────────────────────────────────────────────────────────────

TEST(ThesisWavelet, AllTraitWaveletsExtractFinite)
{
    // Every wavelet with coefficient traits in Types.hpp must run end-to-end and
    // produce finite features. Longer filters need a longer signal than their
    // support, so use 512 samples.
    const std::vector<std::string> wavelets = {"haar",
        "daub4",
        "daub6",
        "daub8",
        "daub10",
        "daub12",
        "daub14",
        "daub16",
        "daub18",
        "daub20",
        "daub22",
        "daub24",
        "daub26",
        "daub28",
        "daub30",
        "daub32",
        "daub34",
        "daub36",
        "daub38",
        "daub40",
        "daub42",
        "daub44",
        "daub46"};

    auto sig = make_sine(512, 100.0, 44100.0);
    for (const auto& w : wavelets)
    {
        auto cfg = make_hc_cfg({"energy", "entropy"}, 3);
        cfg.wavelet = w;
        auto fv = extract_handcrafted(sig, cfg, 44100.0);
        EXPECT_FALSE(fv.empty()) << "empty feature vector for wavelet " << w;
        for (double v : fv) EXPECT_TRUE(std::isfinite(v)) << "non-finite feature for wavelet " << w;
    }
}

TEST(ThesisWavelet, UnknownWaveletThrows)
{
    auto cfg = make_hc_cfg({"energy"}, 2);
    cfg.wavelet = "not-a-wavelet";
    auto sig = make_sine();
    EXPECT_THROW(extract_handcrafted(sig, cfg, 44100.0), std::invalid_argument);
}

TEST(ThesisWavelet, DifferentWaveletsDifferentFeatures)
{
    // Haar (2-tap) and Daub20 (20-tap) must not yield identical decompositions.
    auto sig = make_sine(512, 100.0, 44100.0);
    auto cfg_haar = make_hc_cfg({"energy"}, 3);
    cfg_haar.wavelet = "haar";
    auto cfg_d20 = make_hc_cfg({"energy"}, 3);
    cfg_d20.wavelet = "daub20";
    EXPECT_NE(
        extract_handcrafted(sig, cfg_haar, 44100.0), extract_handcrafted(sig, cfg_d20, 44100.0));
}

// ─── voice/eeg/fused extraction via extract_features (audit C12) ──────────────

namespace
{
nn::Tensor make_column_tensor(int n, double freq, double sr)
{
    nn::Tensor t(static_cast<nn::Index>(n), 1);
    for (int i = 0; i < n; ++i)
        t.at(i, 0) = test::synthetic_value(i, freq / sr, static_cast<std::uint32_t>(freq));
    return t;
}

// A minimal two-sample view whose samples each carry distinct audio + EEG.
ThesisDatasetView make_paired_view()
{
    ThesisDatasetView view;
    for (int s = 0; s < 2; ++s)
    {
        ThesisSample sample;
        sample.audio = make_column_tensor(256, 100.0 + s, 44100.0);
        sample.eeg = make_column_tensor(256, 10.0 + s, 1024.0);
        sample.subject_id = s;
        sample.stimulus = s;
        sample.text_phrase = (s == 0) ? "a" : "e";
        view.samples.push_back(std::move(sample));
    }
    view.n_subjects = 2;
    view.n_stimuli = 2;
    return view;
}

ThesisConfig::FeatureExtraction make_hc_fe(const std::vector<std::string>& descs, int level = 2)
{
    ThesisConfig::FeatureExtraction fe;
    fe.strategy = "handcrafted";
    fe.handcrafted = make_hc_cfg(descs, level);
    return fe;
}

// Handcrafted extraction reads only the strategy and the modality (+ fusion_mode when
// fused): no training block, no seed.
std::vector<FeatureSet> run_hc(const ThesisDatasetView& view,
    const ThesisConfig::FeatureExtraction& fe,
    const std::string& modality,
    const std::string& fusion_mode = "")
{
    ThesisConfig cfg;
    cfg.dataset.modality = modality;
    if (!fusion_mode.empty()) cfg.dataset.fusion_mode = fusion_mode;
    cfg.feature_extraction = fe;
    return extract_features(view, cfg, nullptr);
}
} // namespace

TEST(ThesisFusion, LateFusionDimIsVoicePlusEeg)
{
    auto view = make_paired_view();
    auto fe = make_hc_fe({"energy"});
    auto voice = run_hc(view, fe, "voice");
    auto eeg = run_hc(view, fe, "eeg");
    auto late = run_hc(view, fe, "fused", "late");

    ASSERT_EQ(voice.size(), 1u);
    ASSERT_EQ(eeg.size(), 1u);
    ASSERT_EQ(late.size(), 1u);
    ASSERT_FALSE(late[0].vectors.empty());

    // Late fusion concatenates the per-signal feature vectors sample-by-sample.
    EXPECT_EQ(late[0].vectors[0].size(), voice[0].vectors[0].size() + eeg[0].vectors[0].size());
    // Concatenation order is voice-part then eeg-part.
    for (size_t i = 0; i < voice[0].vectors[0].size(); ++i)
        EXPECT_DOUBLE_EQ(late[0].vectors[0][i], voice[0].vectors[0][i]);
}

TEST(ThesisFusion, EarlyFusionSingleExtractionDiffersFromLate)
{
    auto view = make_paired_view();
    auto fe = make_hc_fe({"energy"});
    auto early = run_hc(view, fe, "fused", "early");
    auto late = run_hc(view, fe, "fused", "late");

    ASSERT_EQ(early.size(), 1u);
    ASSERT_EQ(late.size(), 1u);
    // Early fuses raw signals (voice++eeg = 512 samples → padded to 512) into one
    // DTWPT pass; late runs two independent passes. Dimensions must not coincide
    // with the late-fusion sum in general, and labels must distinguish the modes.
    EXPECT_NE(early[0].label, late[0].label);
    EXPECT_NE(early[0].label.find("fused-early"), std::string::npos);
    EXPECT_NE(late[0].label.find("fused-late"), std::string::npos);
}

TEST(ThesisFusion, UnknownFusionModeThrows)
{
    auto view = make_paired_view();
    auto fe = make_hc_fe({"energy"});
    EXPECT_THROW(run_hc(view, fe, "fused", "bogus"), std::invalid_argument);
}

// ─── SNN-AE temporal encoding (poisson/latency rate coding) ───────────────────
// A LIF autoencoder only carries information through spikes, so each pooled
// sample is expanded into `time_steps` spike frames and the feature is the mean
// latent spike-rate. Without this the SNN latent collapses to (near) all-zeros
// — these tests guard the data-presentation contract.

namespace
{
// Several samples with distinct audio frequencies so their latents should
// differ. Distinct per-sample content is what makes a non-degenerate latent
// observable.
ThesisDatasetView make_multi_view(int n_samples)
{
    ThesisDatasetView view;
    for (int s = 0; s < n_samples; ++s)
    {
        ThesisSample sample;
        sample.audio = make_column_tensor(1024, 80.0 + 25.0 * s, 44100.0);
        sample.eeg = make_column_tensor(1024, 8.0 + 3.0 * s, 1024.0);
        sample.subject_id = s % 2;
        sample.stimulus = s;
        sample.text_phrase = (s % 2 == 0) ? "a" : "e";
        view.samples.push_back(std::move(sample));
    }
    view.n_subjects = 2;
    view.n_stimuli = n_samples;
    return view;
}

// The keys the old struct defaults supplied are declared at those values: mse, and for the
// SNN-AE a 0.2 threshold with the firing-rate regularizer off. time_steps only for a
// temporal code (direct has one analog frame).
ThesisConfig::FeatureExtraction make_ae_fe(
    const std::string& model, const std::string& encoding, int time_steps)
{
    ThesisConfig::FeatureExtraction fe;
    fe.strategy = "autoencoder";
    fe.autoencoder.model = model;
    fe.autoencoder.encoder_layer_spec =
        std::vector<std::string>{"linear:16:leaky", "linear:8:identity"};
    fe.autoencoder.decoder_layer_spec =
        std::vector<std::string>{"linear:16:leaky", "linear:output:identity"};
    fe.autoencoder.ae_loss_type = "mse";
    if (model == "snn-ae")
    {
        fe.autoencoder.encoding = encoding;
        if (encoding != "direct") fe.autoencoder.time_steps = time_steps;
        fe.autoencoder.voltage_threshold = 0.2f;
        fe.autoencoder.firing_rate_reg_lambda = 0.0f;
    }
    return fe;
}

ThesisConfig::Training make_ae_training()
{
    ThesisConfig::Training tr;
    tr.epochs = 5; // fast; enough to move off the zero init
    tr.learning_rate = 1e-3f;
    tr.optimizer_type = "adam";
    tr.gradient_clip_norm = 0.0f;
    tr.samples_per_batch = 8;
    return tr;
}

// An EEG autoencoder run of `fe` at `seed`.
std::vector<FeatureSet> run_ae(const ThesisDatasetView& view,
    const ThesisConfig::FeatureExtraction& fe,
    const ThesisConfig::Training& training,
    std::uint32_t seed)
{
    ThesisConfig cfg = test::phase00("eeg");
    cfg.experiment.seed = seed;
    cfg.feature_extraction = fe;
    cfg.training = training;
    cfg.validate();
    return extract_features(view, cfg, nullptr);
}

// Max |value| and cross-sample variance of the latent vectors.
void latent_stats(const std::vector<std::vector<double>>& v, double& max_abs, double& max_var)
{
    max_abs = 0.0;
    max_var = 0.0;
    if (v.empty()) return;
    const size_t dim = v[0].size();
    for (const auto& row : v)
        for (double x : row) max_abs = std::max(max_abs, std::abs(x));
    for (size_t d = 0; d < dim; ++d)
    {
        double mean = 0.0;
        for (const auto& row : v) mean += row[d];
        mean /= static_cast<double>(v.size());
        double var = 0.0;
        for (const auto& row : v) var += (row[d] - mean) * (row[d] - mean);
        var /= static_cast<double>(v.size());
        max_var = std::max(max_var, var);
    }
}
} // namespace

TEST(ThesisSnnAe, PoissonLatentIsNonDegenerate)
{
    auto view = make_multi_view(6);
    auto fe = make_ae_fe("snn-ae", "poisson", 8);
    auto tr = make_ae_training();

    auto sets = run_ae(view, fe, tr, 7u);
    ASSERT_EQ(sets.size(), 1u);
    const auto& vecs = sets[0].vectors;
    ASSERT_EQ(vecs.size(), view.samples.size());
    ASSERT_EQ(vecs[0].size(), 8u); // latent dim from encoder spec

    double max_abs = 0.0, max_var = 0.0;
    latent_stats(vecs, max_abs, max_var);
    EXPECT_GT(max_abs, 0.0) << "SNN-AE latent is all zeros — spikes never reached threshold";
    EXPECT_GT(max_var, 0.0) << "SNN-AE latent identical across samples — no discriminative info";
}

// Regression guard, not a hypothetical. The 2026-07-20 Phase 00 run built "base" (declared
// 64 hidden / 32 latent) and "small" (32 / 16) as the SAME 32 / 32 network: the width helpers
// of that time took a width equal to their fallback (64 hidden, 16 latent) as "not found" and
// moved on to the next layer. Every base/small pair of the thesis tables came out identical to
// 8 decimals; nothing failed. The widths below are the Phase 00 profiles' own. The latent
// width is the feature width, so it is checked here; the hidden width is not observable
// through extract_features().
TEST(ThesisAeArchitecture, LatentWidthIsTheDeclaredLastEncoderWidth)
{
    auto view = make_multi_view(6);
    auto tr = make_ae_training();
    tr.epochs = 1; // only the shape is under test
    const std::vector<std::pair<std::vector<std::string>, size_t>> cases = {
        {{"linear:64:leaky", "linear:32:identity"}, 32u}, // base
        {{"linear:32:leaky", "linear:16:identity"}, 16u}, // small (the July run used 32)
        {{"linear:16:leaky", "linear:8:identity"}, 8u},   // tiny
    };
    for (const auto& [encoder, latent] : cases)
    {
        auto fe = make_ae_fe("ann-ae", "direct", 0);
        fe.autoencoder.encoder_layer_spec = encoder;
        fe.autoencoder.decoder_layer_spec =
            std::vector<std::string>{encoder.front(), "linear:output:identity"};
        auto sets = run_ae(view, fe, tr, 7u);
        ASSERT_EQ(sets.size(), 1u);
        ASSERT_FALSE(sets[0].vectors.empty());
        EXPECT_EQ(sets[0].vectors[0].size(), latent) << encoder.front() << ", " << encoder.back();
    }
}

// ─── reproducibility ─────────────────────────────────────────────────────────
// Contract: the SAME profile with the SAME seed must produce BYTE-IDENTICAL features.
// Results must not change independently of a run.
//
// This is a regression guard, not a hypothetical. extract_features() used to seed only the
// spike frames and leave AutoencoderConfig::initializer_seed unset, so the initializers fell
// back to std::random_device (see include/initializers/{xavier,kaiming_snn}.hpp) and every
// run drew different weights. Two things followed: autoencoder feature extraction was not
// reproducible at all, and ThesisSnnAe.PoissonLatentIsNonDegenerate failed ~24% of runs
// (measured 6/25) whenever an unlucky draw left every encoder neuron below V_th.
//
// The handcrafted branch trains nothing and was always reproducible; it is included so a
// future regression is attributed to the branch that actually broke.
TEST(ThesisReproducibility, AutoencoderFeaturesAreIdenticalAcrossRuns)
{
    auto view = make_multi_view(6);
    auto tr = make_ae_training();

    for (const char* encoding : {"direct", "latency", "poisson"})
    {
        auto fe = make_ae_fe("snn-ae", encoding, 8);
        const auto a = run_ae(view, fe, tr, 7u);
        const auto b = run_ae(view, fe, tr, 7u);

        ASSERT_EQ(a.size(), b.size()) << "encoding=" << encoding;
        ASSERT_EQ(a[0].vectors.size(), b[0].vectors.size()) << "encoding=" << encoding;
        for (size_t i = 0; i < a[0].vectors.size(); ++i)
        {
            ASSERT_EQ(a[0].vectors[i].size(), b[0].vectors[i].size());
            for (size_t k = 0; k < a[0].vectors[i].size(); ++k)
            {
                EXPECT_DOUBLE_EQ(a[0].vectors[i][k], b[0].vectors[i][k])
                    << "encoding=" << encoding << " sample=" << i << " dim=" << k
                    << " -- same seed produced different features; weight init is probably "
                       "unseeded again (AutoencoderConfig::initializer_seed)";
            }
        }
    }
}

TEST(ThesisReproducibility, AnnAutoencoderFeaturesAreIdenticalAcrossRuns)
{
    auto view = make_multi_view(6);
    auto tr = make_ae_training();
    auto fe = make_ae_fe("ann-ae", "direct", 1);

    const auto a = run_ae(view, fe, tr, 11u);
    const auto b = run_ae(view, fe, tr, 11u);

    ASSERT_EQ(a[0].vectors.size(), b[0].vectors.size());
    for (size_t i = 0; i < a[0].vectors.size(); ++i)
        for (size_t k = 0; k < a[0].vectors[i].size(); ++k)
            EXPECT_DOUBLE_EQ(a[0].vectors[i][k], b[0].vectors[i][k])
                << "sample=" << i << " dim=" << k;
}

// A different seed SHOULD give different features -- otherwise the seed is being ignored
// entirely, which would make the above test pass vacuously.
TEST(ThesisReproducibility, DifferentSeedChangesAutoencoderFeatures)
{
    auto view = make_multi_view(6);
    auto tr = make_ae_training();
    // ann-ae, not snn-ae: a spiking latent can sit fully below (or above) V_th in this tiny
    // 6-sample config, in which case it is degenerate and NO initialisation changes it --
    // the control would then fail for a reason unrelated to seed propagation. The dense
    // autoencoder has no threshold, so its output must move when the weights move.
    auto fe = make_ae_fe("ann-ae", "direct", 1);

    const auto a = run_ae(view, fe, tr, 7u);
    const auto b = run_ae(view, fe, tr, 8u);

    bool any_diff = false;
    for (size_t i = 0; i < a[0].vectors.size() && !any_diff; ++i)
        for (size_t k = 0; k < a[0].vectors[i].size(); ++k)
            if (a[0].vectors[i][k] != b[0].vectors[i][k])
            {
                any_diff = true;
                break;
            }

    EXPECT_TRUE(any_diff)
        << "seeds 7 and 8 gave identical features -- the seed is not reaching the model, so "
           "the reproducibility tests above would pass vacuously";
}

TEST(ThesisSnnAe, EncodingChangesTheFeature)
{
    auto view = make_multi_view(4);
    auto tr = make_ae_training();

    auto poisson = run_ae(view, make_ae_fe("snn-ae", "poisson", 8), tr, 7u);
    auto direct = run_ae(view, make_ae_fe("snn-ae", "direct", 1), tr, 7u);

    ASSERT_EQ(poisson.size(), 1u);
    ASSERT_EQ(direct.size(), 1u);
    ASSERT_EQ(poisson[0].vectors.size(), direct[0].vectors.size());

    // The temporal (rate-coded) path must produce a materially different feature
    // than the one-shot analog path.
    double max_delta = 0.0;
    for (size_t s = 0; s < poisson[0].vectors.size(); ++s)
        for (size_t d = 0; d < poisson[0].vectors[s].size(); ++d)
            max_delta =
                std::max(max_delta, std::abs(poisson[0].vectors[s][d] - direct[0].vectors[s][d]));
    EXPECT_GT(max_delta, 1e-6) << "poisson and direct encodings yield identical features";
}

// ─── autoencoder reconstruction loss selection (mse | mae) ───────────────────
//
// The Trainer's loss is a compile-time template parameter, so ae_loss_type must be
// dispatched to a concrete instantiation. These guard that the profile field is
// parsed, validated, and defaulted — i.e. that "mae" is actually reachable.

namespace
{
// A Phase-00 EEG ann-ae run, every key it reads declared.
ThesisConfig make_min_ae_config()
{
    ThesisConfig cfg = test::phase00("eeg");
    test::set_training(cfg, 5, 1e-3f, 8);
    test::set_autoencoder(cfg,
        "ann-ae",
        {"linear:64:leaky", "linear:32:identity"},
        {"linear:64:leaky", "linear:output:identity"},
        "mse");
    return cfg;
}
} // namespace

TEST(ThesisAeLoss, HasNoDefault)
{
    // It used to default to "mse" (and the model to "lstm-ae"): a profile that forgot the
    // key trained with a loss nobody chose. Now an undeclared value cannot be read.
    ThesisConfig cfg;
    EXPECT_FALSE(cfg.feature_extraction.autoencoder.ae_loss_type.declared());
    EXPECT_THROW((void) cfg.feature_extraction.autoencoder.ae_loss_type.get(), std::logic_error);
    EXPECT_THROW((void) cfg.feature_extraction.autoencoder.model.get(), std::logic_error);
}

TEST(ThesisAeLoss, AcceptsMseAndMae)
{
    auto cfg = make_min_ae_config();
    cfg.feature_extraction.autoencoder.ae_loss_type = "mse";
    EXPECT_NO_THROW(cfg.validate());
    cfg.feature_extraction.autoencoder.ae_loss_type = "mae";
    EXPECT_NO_THROW(cfg.validate());
}

TEST(ThesisAeLoss, RejectsUnsupportedLoss)
{
    auto cfg = make_min_ae_config();
    // CrossEntropy is classification; l1/"" are not loss tokens this path knows.
    for (const char* bad : {"crossentropy", "l1", ""})
    {
        cfg.feature_extraction.autoencoder.ae_loss_type = bad;
        EXPECT_THROW(cfg.validate(), std::invalid_argument) << "should reject: " << bad;
    }
}

// ─── encoding <-> loss invariant (.wiki/Concepts/Spike-Encoding.md) ──────────

namespace
{
ThesisConfig make_snn_ae_config(const char* encoding, const char* loss)
{
    ThesisConfig cfg = make_min_ae_config();
    cfg.feature_extraction.autoencoder.model = "snn-ae";
    cfg.feature_extraction.autoencoder.ae_loss_type = loss;
    test::set_snn(cfg, encoding, 16, 0.2f, 0.0f, 0.0f, 0.0f);
    return cfg;
}
} // namespace

TEST(ThesisSpikeLoss, MatchedPairingsAccepted)
{
    EXPECT_NO_THROW(make_snn_ae_config("poisson", "spikecount").validate());
    EXPECT_NO_THROW(make_snn_ae_config("latency", "spiketime").validate());
}

TEST(ThesisSpikeLoss, MismatchedPairingsRejected)
{
    // SpikeTimeLoss sees only the first spike -> blind to rate information.
    EXPECT_THROW(make_snn_ae_config("poisson", "spiketime").validate(), std::invalid_argument);
    // SpikeCountLoss treats absent spikes as count 0 -> wrong gradient for late spikes.
    EXPECT_THROW(make_snn_ae_config("latency", "spikecount").validate(), std::invalid_argument);
}

TEST(ThesisSpikeLoss, RejectedForDirectEncoding)
{
    // direct is analog: there are no spikes for a spike loss to read.
    EXPECT_THROW(make_snn_ae_config("direct", "spikecount").validate(), std::invalid_argument);
    EXPECT_THROW(make_snn_ae_config("direct", "spiketime").validate(), std::invalid_argument);
    EXPECT_NO_THROW(make_snn_ae_config("direct", "mse").validate());
}

TEST(ThesisSpikeLoss, RejectedForNonSpikingModel)
{
    // ann-ae emits continuous values, never spikes.
    auto cfg = make_min_ae_config();
    cfg.feature_extraction.autoencoder.ae_loss_type = "spikecount";
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(ThesisSpikeLoss, ContinuousLossesRemainAvailableAsBaseline)
{
    // mse/mae stay selectable for spiking encodings as an explicit opt-out baseline.
    EXPECT_NO_THROW(make_snn_ae_config("poisson", "mse").validate());
    EXPECT_NO_THROW(make_snn_ae_config("latency", "mae").validate());
}

TEST(ThesisAeLoss, LstmAeTrainsWithMseOnly)
{
    // The LSTM-AE path is MSE only; another token would be silently ignored, so it is refused.
    ThesisConfig cfg = test::phase00("eeg");
    test::set_training(cfg, 5, 1e-3f, 8);
    test::set_autoencoder(cfg, "lstm-ae", {"linear:64:leaky", "linear:32:identity"}, {}, "mse");
    EXPECT_NO_THROW(cfg.validate());
    cfg.feature_extraction.autoencoder.ae_loss_type = "mae";
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

// The LSTM-AE reads the same spectral input as the other autoencoders and PCA, not the
// waveform. Two checks through the public API: a signal shorter than one 512-sample spectrum
// frame is refused (the old waveform framing accepted any length), and samples the spectrum
// does not read do not change the code.
namespace
{
ThesisConfig lstm_ae_config()
{
    ThesisConfig cfg = test::phase00("eeg");
    test::set_training(cfg, 2, 1e-3f, 4);
    test::set_autoencoder(cfg, "lstm-ae", {"linear:16:leaky", "linear:8:identity"}, {}, "mse");
    cfg.validate();
    return cfg;
}

ThesisDatasetView two_speakers_two_shifts(int length)
{
    ThesisDatasetView view;
    const double freq[2] = {0.05, 0.11};
    const int shift[2] = {0, 137};
    for (int speaker = 0; speaker < 2; ++speaker)
        for (int take = 0; take < 2; ++take)
        {
            ThesisSample m;
            m.subject_id = speaker;
            m.stimulus = take;
            m.eeg = nn::Tensor(1, length);
            for (int i = 0; i < length; ++i)
                m.eeg.at(0, i) = test::synthetic_value(i + shift[take], freq[speaker], 7U);
            view.samples.push_back(std::move(m));
        }
    view.n_subjects = 2;
    view.n_stimuli = 2;
    return view;
}
} // namespace

TEST(ThesisLstmAe, RefusesASignalShorterThanOneSpectrumFrame)
{
    EXPECT_THROW((void) extract_features(two_speakers_two_shifts(511), lstm_ae_config(), nullptr),
        std::invalid_argument);
}

// Welch reads whole 512-sample frames with a 256 hop: of 4196 samples it reads the first
// 4096 (15 frames) and never the last 100. Two takes that differ ONLY there have the same
// spectral input, so the LSTM-AE must give them bit-identical codes; the old waveform
// framing read every sample and would not.
TEST(ThesisLstmAe, ReadsTheSpectrumNotTheWaveform)
{
    ThesisDatasetView view;
    const double freq[2] = {0.05, 0.11};
    for (int speaker = 0; speaker < 2; ++speaker)
        for (int take = 0; take < 2; ++take)
        {
            ThesisSample m;
            m.subject_id = speaker;
            m.stimulus = take;
            m.eeg = nn::Tensor(1, 4196);
            for (int i = 0; i < 4196; ++i)
                m.eeg.at(0, i) = test::synthetic_value(i, freq[speaker], 7U) +
                                 (i >= 4096 && take == 1 ? 5.0f : 0.0f); // unread tail
            view.samples.push_back(std::move(m));
        }
    view.n_subjects = 2;
    view.n_stimuli = 2;

    const auto sets = extract_features(view, lstm_ae_config(), nullptr);
    const auto& v = sets[0].vectors; // a0, a1, b0, b1
    ASSERT_EQ(v.size(), 4U);
    ASSERT_EQ(v[0].size(), 8U);
    EXPECT_EQ(v[0], v[1]) << "an unread tail must not change the code";
    EXPECT_EQ(v[2], v[3]) << "an unread tail must not change the code";
    EXPECT_NE(v[0], v[2]) << "two frequencies must not share a code";
}

TEST(ThesisAeLoss, ParsedFromProfileJson)
{
    const auto j = nlohmann::json::parse(R"({
        "experiment": {"run_tag": "t", "seed": 42, "repeats": 1},
        "dataset": {"root": "/x", "results_dir": "results/thesis", "modality": "eeg", "eeg_condition": "pronounced",
                    "max_samples": 0},
        "feature_extraction": {
            "strategy": "autoencoder",
            "autoencoder": {
                "model": "ann-ae",
                "encoder_layer_spec": ["linear:64:leaky", "linear:32:identity"],
                "decoder_layer_spec": ["linear:64:leaky", "linear:output:identity"],
                "ae_loss_type": "mae"
            }
        },
        "paraconsistent": {"enabled": true, "speaker_subsets": false},
        "classifier": {"enabled": false},
        "training": {"epochs": 5, "learning_rate": 0.001, "optimizer_type": "adam",
                     "gradient_clip_norm": 0.0, "samples_per_batch": 8}
    })");
    const auto cfg = ThesisConfig::from_json(j);
    EXPECT_EQ(cfg.feature_extraction.autoencoder.ae_loss_type, "mae");
    EXPECT_NO_THROW(cfg.validate());
}
