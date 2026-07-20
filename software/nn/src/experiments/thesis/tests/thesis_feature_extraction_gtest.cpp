// Unit tests for ThesisFeatureExtraction: scalar descriptors and extract_handcrafted.
// No SQLite or ProgressManager dependency — all inputs are synthetic.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../lib/include/ThesisConfig.hpp"
#include "../lib/include/ThesisFeatureExtraction.hpp"

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

TEST(ThesisZCR, EmptyOrSingleReturnsZero)
{
    EXPECT_DOUBLE_EQ(compute_zcr({}), 0.0);
    EXPECT_DOUBLE_EQ(compute_zcr({1.0}), 0.0);
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

TEST(ThesisEntropy, EmptyReturnsZero)
{
    EXPECT_DOUBLE_EQ(compute_entropy({}), 0.0);
}

TEST(ThesisEntropy, AllZeroesReturnsZero)
{
    EXPECT_DOUBLE_EQ(compute_entropy({0.0, 0.0, 0.0}), 0.0);
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

TEST(ThesisTeager, TooShortReturnsZero)
{
    EXPECT_DOUBLE_EQ(compute_teager({1.0, 2.0}), 0.0);
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

TEST(ThesisJitter, TooFewPeaksReturnsNaN)
{
    // Flat signal → no peaks
    std::vector<double> sig(32, 0.5);
    EXPECT_TRUE(std::isnan(compute_jitter(sig, 44100.0)));
}

TEST(ThesisJitter, RegularSineNearZeroJitter)
{
    // Regular 100 Hz sine at 44100 Hz → very consistent periods → low jitter
    const int N = 4096;
    const double freq = 100.0;
    const double sr = 44100.0;
    std::vector<double> sig(N);
    for (int i = 0; i < N; ++i) sig[i] = std::sin(2.0 * M_PI * freq * i / sr);

    double j = compute_jitter(sig, sr);
    if (!std::isnan(j))
    {
        EXPECT_LT(j, 0.05); // < 5% jitter for regular sine
    }
}

// ─── compute_shimmer ─────────────────────────────────────────────────────────

TEST(ThesisShimmer, TooFewPeaksReturnsNaN)
{
    std::vector<double> sig(32, 0.5);
    EXPECT_TRUE(std::isnan(compute_shimmer(sig, 44100.0)));
}

TEST(ThesisShimmer, RegularSineNearZeroShimmer)
{
    const int N = 4096;
    const double freq = 100.0;
    const double sr = 44100.0;
    std::vector<double> sig(N);
    for (int i = 0; i < N; ++i) sig[i] = std::sin(2.0 * M_PI * freq * i / sr);

    double s = compute_shimmer(sig, sr);
    if (!std::isnan(s))
    {
        EXPECT_LT(s, 0.05);
    }
}

// ─── extract_handcrafted ─────────────────────────────────────────────────────

namespace
{
ThesisConfig::HandcraftedConfig make_hc_cfg(
    const std::vector<std::string>& descs, int dtwpt_level = 2)
{
    ThesisConfig::HandcraftedConfig cfg;
    cfg.descriptors = descs;
    cfg.dtwpt_level = dtwpt_level;
    cfg.scale = "mel";
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

TEST(ThesisHandcrafteExtract, AllDescriptors_NonEmpty)
{
    auto cfg = make_hc_cfg({"energy", "zcr", "entropy", "teager", "jitter", "shimmer"}, 2);
    auto sig = make_sine(256, 100.0, 44100.0);
    auto fv = extract_handcrafted(sig, cfg, 44100.0);
    // 4 MEL bins × 6 descriptors = 24; no NaN (jitter/shimmer replaced by 0 when undefined)
    EXPECT_EQ(fv.size(), 24u);
    for (size_t i = 0; i < fv.size(); ++i) EXPECT_FALSE(std::isnan(fv[i])) << "NaN at index " << i;
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
        t.at(i, 0) = static_cast<float>(std::sin(2.0 * M_PI * freq * i / sr));
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
} // namespace

TEST(ThesisFusion, LateFusionDimIsVoicePlusEeg)
{
    auto view = make_paired_view();
    auto fe = make_hc_fe({"energy"});
    ThesisConfig::Training tr; // defaults unused by handcrafted path

    auto voice = extract_features(view, fe, tr, "voice");
    auto eeg = extract_features(view, fe, tr, "eeg");
    auto late = extract_features(view, fe, tr, "fused", "late");

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
    ThesisConfig::Training tr;

    auto early = extract_features(view, fe, tr, "fused", "early");
    auto late = extract_features(view, fe, tr, "fused", "late");

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
    ThesisConfig::Training tr;
    EXPECT_THROW(extract_features(view, fe, tr, "fused", "bogus"), std::invalid_argument);
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

ThesisConfig::FeatureExtraction make_ae_fe(
    const std::string& model, const std::string& encoding, int time_steps)
{
    ThesisConfig::FeatureExtraction fe;
    fe.strategy = "autoencoder";
    fe.autoencoder.model = model;
    fe.autoencoder.encoder_layer_spec = {"linear:16:leaky", "linear:8:identity"};
    fe.autoencoder.decoder_layer_spec = {"linear:16:leaky", "linear:output:identity"};
    fe.autoencoder.encoding = encoding;
    fe.autoencoder.time_steps = time_steps;
    return fe;
}

ThesisConfig::Training make_ae_training()
{
    ThesisConfig::Training tr;
    tr.epochs = 5; // fast; enough to move off the zero init
    tr.samples_per_batch = 8;
    return tr;
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

    auto sets = extract_features(view, fe, tr, "eeg", "late", /*seed=*/7u);
    ASSERT_EQ(sets.size(), 1u);
    const auto& vecs = sets[0].vectors;
    ASSERT_EQ(vecs.size(), view.samples.size());
    ASSERT_EQ(vecs[0].size(), 8u); // latent dim from encoder spec

    double max_abs = 0.0, max_var = 0.0;
    latent_stats(vecs, max_abs, max_var);
    EXPECT_GT(max_abs, 0.0) << "SNN-AE latent is all zeros — spikes never reached threshold";
    EXPECT_GT(max_var, 0.0) << "SNN-AE latent identical across samples — no discriminative info";
}

TEST(ThesisSnnAe, EncodingChangesTheFeature)
{
    auto view = make_multi_view(4);
    auto tr = make_ae_training();

    auto poisson =
        extract_features(view, make_ae_fe("snn-ae", "poisson", 8), tr, "eeg", "late", 7u);
    auto direct = extract_features(view, make_ae_fe("snn-ae", "direct", 1), tr, "eeg", "late", 7u);

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
