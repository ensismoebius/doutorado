// Unit tests for E05FeatureExtraction: scalar descriptors and extract_handcrafted.
// No SQLite or ProgressManager dependency — all inputs are synthetic.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "../lib/include/E05Config.hpp"
#include "../lib/include/E05FeatureExtraction.hpp"

using namespace e05;

// ─── compute_energy ──────────────────────────────────────────────────────────

TEST(E05Energy, EmptyReturnsZero)
{
    EXPECT_DOUBLE_EQ(compute_energy({}), 0.0);
}

TEST(E05Energy, ConstantSignal)
{
    std::vector<double> sig(8, 3.0);
    EXPECT_DOUBLE_EQ(compute_energy(sig), 8.0 * 9.0);
}

TEST(E05Energy, UnitImpulse)
{
    std::vector<double> sig(16, 0.0);
    sig[0] = 1.0;
    EXPECT_DOUBLE_EQ(compute_energy(sig), 1.0);
}

// ─── compute_zcr ─────────────────────────────────────────────────────────────

TEST(E05ZCR, EmptyOrSingleReturnsZero)
{
    EXPECT_DOUBLE_EQ(compute_zcr({}), 0.0);
    EXPECT_DOUBLE_EQ(compute_zcr({1.0}), 0.0);
}

TEST(E05ZCR, AlternatingSignal)
{
    // +-+-+- … → every consecutive pair crosses zero
    std::vector<double> sig = {1.0, -1.0, 1.0, -1.0, 1.0};
    // 4 crossings / 4 intervals = 1.0
    EXPECT_DOUBLE_EQ(compute_zcr(sig), 1.0);
}

TEST(E05ZCR, ConstantPositiveNoCrossings)
{
    std::vector<double> sig(10, 1.0);
    EXPECT_DOUBLE_EQ(compute_zcr(sig), 0.0);
}

TEST(E05ZCR, RateInRange)
{
    std::vector<double> sig(100);
    for (size_t i = 0; i < sig.size(); ++i)
        sig[i] = (i % 5 == 0) ? -1.0 : 1.0;
    double zcr = compute_zcr(sig);
    EXPECT_GE(zcr, 0.0);
    EXPECT_LE(zcr, 1.0);
}

// ─── compute_entropy ─────────────────────────────────────────────────────────

TEST(E05Entropy, EmptyReturnsZero)
{
    EXPECT_DOUBLE_EQ(compute_entropy({}), 0.0);
}

TEST(E05Entropy, AllZeroesReturnsZero)
{
    EXPECT_DOUBLE_EQ(compute_entropy({0.0, 0.0, 0.0}), 0.0);
}

TEST(E05Entropy, UniformSignalMaxEntropy)
{
    // |v|/total uniform → each p = 1/n → entropy = log2(n)
    std::vector<double> sig(8, 1.0);
    double ent = compute_entropy(sig);
    EXPECT_NEAR(ent, std::log2(8.0), 1e-10);
}

TEST(E05Entropy, ImpulseLowEntropy)
{
    // All energy in one sample → entropy ≈ 0
    std::vector<double> sig(16, 0.0);
    sig[0] = 1.0;
    EXPECT_NEAR(compute_entropy(sig), 0.0, 1e-10);
}

// ─── compute_teager ──────────────────────────────────────────────────────────

TEST(E05Teager, TooShortReturnsZero)
{
    EXPECT_DOUBLE_EQ(compute_teager({1.0, 2.0}), 0.0);
}

TEST(E05Teager, ConstantSignalZero)
{
    // Teager of constant c: c^2 - c*c = 0
    std::vector<double> sig(10, 2.0);
    EXPECT_NEAR(compute_teager(sig), 0.0, 1e-12);
}

TEST(E05Teager, PureSinePositive)
{
    // Pure sine has positive mean Teager energy
    const int N = 64;
    std::vector<double> sig(N);
    for (int i = 0; i < N; ++i)
        sig[i] = std::sin(2.0 * M_PI * 4.0 * i / N);
    EXPECT_GT(compute_teager(sig), 0.0);
}

// ─── compute_jitter ──────────────────────────────────────────────────────────

TEST(E05Jitter, TooFewPeaksReturnsNaN)
{
    // Flat signal → no peaks
    std::vector<double> sig(32, 0.5);
    EXPECT_TRUE(std::isnan(compute_jitter(sig, 44100.0)));
}

TEST(E05Jitter, RegularSineNearZeroJitter)
{
    // Regular 100 Hz sine at 44100 Hz → very consistent periods → low jitter
    const int N = 4096;
    const double freq = 100.0;
    const double sr   = 44100.0;
    std::vector<double> sig(N);
    for (int i = 0; i < N; ++i)
        sig[i] = std::sin(2.0 * M_PI * freq * i / sr);

    double j = compute_jitter(sig, sr);
    if (!std::isnan(j))
    {
        EXPECT_LT(j, 0.05); // < 5% jitter for regular sine
    }
}

// ─── compute_shimmer ─────────────────────────────────────────────────────────

TEST(E05Shimmer, TooFewPeaksReturnsNaN)
{
    std::vector<double> sig(32, 0.5);
    EXPECT_TRUE(std::isnan(compute_shimmer(sig, 44100.0)));
}

TEST(E05Shimmer, RegularSineNearZeroShimmer)
{
    const int N = 4096;
    const double freq = 100.0;
    const double sr   = 44100.0;
    std::vector<double> sig(N);
    for (int i = 0; i < N; ++i)
        sig[i] = std::sin(2.0 * M_PI * freq * i / sr);

    double s = compute_shimmer(sig, sr);
    if (!std::isnan(s))
    {
        EXPECT_LT(s, 0.05);
    }
}

// ─── extract_handcrafted ─────────────────────────────────────────────────────

namespace
{
E05Config::HandcraftedConfig make_hc_cfg(const std::vector<std::string>& descs,
    int dtwpt_level = 2)
{
    E05Config::HandcraftedConfig cfg;
    cfg.descriptors = descs;
    cfg.dtwpt_level = dtwpt_level;
    cfg.scale       = "mel";
    return cfg;
}

// Power-of-2 length sine signal.
std::vector<double> make_sine(int N = 256, double freq = 10.0, double sr = 44100.0)
{
    std::vector<double> s(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        s[static_cast<size_t>(i)] = std::sin(2.0 * M_PI * freq * i / sr);
    return s;
}
} // namespace

TEST(E05HandcrafteExtract, EnergyOnly_DimMatchesBands)
{
    auto cfg = make_hc_cfg({"energy"}, 2);
    auto sig = make_sine();
    auto fv  = extract_handcrafted(sig, cfg, 44100.0);
    // level-2 DTWPT → 4 sub-bands; at 44100 Hz MEL scale all 4 map to distinct bins
    EXPECT_EQ(fv.size(), 4u);
}

TEST(E05HandcrafteExtract, MultipleDescriptorsDimAdditive)
{
    auto cfg = make_hc_cfg({"energy", "zcr", "entropy"}, 2);
    auto sig = make_sine();
    auto fv  = extract_handcrafted(sig, cfg, 44100.0);
    // 4 MEL bins × 3 descriptors = 12
    EXPECT_EQ(fv.size(), 12u);
}

TEST(E05HandcrafteExtract, AllDescriptors_NonEmpty)
{
    auto cfg = make_hc_cfg({"energy", "zcr", "entropy", "teager", "jitter", "shimmer"}, 2);
    auto sig = make_sine(256, 100.0, 44100.0);
    auto fv  = extract_handcrafted(sig, cfg, 44100.0);
    // 4 MEL bins × 6 descriptors = 24; no NaN (jitter/shimmer replaced by 0 when undefined)
    EXPECT_EQ(fv.size(), 24u);
    for (size_t i = 0; i < fv.size(); ++i)
        EXPECT_FALSE(std::isnan(fv[i])) << "NaN at index " << i;
}

TEST(E05HandcrafteExtract, Level3GivesMoreBands)
{
    auto cfg2 = make_hc_cfg({"energy"}, 2);
    auto cfg3 = make_hc_cfg({"energy"}, 3);
    auto sig  = make_sine(512);
    EXPECT_LT(extract_handcrafted(sig, cfg2, 44100.0).size(),
              extract_handcrafted(sig, cfg3, 44100.0).size());
}

TEST(E05HandcrafteExtract, EnergyValuesNonNegative)
{
    auto cfg = make_hc_cfg({"energy"}, 2);
    auto sig = make_sine();
    for (double v : extract_handcrafted(sig, cfg, 44100.0))
        EXPECT_GE(v, 0.0);
}

// ─── pre-emphasis ────────────────────────────────────────────────────────────

TEST(E05PreEmphasis, MatchesThesisWorkedExample)
{
    // Thesis §Pré-ênfase: x=[1.0,0.9,0.6], alpha=0.97 → [1.0,-0.07,-0.273].
    std::vector<double> sig = {1.0, 0.9, 0.6};
    apply_preemphasis(sig, 0.97);
    ASSERT_EQ(sig.size(), 3u);
    EXPECT_NEAR(sig[0], 1.0, 1e-12);      // y[0] unchanged
    EXPECT_NEAR(sig[1], -0.07, 1e-12);    // 0.9 - 0.97*1.0
    EXPECT_NEAR(sig[2], -0.273, 1e-12);   // 0.6 - 0.97*0.9
}

TEST(E05PreEmphasis, UsesOriginalNotFilteredPredecessor)
{
    // Back-to-front order: y[2] must use the ORIGINAL x[1], not the filtered one.
    std::vector<double> sig = {2.0, 2.0, 2.0};
    apply_preemphasis(sig, 0.5);
    EXPECT_NEAR(sig[0], 2.0, 1e-12);            // unchanged
    EXPECT_NEAR(sig[1], 2.0 - 0.5 * 2.0, 1e-12); // 1.0
    EXPECT_NEAR(sig[2], 2.0 - 0.5 * 2.0, 1e-12); // 1.0 (uses original x[1]=2.0)
}

TEST(E05PreEmphasis, AlphaZeroIsIdentity)
{
    std::vector<double> sig = {0.3, -0.5, 0.8, 1.0};
    const std::vector<double> orig = sig;
    apply_preemphasis(sig, 0.0);
    EXPECT_EQ(sig, orig);
}

TEST(E05PreEmphasis, ShortSignalsUnchanged)
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

TEST(E05Wavelet, AllTraitWaveletsExtractFinite)
{
    // Every wavelet with coefficient traits in Types.hpp must run end-to-end and
    // produce finite features. Longer filters need a longer signal than their
    // support, so use 512 samples.
    const std::vector<std::string> wavelets = {
        "haar",   "daub4",  "daub6",  "daub8",  "daub10", "daub12",
        "daub14", "daub16", "daub18", "daub20", "daub22", "daub24",
        "daub26", "daub28", "daub30", "daub32", "daub34", "daub36",
        "daub38", "daub40", "daub42", "daub44", "daub46"};

    auto sig = make_sine(512, 100.0, 44100.0);
    for (const auto& w : wavelets)
    {
        auto cfg = make_hc_cfg({"energy", "entropy"}, 3);
        cfg.wavelet = w;
        auto fv = extract_handcrafted(sig, cfg, 44100.0);
        EXPECT_FALSE(fv.empty()) << "empty feature vector for wavelet " << w;
        for (double v : fv)
            EXPECT_TRUE(std::isfinite(v)) << "non-finite feature for wavelet " << w;
    }
}

TEST(E05Wavelet, UnknownWaveletThrows)
{
    auto cfg = make_hc_cfg({"energy"}, 2);
    cfg.wavelet = "not-a-wavelet";
    auto sig = make_sine();
    EXPECT_THROW(extract_handcrafted(sig, cfg, 44100.0), std::invalid_argument);
}

TEST(E05Wavelet, DifferentWaveletsDifferentFeatures)
{
    // Haar (2-tap) and Daub20 (20-tap) must not yield identical decompositions.
    auto sig = make_sine(512, 100.0, 44100.0);
    auto cfg_haar = make_hc_cfg({"energy"}, 3);
    cfg_haar.wavelet = "haar";
    auto cfg_d20 = make_hc_cfg({"energy"}, 3);
    cfg_d20.wavelet = "daub20";
    EXPECT_NE(extract_handcrafted(sig, cfg_haar, 44100.0),
              extract_handcrafted(sig, cfg_d20, 44100.0));
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
E05DatasetView make_paired_view()
{
    E05DatasetView view;
    for (int s = 0; s < 2; ++s)
    {
        E05Sample sample;
        sample.audio = make_column_tensor(256, 100.0 + s, 44100.0);
        sample.eeg   = make_column_tensor(256, 10.0 + s, 1024.0);
        sample.subject_id = s;
        sample.stimulus = s;
        sample.text_phrase = (s == 0) ? "a" : "e";
        view.samples.push_back(std::move(sample));
    }
    view.n_subjects = 2;
    view.n_stimuli = 2;
    return view;
}

E05Config::FeatureExtraction make_hc_fe(const std::vector<std::string>& descs, int level = 2)
{
    E05Config::FeatureExtraction fe;
    fe.strategy = "handcrafted";
    fe.handcrafted = make_hc_cfg(descs, level);
    return fe;
}
} // namespace

TEST(E05Fusion, LateFusionDimIsVoicePlusEeg)
{
    auto view = make_paired_view();
    auto fe = make_hc_fe({"energy"});
    E05Config::Training tr; // defaults unused by handcrafted path

    auto voice = extract_features(view, fe, tr, "voice");
    auto eeg   = extract_features(view, fe, tr, "eeg");
    auto late  = extract_features(view, fe, tr, "fused", "late");

    ASSERT_EQ(voice.size(), 1u);
    ASSERT_EQ(eeg.size(), 1u);
    ASSERT_EQ(late.size(), 1u);
    ASSERT_FALSE(late[0].vectors.empty());

    // Late fusion concatenates the per-signal feature vectors sample-by-sample.
    EXPECT_EQ(late[0].vectors[0].size(),
              voice[0].vectors[0].size() + eeg[0].vectors[0].size());
    // Concatenation order is voice-part then eeg-part.
    for (size_t i = 0; i < voice[0].vectors[0].size(); ++i)
        EXPECT_DOUBLE_EQ(late[0].vectors[0][i], voice[0].vectors[0][i]);
}

TEST(E05Fusion, EarlyFusionSingleExtractionDiffersFromLate)
{
    auto view = make_paired_view();
    auto fe = make_hc_fe({"energy"});
    E05Config::Training tr;

    auto early = extract_features(view, fe, tr, "fused", "early");
    auto late  = extract_features(view, fe, tr, "fused", "late");

    ASSERT_EQ(early.size(), 1u);
    ASSERT_EQ(late.size(), 1u);
    // Early fuses raw signals (voice++eeg = 512 samples → padded to 512) into one
    // DTWPT pass; late runs two independent passes. Dimensions must not coincide
    // with the late-fusion sum in general, and labels must distinguish the modes.
    EXPECT_NE(early[0].label, late[0].label);
    EXPECT_NE(early[0].label.find("fused-early"), std::string::npos);
    EXPECT_NE(late[0].label.find("fused-late"), std::string::npos);
}

TEST(E05Fusion, UnknownFusionModeThrows)
{
    auto view = make_paired_view();
    auto fe = make_hc_fe({"energy"});
    E05Config::Training tr;
    EXPECT_THROW(extract_features(view, fe, tr, "fused", "bogus"), std::invalid_argument);
}
