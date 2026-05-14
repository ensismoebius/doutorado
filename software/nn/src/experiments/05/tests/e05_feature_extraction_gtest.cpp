// Unit tests for E05FeatureExtraction: scalar descriptors and extract_handcrafted.
// No SQLite or ProgressManager dependency — all inputs are synthetic.

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
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
        EXPECT_LT(j, 0.05); // < 5% jitter for regular sine
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
        EXPECT_LT(s, 0.05);
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
    auto fv  = extract_handcrafted(sig, cfg);
    // level-2 DTWPT → 4 leaf nodes → 4 energy values
    EXPECT_EQ(fv.size(), 4u);
}

TEST(E05HandcrafteExtract, MultipleDescriptorsDimAdditive)
{
    auto cfg = make_hc_cfg({"energy", "zcr", "entropy"}, 2);
    auto sig = make_sine();
    auto fv  = extract_handcrafted(sig, cfg);
    // 4 bands × 3 descriptors = 12
    EXPECT_EQ(fv.size(), 12u);
}

TEST(E05HandcrafteExtract, AllDescriptors_NonEmpty)
{
    auto cfg = make_hc_cfg({"energy", "zcr", "entropy", "teager", "jitter", "shimmer"}, 2);
    auto sig = make_sine(256, 100.0, 44100.0);
    auto fv  = extract_handcrafted(sig, cfg);
    // 4 bands × 6 descriptors = 24; no NaN (jitter/shimmer replaced by 0 when undefined)
    EXPECT_EQ(fv.size(), 24u);
    for (size_t i = 0; i < fv.size(); ++i)
        EXPECT_FALSE(std::isnan(fv[i])) << "NaN at index " << i;
}

TEST(E05HandcrafteExtract, Level3GivesMoreBands)
{
    auto cfg2 = make_hc_cfg({"energy"}, 2);
    auto cfg3 = make_hc_cfg({"energy"}, 3);
    auto sig  = make_sine(512);
    EXPECT_LT(extract_handcrafted(sig, cfg2).size(),
              extract_handcrafted(sig, cfg3).size());
}

TEST(E05HandcrafteExtract, EnergyValuesNonNegative)
{
    auto cfg = make_hc_cfg({"energy"}, 2);
    auto sig = make_sine();
    for (double v : extract_handcrafted(sig, cfg))
        EXPECT_GE(v, 0.0);
}
