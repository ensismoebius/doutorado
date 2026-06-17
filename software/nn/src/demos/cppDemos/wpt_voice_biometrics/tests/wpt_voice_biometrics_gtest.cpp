/**
 * @file wpt_voice_biometrics_gtest.cpp
 * @brief Unit tests for wpt_voice_biometrics signal processing functions.
 *
 * Functions under test are replicated here (they are defined inside main.cpp
 * with no external linkage).  codificacao::encode_poisson is tested via the
 * real implementation compiled in by CMakeLists.txt.
 *
 * Synthetic data only — no filesystem access.
 */

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "codificacao.hpp"
#include "tensor/Tensor.hpp"
#include "wavelet/waveletOperations.hpp"

// ============================================================
// Replicated helpers (identical logic to main.cpp)
// ============================================================

struct ExtractionConfig
{
    int sample_rate{44100};
    int window_size{512};
    int hop_size{256};
    int num_bands{32};
    int wpt_level{5};
};

static auto compute_wpt_level(int window_size, int num_bandas) -> int
{
    const int nivel_max = static_cast<int>(std::floor(std::log2(std::max(1, window_size))));
    const int nivel_nec = static_cast<int>(std::ceil(std::log2(std::max(1, num_bandas))));
    return std::max(1, std::min(nivel_max, nivel_nec));
}

static auto generate_hann_window(int size) -> std::vector<double>
{
    std::vector<double> w(static_cast<size_t>(size));
    for (int i = 0; i < size; ++i)
    {
        constexpr double pi = std::numbers::pi;
        w[static_cast<size_t>(i)] =
            0.5 * (1.0 - std::cos(2.0 * pi * static_cast<double>(i) /
                                      static_cast<double>(size - 1)));
    }
    return w;
}

static auto apply_windowing(const std::vector<double>& signal, const ExtractionConfig& cfg)
    -> std::vector<std::vector<double>>
{
    std::vector<std::vector<double>> windows;
    if (signal.size() < static_cast<size_t>(cfg.window_size)) return windows;
    const auto window = generate_hann_window(cfg.window_size);
    for (int start = 0; start + cfg.window_size <= static_cast<int>(signal.size());
         start += cfg.hop_size)
    {
        std::vector<double> seg(static_cast<size_t>(cfg.window_size));
        for (int i = 0; i < cfg.window_size; ++i)
            seg[static_cast<size_t>(i)] =
                signal[static_cast<size_t>(start + i)] * window[static_cast<size_t>(i)];
        windows.push_back(std::move(seg));
    }
    return windows;
}

static auto interpolate_to_size(const std::vector<double>& src, int dst_size)
    -> std::vector<float>
{
    if (dst_size <= 0) return {};
    if (static_cast<int>(src.size()) == dst_size)
        return std::vector<float>(src.begin(), src.end());
    std::vector<float> out(static_cast<size_t>(dst_size));
    for (int i = 0; i < dst_size; ++i)
    {
        double pos = (static_cast<double>(i) / static_cast<double>(dst_size - 1)) *
                     static_cast<double>(src.size() - 1);
        auto idx0 = static_cast<size_t>(std::floor(pos));
        auto idx1 = static_cast<size_t>(std::ceil(pos));
        double frac = pos - static_cast<double>(idx0);
        out[static_cast<size_t>(i)] =
            static_cast<float>((1.0 - frac) * src[idx0] + frac * src[idx1]);
    }
    return out;
}

static auto compute_wpt_energy(const std::vector<double>& window, int num_bands, int wpt_level)
    -> std::vector<float>
{
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const std::vector<double> haar{inv_sqrt2, inv_sqrt2};
    int target_size = wavelets::get_next_power_of_two(static_cast<double>(window.size()));
    std::vector<double> padded(window.begin(), window.end());
    padded.resize(static_cast<size_t>(target_size), 0.0);
    auto transform = wavelets::malat(
        padded, std::span<const double>(haar), wavelets::PACKET_WAVELET,
        static_cast<unsigned int>(wpt_level));
    auto energies = wavelets::extract_subband_energies(transform, wpt_level);
    return interpolate_to_size(energies, num_bands);
}

static auto preprocess_energy(const std::vector<float>& energy) -> std::vector<float>
{
    std::vector<float> out(energy.size());
    float max_value = 0.0F;
    for (size_t i = 0; i < energy.size(); ++i)
    {
        float v = std::log1pf(std::max(0.0F, energy[i]));
        out[i] = v;
        max_value = std::max(max_value, v);
    }
    if (max_value < 1e-8F) { std::fill(out.begin(), out.end(), 0.0F); return out; }
    std::transform(out.begin(), out.end(), out.begin(),
        [max_value](float v) { return std::clamp(v / max_value, 0.0F, 1.0F); });
    return out;
}

// ============================================================
// Fixtures
// ============================================================

class WptVoiceBioTest : public ::testing::Test
{
   protected:
    static constexpr int kSampleRate = 8000;
    static constexpr int kWindowSize = 512;
    static constexpr int kHopSize = 256;
    static constexpr int kNumBands = 32;

    ExtractionConfig cfg{kSampleRate, kWindowSize, kHopSize, kNumBands, 5};

    // 1-second synthetic sine
    std::vector<double> audio;

    void SetUp() override
    {
        audio.resize(static_cast<size_t>(kSampleRate));
        for (int i = 0; i < kSampleRate; ++i)
            audio[static_cast<size_t>(i)] =
                std::sin(2.0 * std::numbers::pi * 440.0 * static_cast<double>(i) /
                         static_cast<double>(kSampleRate));
    }
};

// ============================================================
// compute_wpt_level
// ============================================================

TEST(WptLevel, Window512_Bands32_ReturnsValidLevel)
{
    int lvl = compute_wpt_level(512, 32);
    EXPECT_GE(lvl, 1);
    EXPECT_LE(lvl, static_cast<int>(std::floor(std::log2(512.0))));
}

TEST(WptLevel, MinimumIsOne)
{
    int lvl = compute_wpt_level(1, 1);
    EXPECT_EQ(lvl, 1);
}

TEST(WptLevel, LargeWindowAndBands)
{
    int lvl = compute_wpt_level(1024, 64);
    EXPECT_GE(lvl, 1);
    EXPECT_LE(lvl, 10);
}

// ============================================================
// generate_hann_window
// ============================================================

TEST(HannWindow, StartsAndEndsNearZero)
{
    auto w = generate_hann_window(512);
    EXPECT_NEAR(w.front(), 0.0, 1e-9);
    EXPECT_NEAR(w.back(), 0.0, 1e-3);
}

TEST(HannWindow, PeakNearHalf)
{
    int n = 512;
    auto w = generate_hann_window(n);
    // Peak at center (index n/2) should be ~1.0
    EXPECT_NEAR(w[static_cast<size_t>(n / 2)], 1.0, 1e-3);
}

TEST(HannWindow, AllValuesInUnitRange)
{
    auto w = generate_hann_window(256);
    for (double v : w)
    {
        EXPECT_GE(v, -1e-12);
        EXPECT_LE(v, 1.0 + 1e-12);
    }
}

// ============================================================
// apply_windowing
// ============================================================

TEST_F(WptVoiceBioTest, ApplyWindowing_NumberOfWindows)
{
    auto wins = apply_windowing(audio, cfg);
    // Expected: floor((n - window_size) / hop_size) + 1
    int expected =
        static_cast<int>((static_cast<int>(audio.size()) - kWindowSize) / kHopSize) + 1;
    EXPECT_EQ(static_cast<int>(wins.size()), expected);
}

TEST_F(WptVoiceBioTest, ApplyWindowing_EachWindowCorrectSize)
{
    auto wins = apply_windowing(audio, cfg);
    for (const auto& w : wins)
        EXPECT_EQ(static_cast<int>(w.size()), kWindowSize);
}

TEST_F(WptVoiceBioTest, ApplyWindowing_ShortSignal_ReturnsEmpty)
{
    std::vector<double> tiny(static_cast<size_t>(kWindowSize - 1), 0.0);
    auto wins = apply_windowing(tiny, cfg);
    EXPECT_TRUE(wins.empty());
}

// ============================================================
// interpolate_to_size
// ============================================================

TEST(InterpolateToSize, SameSize_ReturnsSameValues)
{
    std::vector<double> src = {1.0, 2.0, 3.0, 4.0};
    auto out = interpolate_to_size(src, 4);
    ASSERT_EQ(out.size(), 4u);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(out[i], static_cast<float>(src[i]), 1e-5F);
}

TEST(InterpolateToSize, ExpandSize_OutputLengthMatches)
{
    std::vector<double> src = {0.0, 1.0, 0.0};
    auto out = interpolate_to_size(src, 10);
    EXPECT_EQ(out.size(), 10u);
}

TEST(InterpolateToSize, ZeroOrNegative_ReturnsEmpty)
{
    std::vector<double> src = {1.0, 2.0};
    EXPECT_TRUE(interpolate_to_size(src, 0).empty());
    EXPECT_TRUE(interpolate_to_size(src, -1).empty());
}

// ============================================================
// compute_wpt_energy
// ============================================================

TEST_F(WptVoiceBioTest, WptEnergy_OutputSizeIsNumBands)
{
    int wpt_lvl = compute_wpt_level(kWindowSize, kNumBands);
    auto w = apply_windowing(audio, cfg);
    ASSERT_FALSE(w.empty());
    auto energy = compute_wpt_energy(w[0], kNumBands, wpt_lvl);
    EXPECT_EQ(static_cast<int>(energy.size()), kNumBands);
}

TEST_F(WptVoiceBioTest, WptEnergy_AllFinite)
{
    int wpt_lvl = compute_wpt_level(kWindowSize, kNumBands);
    auto w = apply_windowing(audio, cfg);
    ASSERT_FALSE(w.empty());
    auto energy = compute_wpt_energy(w[0], kNumBands, wpt_lvl);
    for (float v : energy)
        EXPECT_TRUE(std::isfinite(v)) << "Non-finite energy value: " << v;
}

TEST_F(WptVoiceBioTest, WptEnergy_NonNegative)
{
    int wpt_lvl = compute_wpt_level(kWindowSize, kNumBands);
    auto w = apply_windowing(audio, cfg);
    ASSERT_FALSE(w.empty());
    auto energy = compute_wpt_energy(w[0], kNumBands, wpt_lvl);
    for (float v : energy)
        EXPECT_GE(v, 0.0F);
}

// ============================================================
// preprocess_energy
// ============================================================

TEST(PreprocessEnergy, OutputInUnitRange)
{
    std::vector<float> raw = {0.0F, 0.5F, 1.0F, 2.0F, 10.0F};
    auto out = preprocess_energy(raw);
    for (float v : out)
    {
        EXPECT_GE(v, 0.0F);
        EXPECT_LE(v, 1.0F);
    }
}

TEST(PreprocessEnergy, MaxIsOne)
{
    std::vector<float> raw = {1.0F, 2.0F, 3.0F};
    auto out = preprocess_energy(raw);
    float mx = *std::max_element(out.begin(), out.end());
    EXPECT_NEAR(mx, 1.0F, 1e-5F);
}

TEST(PreprocessEnergy, AllZeroInput_AllZeroOutput)
{
    std::vector<float> raw(8, 0.0F);
    auto out = preprocess_energy(raw);
    for (float v : out) EXPECT_FLOAT_EQ(v, 0.0F);
}

// ============================================================
// Integration: build_features-style pipeline + encode_poisson
// ============================================================

TEST_F(WptVoiceBioTest, Pipeline_EncodePoisson_SpikesAreBinary)
{
    int wpt_lvl = compute_wpt_level(kWindowSize, kNumBands);
    cfg.wpt_level = wpt_lvl;
    auto windows = apply_windowing(audio, cfg);
    ASSERT_FALSE(windows.empty());

    auto energy = compute_wpt_energy(windows[0], kNumBands, wpt_lvl);
    auto normed = preprocess_energy(energy);

    // Pack into a 1×num_bands Tensor
    nn::Tensor frame(1, static_cast<size_t>(kNumBands));
    for (size_t j = 0; j < static_cast<size_t>(kNumBands); ++j)
        frame.at(0, j) = normed[j];

    std::mt19937 rng(42);
    nn::Tensor spikes = codificacao::encode_poisson(frame, /*steps=*/10, rng);

    for (size_t i = 0; i < spikes.rows(); ++i)
        for (size_t j = 0; j < spikes.cols(); ++j)
        {
            float v = spikes.at(i, j);
            EXPECT_TRUE(v == 0.0F || v == 1.0F);
        }
}
