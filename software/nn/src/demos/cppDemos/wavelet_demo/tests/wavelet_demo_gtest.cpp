/**
 * @file wavelet_demo_gtest.cpp
 * @brief Unit tests for wavelet_demo core logic: generateSignal and wavelet transforms.
 *
 * Only the signal generation and wavelet computation are tested here.
 * The matplotlib plotting functions are NOT called from tests.
 */

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "wavelet/Types.hpp"
#include "wavelet/WaveletTransformResults.hpp"
#include "wavelet/waveletOperations.hpp"

// Replicated from wavelet_demo.cpp (no external linkage — only in the demo main())
static auto generateSignal(
    double freq1, double freq2, size_t sample_rate, size_t duration_seconds) -> std::vector<double>
{
    int signal_length = static_cast<int>(duration_seconds * sample_rate);
    std::vector<double> signal(static_cast<size_t>(signal_length));
    for (int n = 0; n < signal_length; ++n)
    {
        signal[static_cast<size_t>(n)] =
            (0.7 * std::sin((2.0 * M_PI * freq1 * static_cast<double>(n)) /
                            static_cast<double>(sample_rate))) +
            (0.3 * std::sin((2.0 * M_PI * freq2 * static_cast<double>(n)) /
                            static_cast<double>(sample_rate)));
    }
    return signal;
}

class WaveletDemoTest : public ::testing::Test
{
   protected:
    static constexpr size_t kSampleRate = 1024;
    static constexpr size_t kDuration = 1;
    static constexpr double kFreq1 = 50.0;
    static constexpr double kFreq2 = 120.0;

    std::vector<double> signal;
    std::span<const double> db8_filter;

    void SetUp() override
    {
        signal = generateSignal(kFreq1, kFreq2, kSampleRate, kDuration);
        db8_filter = wavelets::get_wavelet<wavelets::Daub8>();
    }
};

// ---- generateSignal tests ----

TEST_F(WaveletDemoTest, GenerateSignal_LengthCorrect)
{
    EXPECT_EQ(signal.size(), kSampleRate * kDuration);
}

TEST_F(WaveletDemoTest, GenerateSignal_NotConstant)
{
    // Signal must oscillate — at least two distinct values
    bool found_diff = false;
    for (size_t i = 1; i < signal.size(); ++i)
    {
        if (std::abs(signal[i] - signal[0]) > 1e-9)
        {
            found_diff = true;
            break;
        }
    }
    EXPECT_TRUE(found_diff);
}

TEST_F(WaveletDemoTest, GenerateSignal_AmplitudeBounded)
{
    for (double v : signal)
    {
        EXPECT_LE(std::abs(v), 1.0 + 1e-9);
    }
}

// ---- DWT (REGULAR_WAVELET) tests ----

TEST_F(WaveletDemoTest, DWT_Level1_OutputNonEmpty)
{
    auto result = wavelets::malat(signal, db8_filter, wavelets::REGULAR_WAVELET, 1);
    // Level 1: approximation (index 0) + detail (index 1) should both be non-empty
    EXPECT_FALSE(result.get_wavelet_transforms(0).empty());
    EXPECT_FALSE(result.get_wavelet_transforms(1).empty());
}

TEST_F(WaveletDemoTest, DWT_LevelsUpToFour_EachLevelNonEmpty)
{
    for (unsigned int lvl = 1; lvl <= 4; ++lvl)
    {
        auto result = wavelets::malat(signal, db8_filter, wavelets::REGULAR_WAVELET, lvl);
        // Approximation at index 0 must be non-empty
        EXPECT_FALSE(result.get_wavelet_transforms(0).empty())
            << "Approximation empty at level " << lvl;
        // Each detail level must be non-empty
        for (unsigned int d = 1; d <= lvl; ++d)
        {
            EXPECT_FALSE(result.get_wavelet_transforms(d).empty())
                << "Detail " << d << " empty at level " << lvl;
        }
    }
}

TEST_F(WaveletDemoTest, DWT_CoefficientsAreFinite)
{
    auto result = wavelets::malat(signal, db8_filter, wavelets::REGULAR_WAVELET, 3);
    for (unsigned int i = 0; i <= 3; ++i)
    {
        for (double v : result.get_wavelet_transforms(i))
        {
            EXPECT_TRUE(std::isfinite(v)) << "Non-finite coefficient at subband " << i;
        }
    }
}

// ---- DWPT (PACKET_WAVELET) tests ----

TEST_F(WaveletDemoTest, DWPT_Level1_PacketCountIsTwo)
{
    auto result = wavelets::malat(signal, db8_filter, wavelets::PACKET_WAVELET, 1);
    long n_parts = result.get_wavelet_packet_amount_of_parts();
    EXPECT_EQ(n_parts, 2L); // 2^1 = 2
}

TEST_F(WaveletDemoTest, DWPT_Level2_PacketCountIsFour)
{
    auto result = wavelets::malat(signal, db8_filter, wavelets::PACKET_WAVELET, 2);
    long n_parts = result.get_wavelet_packet_amount_of_parts();
    EXPECT_EQ(n_parts, 4L); // 2^2 = 4
}

TEST_F(WaveletDemoTest, DWPT_Level3_PacketCountIsEight)
{
    auto result = wavelets::malat(signal, db8_filter, wavelets::PACKET_WAVELET, 3);
    long n_parts = result.get_wavelet_packet_amount_of_parts();
    EXPECT_EQ(n_parts, 8L); // 2^3 = 8
}

TEST_F(WaveletDemoTest, DWPT_Level4_PacketCountIsSixteen)
{
    auto result = wavelets::malat(signal, db8_filter, wavelets::PACKET_WAVELET, 4);
    long n_parts = result.get_wavelet_packet_amount_of_parts();
    EXPECT_EQ(n_parts, 16L); // 2^4 = 16
}

TEST_F(WaveletDemoTest, DWPT_Level2_PacketCoefficientsAreFinite)
{
    auto result = wavelets::malat(signal, db8_filter, wavelets::PACKET_WAVELET, 2);
    long n_parts = result.get_wavelet_packet_amount_of_parts();
    for (long i = 0; i < n_parts; ++i)
    {
        auto part = wavelets::WaveletTransformResults::get_wavelet_packet_transforms(
            result.transformedSignal, i, result.levelsOfTransformation);
        EXPECT_FALSE(part.empty()) << "Packet " << i << " is empty";
        for (double v : part)
        {
            EXPECT_TRUE(std::isfinite(v)) << "Non-finite in packet " << i;
        }
    }
}
