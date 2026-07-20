/**
 * @file src/experiments/waveletAE/tests/WaveletAEWavelets_gtest.cpp
 * @brief Implementation for WaveletAEwavelets gtest.
 *

 */

#include <vector>

#include "../WaveletAEWavelets.hpp"
#include "gtest/gtest.h"

TEST(WaveletAEWaveletsTest, RejectsEmptySignal)
{
    std::vector<double> empty_signal;
    EXPECT_THROW((void) get_wavelet_coeffs("Haar", empty_signal, 3), std::invalid_argument);
}

TEST(WaveletAEWaveletsTest, AcceptsNonPowerOfTwoSignalViaPadding)
{
    std::vector<double> signal = {0.2, 0.4, -0.1, 0.7, 0.9, -0.3, 0.6}; // size = 7

    auto result = get_wavelet_coeffs("Haar", signal, 3);

    EXPECT_TRUE(result.packet);
    EXPECT_GT(result.levelsOfTransformation, 0);
    EXPECT_FALSE(result.transformedSignal.empty());
}

TEST(WaveletAEWaveletsTest, FallsBackToHaarForUnknownWavelet)
{
    std::vector<double> signal = {0.2, 0.4, -0.1, 0.7, 0.9, -0.3, 0.6};

    auto fallback_result = get_wavelet_coeffs("UnknownWavelet", signal, 3);
    auto haar_result = get_wavelet_coeffs("Haar", signal, 3);

    EXPECT_EQ(fallback_result.packet, haar_result.packet);
    EXPECT_EQ(fallback_result.levelsOfTransformation, haar_result.levelsOfTransformation);
    ASSERT_EQ(fallback_result.transformedSignal.size(), haar_result.transformedSignal.size());
    for (std::size_t i = 0; i < haar_result.transformedSignal.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(fallback_result.transformedSignal[i], haar_result.transformedSignal[i]);
    }
}
