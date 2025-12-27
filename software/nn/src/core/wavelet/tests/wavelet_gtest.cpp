
#include "../Types.h"
#include "../WaveletTransformResults.h"
#include "../waveletOperations.h"
#include "gtest/gtest.h"

namespace wavelets::test
{

TEST(WaveletTypesTest, TestGetWavelet)
{
    auto haarWavelet = wavelets::get_wavelet<wavelets::Haar>();
    EXPECT_FALSE(haarWavelet.empty());
}

TEST(WaveletOperationsTest, TestGetNextPowerOfTwo)
{
    EXPECT_EQ(wavelets::getNextPowerOfTwo(3), 4);
    EXPECT_EQ(wavelets::getNextPowerOfTwo(4), 4);
    EXPECT_EQ(wavelets::getNextPowerOfTwo(5), 8);
    EXPECT_EQ(wavelets::getNextPowerOfTwo(7), 8);
    EXPECT_EQ(wavelets::getNextPowerOfTwo(8), 8);
    EXPECT_EQ(wavelets::getNextPowerOfTwo(9), 16);
}

TEST(WaveletOperationsTest, TestMalatRegularTransform)
{
    std::vector<double> signal(64);
    for (size_t i = 0; i < 16; ++i)
    {
        signal[i] = static_cast<double>(i + 1);
    }
    auto haarFilter = wavelets::get_wavelet<wavelets::Haar>();

    auto result = wavelets::malat(signal, haarFilter, wavelets::REGULAR_WAVELET, 1);
    EXPECT_FALSE(result.transformedSignal.empty());
    EXPECT_EQ(result.levelsOfTransformation, 1U);
    EXPECT_FALSE(result.packet);
}

TEST(WaveletOperationsTest, TestMalatPacketTransform)
{
    std::vector<double> signal(64);
    for (size_t i = 0; i < 16; ++i)
    {
        signal[i] = static_cast<double>(i + 1);
    }
    auto haarFilter = wavelets::get_wavelet<wavelets::Haar>();

    auto result = wavelets::malat(signal, haarFilter, wavelets::PACKET_WAVELET, 1);
    EXPECT_FALSE(result.transformedSignal.empty());
    EXPECT_EQ(result.levelsOfTransformation, 1U);
    EXPECT_TRUE(result.packet);
}

TEST(WaveletTransformResultsTest, TestExtraction)
{
    WaveletTransformResults results;
    results.levelsOfTransformation = 2;
    results.transformedSignal = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    // Test whole signal extraction
    auto wholeSignal = results.getWaveletTransforms(-1);
    EXPECT_EQ(wholeSignal, results.transformedSignal);

    // Test approximation extraction
    auto approximation = results.getWaveletTransforms(0);
    EXPECT_FALSE(approximation.empty());

    // Test detail extraction
    auto detail1 = results.getWaveletTransforms(1);
    EXPECT_FALSE(detail1.empty());
}

TEST(WaveletTransformResultsTest, TestMaxItems)
{
    const unsigned int maxItems = 4;
    WaveletTransformResults results(maxItems);
    results.transformedSignal = {1.0, 2.0, 3.0, 4.0};

    // Should not throw when within maxItems
    EXPECT_NO_THROW(results.getWaveletTransforms(-1));

    // Adding more items should throw
    results.transformedSignal.push_back(5.0);
    EXPECT_THROW(results.getWaveletTransforms(-1), std::runtime_error);
}

TEST(WaveletOperationsTest, TestMalatHaarCorrectness)
{
    // Test with a simple signal where we know the expected output
    std::vector<double> signal = {1.0, 1.0, 1.0, 1.0};
    auto haarFilter = wavelets::get_wavelet<wavelets::Haar>();

    auto result = wavelets::malat(signal, haarFilter, wavelets::REGULAR_WAVELET, 1);
    EXPECT_EQ(result.transformedSignal.size(), 4);

    // Haar wavelet coefficients are approximately 1/sqrt(2) ≈ 0.707106781
    // But in the code, they are 0.7071067
    double coeff = 0.7071067;
    double expected_low = (1.0 * coeff) + (1.0 * coeff);     // 1.4142134
    double expected_high = (1.0 * coeff) + (1.0 * (-coeff)); // 0

    EXPECT_NEAR(result.transformedSignal[0], expected_low, 1e-6);
    EXPECT_NEAR(result.transformedSignal[1], expected_low, 1e-6);
    EXPECT_NEAR(result.transformedSignal[2], expected_high, 1e-6);
    EXPECT_NEAR(result.transformedSignal[3], expected_high, 1e-6);
}

TEST(WaveletOperationsTest, TestMalatEnergyConservation)
{
    // Test energy conservation for Haar wavelet (orthogonal)
    std::vector<double> signal(64);
    for (size_t i = 0; i < 16; ++i)
    {
        signal[i] = static_cast<double>(i + 1);
    }
    auto haarFilter = wavelets::get_wavelet<wavelets::Haar>();

    auto result = wavelets::malat(signal, haarFilter, wavelets::REGULAR_WAVELET, 1);

    // Calculate energy of original signal
    double original_energy = 0.0;
    for (double val : signal)
    {
        original_energy += val * val;
    }

    // Calculate energy of transformed signal
    double transformed_energy = 0.0;
    for (double val : result.transformedSignal)
    {
        transformed_energy += val * val;
    }

    // For orthogonal wavelets, energy should be preserved (within floating point precision)
    EXPECT_NEAR(original_energy, transformed_energy, 1e-3);
}

TEST(WaveletOperationsTest, TestMalatDaub4EnergyConservation)
{
    std::vector<double> signal(64);
    for (size_t i = 0; i < 16; ++i)
    {
        signal[i] = static_cast<double>(i + 1);
    }
    auto daub4Filter = wavelets::get_wavelet<wavelets::Daub4>();

    auto result = wavelets::malat(signal, daub4Filter, wavelets::REGULAR_WAVELET, 1);

    double original_energy = 0.0;
    for (double val : signal)
    {
        original_energy += val * val;
    }

    double transformed_energy = 0.0;
    for (double val : result.transformedSignal)
    {
        transformed_energy += val * val;
    }

    EXPECT_NEAR(original_energy, transformed_energy, 1e-4);
}

TEST(WaveletOperationsTest, TestMalatDaub6EnergyConservation)
{
    std::vector<double> signal(64);
    for (size_t i = 0; i < 16; ++i)
    {
        signal[i] = static_cast<double>(i + 1);
    }
    auto daub6Filter = wavelets::get_wavelet<wavelets::Daub6>();

    auto result = wavelets::malat(signal, daub6Filter, wavelets::REGULAR_WAVELET, 1);

    double original_energy = 0.0;
    for (double val : signal)
    {
        original_energy += val * val;
    }

    double transformed_energy = 0.0;
    for (double val : result.transformedSignal)
    {
        transformed_energy += val * val;
    }

    EXPECT_NEAR(original_energy, transformed_energy, 1e-4);
}

TEST(WaveletOperationsTest, TestMalatDaub8EnergyConservation)
{
    std::vector<double> signal(64);
    for (size_t i = 0; i < 16; ++i)
    {
        signal[i] = static_cast<double>(i + 1);
    }
    auto daub8Filter = wavelets::get_wavelet<wavelets::Daub8>();

    auto result = wavelets::malat(signal, daub8Filter, wavelets::REGULAR_WAVELET, 1);

    double original_energy = 0.0;
    for (double val : signal)
    {
        original_energy += val * val;
    }

    double transformed_energy = 0.0;
    for (double val : result.transformedSignal)
    {
        transformed_energy += val * val;
    }

    EXPECT_NEAR(original_energy, transformed_energy, 1e-4);
}

TEST(WaveletOperationsTest, TestMalatHaarPacketEnergyConservation)
{
    std::vector<double> signal(64);
    for (size_t i = 0; i < 16; ++i)
    {
        signal[i] = static_cast<double>(i + 1);
    }
    auto haarFilter = wavelets::get_wavelet<wavelets::Haar>();

    auto result = wavelets::malat(signal, haarFilter, wavelets::PACKET_WAVELET, 1);

    double original_energy = 0.0;
    for (double val : signal)
    {
        original_energy += val * val;
    }

    double transformed_energy = 0.0;
    for (double val : result.transformedSignal)
    {
        transformed_energy += val * val;
    }

    EXPECT_NEAR(original_energy, transformed_energy, 1e-3);
}

TEST(WaveletOperationsTest, TestMalatDaub4PacketEnergyConservation)
{
    std::vector<double> signal(64);
    for (size_t i = 0; i < 16; ++i)
    {
        signal[i] = static_cast<double>(i + 1);
    }
    auto daub4Filter = wavelets::get_wavelet<wavelets::Daub4>();

    auto result = wavelets::malat(signal, daub4Filter, wavelets::PACKET_WAVELET, 1);

    double original_energy = 0.0;
    for (double val : signal)
    {
        original_energy += val * val;
    }

    double transformed_energy = 0.0;
    for (double val : result.transformedSignal)
    {
        transformed_energy += val * val;
    }

    EXPECT_NEAR(original_energy, transformed_energy, 1e-4);
}

TEST(WaveletOperationsTest, TestMalatDaub6PacketEnergyConservation)
{
    std::vector<double> signal(64);
    for (size_t i = 0; i < 16; ++i)
    {
        signal[i] = static_cast<double>(i + 1);
    }
    auto daub6Filter = wavelets::get_wavelet<wavelets::Daub6>();

    auto result = wavelets::malat(signal, daub6Filter, wavelets::PACKET_WAVELET, 1);

    double original_energy = 0.0;
    for (double val : signal)
    {
        original_energy += val * val;
    }

    double transformed_energy = 0.0;
    for (double val : result.transformedSignal)
    {
        transformed_energy += val * val;
    }

    EXPECT_NEAR(original_energy, transformed_energy, 1e-4);
}

TEST(WaveletOperationsTest, TestMalatDaub8PacketEnergyConservation)
{
    std::vector<double> signal(64);
    for (size_t i = 0; i < 16; ++i)
    {
        signal[i] = static_cast<double>(i + 1);
    }
    auto daub8Filter = wavelets::get_wavelet<wavelets::Daub8>();

    auto result = wavelets::malat(signal, daub8Filter, wavelets::PACKET_WAVELET, 1);

    double original_energy = 0.0;
    for (double val : signal)
    {
        original_energy += val * val;
    }

    double transformed_energy = 0.0;
    for (double val : result.transformedSignal)
    {
        transformed_energy += val * val;
    }

    EXPECT_NEAR(original_energy, transformed_energy, 1e-4);
}

} // namespace wavelets::test
