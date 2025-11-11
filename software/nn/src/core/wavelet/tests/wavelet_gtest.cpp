#include <cmath>

#include "../Types.h"
#include "../WaveletTransformResults.h"
#include "../waveletOperations.h"
#include "gtest/gtest.h"

namespace wavelets::test
{

TEST(WaveletTypesTest, TestInit)
{
    wavelets::init();
    auto allWavelets = wavelets::all();
    EXPECT_FALSE(allWavelets.empty());

    // Test with specific wavelets
    std::vector<std::string> chosenWavelets = {"haar", "db1"};
    wavelets::init(chosenWavelets);
    auto specificWavelets = wavelets::all();
    EXPECT_LE(specificWavelets.size(), chosenWavelets.size());
}

TEST(WaveletTypesTest, TestGetWavelet)
{
    wavelets::init();
    auto haarWavelet = wavelets::get("haar");
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
    std::vector<long double> signal = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    wavelets::init();
    auto haarFilter = wavelets::get("haar");

    auto result = wavelets::malat(signal, haarFilter, REGULAR_WAVELET, 1);
    EXPECT_FALSE(result.transformedSignal.empty());
    EXPECT_EQ(result.levelsOfTransformation, 1U);
    EXPECT_FALSE(result.packet);
}

TEST(WaveletOperationsTest, TestMalatPacketTransform)
{
    std::vector<long double> signal = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    wavelets::init();
    auto haarFilter = wavelets::get("haar");

    auto result = wavelets::malat(signal, haarFilter, PACKET_WAVELET, 1);
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

} // namespace wavelets::test
