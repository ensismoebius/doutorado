
#include "../Wav.h"
#include "../filtersOperations.h"
#include "../simpleSignalOperations.h"
#include "gtest/gtest.h"

TEST(SimpleSignalOperationsTest, TestAMDF)
{
    std::vector<long double> signal = {1.0, 2.0, 3.0, 2.0, 1.0};
    auto result = amdf(signal);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.size(), signal.size());
}

TEST(FiltersOperationsTest, TestCreateAlpha)
{
    double samplingRate = 44100.0;
    double cutoffFrequency = 2000.0;
    auto alpha = createAlpha(samplingRate, cutoffFrequency);
    EXPECT_GT(alpha, 0.0);
    EXPECT_LT(alpha, 1.0);
}

TEST(WavFileTest, TestReadWavFile)
{
    // Create a sample WAV file for testing
    std::string filepath = "test.wav";
    Wav testWav;
    std::vector<long double> testData = {0.0, 0.1, 0.2, 0.3};
    testWav.write(filepath);

    // Test reading the file
    Wav wavFile;
    EXPECT_NO_THROW(wavFile.read(filepath));
    auto data = wavFile.getData();
    EXPECT_FALSE(data.empty());
}

TEST(WavFileTest, TestWriteWavFile)
{
    std::string filepath = "output.wav";
    Wav wavFile;
    std::vector<long double> data = {0.0, 0.1, 0.2, 0.3};
    EXPECT_NO_THROW(wavFile.write(filepath));
    EXPECT_EQ(wavFile.getPath(), filepath);
}