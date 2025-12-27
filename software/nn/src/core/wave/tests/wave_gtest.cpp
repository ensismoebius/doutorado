#include <filesystem>

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

TEST(WavFileTest, WriteThenRead)
{
    const std::string filepath = std::filesystem::temp_directory_path().string() + "/output.wav";
    std::vector<float> data = {0.0F, 0.1F, 0.2F, 0.3F};

    // write
    Wav writer;
    ASSERT_NO_THROW(writer.write(filepath, data, 44100));
    ASSERT_EQ(writer.getPath(), filepath);

    // read
    Wav reader;
    ASSERT_NO_THROW(reader.read(filepath)); // flawfinder: ignore
    auto readData = reader.getData();
    ASSERT_FALSE(readData.empty());
    // optional: compare contents (convert types if needed)
}
