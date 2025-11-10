#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>

#include <filesystem>

#include "dataLoaders/AudioLoader.h"

class AudioLoaderTest : public ::testing::Test
{
   protected:
    std::string testFile;

    void SetUp() override
    {
        testFile = "audio_test.mat";

        // Remove any leftover file from previous runs
        std::filesystem::remove(testFile);

        // Create test data
        std::vector<double> data(2 * 176402, 0.0);

        // Fill first row with test audio data
        for (size_t i = 0; i < 176400; ++i)
        {
            data[i] = static_cast<double>(i % 1000) / 1000.0;
        }
        data[176400] = 1.0;  // stimulus ID
        data[176401] = 42.0; // EEG index

        // Fill second row
        for (size_t i = 176402; i < 176402 + 176400; ++i)
        {
            data[i] = static_cast<double>((i - 176402) % 1000) / 500.0;
        }
        data[176402 + 176400] = 2.0;  // stimulus ID
        data[176402 + 176401] = 43.0; // EEG index

        // Create the MAT file and write the data
        matioCpp::File file = matioCpp::File::Create(testFile);

        // Create the variable and write it to the file
        matioCpp::MultiDimensionalArray<double> audio_data("audio_data", {2, 176402}, data.data());
        file.write(audio_data);
        file.close();
    }

    void TearDown() override
    {
        // Clean up the test file
        std::remove(testFile.c_str());
    }
};

TEST_F(AudioLoaderTest, LoadsAudioDataCorrectly)
{
    // Load the first row of audio data
    auto [audioSamples, eegIndex] = nn::dataLoaders::loadAudioFromMat(testFile, 0);

    // Check dimensions
    EXPECT_EQ(audioSamples.size(), 176400);

    // Basic sanity checks
    EXPECT_FALSE(audioSamples.hasNaN());
    EXPECT_GE(eegIndex, 0); // EEG index should be non-negative
}

TEST_F(AudioLoaderTest, ThrowsOnInvalidFile)
{
    EXPECT_THROW(nn::dataLoaders::loadAudioFromMat("nonexistent.mat"), std::runtime_error);
}

TEST_F(AudioLoaderTest, ThrowsOnInvalidRowIndex)
{
    EXPECT_THROW(nn::dataLoaders::loadAudioFromMat(testFile, 999999), // Very large index
                 std::runtime_error);
}