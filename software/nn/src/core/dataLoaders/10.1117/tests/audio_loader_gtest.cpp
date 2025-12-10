#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>

#include <filesystem>
#include <unistd.h> // For getpid()

#include "../AudioLoader.h"

namespace
{
// Constants for test data dimensions and indices
constexpr size_t kNumRows = 2;
constexpr size_t kNumCols = 176402;         // total columns in MAT variable (samples + 2 labels)
constexpr size_t kNumAudioSamples = 176400; // number of audio samples per row
constexpr size_t kStimulusCol =
    176400; // column index in MAT for stimulus ID (0-based column index)
constexpr size_t kEEGIndexCol = 176401;     // column index in MAT for EEG index
constexpr size_t kModBase = 1000;           // modulus used to generate sample values
constexpr double kRow0Div = 1000.0;         // divisor for row 0 sample normalization
constexpr double kRow1Div = 500.0;          // divisor for row 1 sample normalization
constexpr size_t kInvalidRowIndex = 999999; // large invalid row index used in tests
constexpr auto kTestFileName = "audio_test.mat";
} // namespace

class AudioLoaderTest : public ::testing::Test
{
   protected:
    std::string testFile;

    void SetUp() override
    {
        testFile = std::string(kTestFileName) + "." + std::to_string(getpid());

        // Remove any leftover file from previous runs
        std::filesystem::remove(testFile);

        // Create test data in column-major order
        std::vector<double> data(kNumRows * kNumCols, 0.0);
        size_t num_rows = kNumRows;

        // Fill data for row 0
        for (size_t i = 0; i < kNumAudioSamples; ++i)
        {
            data[(i * num_rows) + 0] = static_cast<double>(i % kModBase) / kRow0Div;
        }
        data[(kStimulusCol * num_rows) + 0] = 1.0;  // stimulus ID
        data[(kEEGIndexCol * num_rows) + 0] = 42.0; // EEG index

        // Fill data for row 1
        for (size_t i = 0; i < kNumAudioSamples; ++i)
        {
            data[(i * num_rows) + 1] = static_cast<double>(i % kModBase) / kRow1Div;
        }
        data[(kStimulusCol * num_rows) + 1] = 2.0;  // stimulus ID
        data[(kEEGIndexCol * num_rows) + 1] = 43.0; // EEG index

        // Create the MAT file and write the data
        matioCpp::File file = matioCpp::File::Create(testFile);

        // Create the variable and write it to the file
        matioCpp::MultiDimensionalArray<double> audio_data(
            nn::dataLoaders::AUDIO_VARIABLE_NAME, {2, 176402}, data.data());
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
    auto [audioSamples, audioStimulus, eegIndex] = nn::dataLoaders::loadAudioFromMat(testFile, 0);

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
    EXPECT_THROW(nn::dataLoaders::loadAudioFromMat(testFile, kInvalidRowIndex), // Very large index
                 std::runtime_error);
}