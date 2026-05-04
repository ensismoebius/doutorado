/**
 * @file audio_loader_gtest.cpp
 * @brief Unit tests for `nn::dataLoaders::loadAudioFromMat()`.
 */

#include <gtest/gtest.h>
#include <matioCpp/File.h>
#include <matioCpp/MultiDimensionalArray.h>
#include <unistd.h> // For getpid()

#include <filesystem>
#include <tuple>

#include "nn/dataLoaders/10.1117/loaders/AudioLoader.h"
#include "nn/dataLoaders/10.1117/schema/METADATA.hpp"
#include "nn/dataLoaders/10.1117/schema/NAMES.hpp"

namespace
{
// Constants for test data dimensions and indices
constexpr size_t kNumRows = 2;
constexpr size_t kNumCols = nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioTotalColumns();
constexpr size_t kNumAudioSamples = nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples();
constexpr size_t kStimulusCol = nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioStimulusColumn();
constexpr size_t kEEGIndexCol = nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioEEGIndexColumn();
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
        testFile = std::filesystem::temp_directory_path().string() + "/" +
                   std::string(kTestFileName) + "." + std::to_string(getpid());

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
            nn::dataLoaders::kAudioMatVariableName, {kNumRows, kNumCols}, data.data());
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
    EXPECT_EQ(audioSamples.size(), nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples());

    // Basic sanity checks
    EXPECT_FALSE(audioSamples.hasNaN());
    EXPECT_EQ(audioStimulus, 1);
    EXPECT_EQ(eegIndex, 42);
    EXPECT_NEAR(audioSamples.at(0, 0), 0.0, 1e-7);
    EXPECT_NEAR(audioSamples.at(1, 0), 0.001, 1e-7);
    EXPECT_NEAR(audioSamples.at(999, 0), 0.999, 1e-7);
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

TEST_F(AudioLoaderTest, SessionSupportsRowsFlatAndMetadata)
{
    nn::dataLoaders::AudioMatSession session(testFile);

    EXPECT_EQ(session.filePath(), testFile);
    EXPECT_EQ(session.rowCount(), kNumRows);

    auto rows = session.readRows(0, kNumRows);
    ASSERT_EQ(rows.size(), kNumRows);

    auto [samples0, stimulus0, eeg0] = rows[0];
    EXPECT_EQ(samples0.rows(), static_cast<int>(kNumAudioSamples));
    EXPECT_EQ(samples0.cols(), 1);
    EXPECT_EQ(stimulus0, 1);
    EXPECT_EQ(eeg0, 42);

    auto flat = session.readRowsFlat(0, kNumRows);
    EXPECT_EQ(flat.samples.size(), kNumRows * kNumAudioSamples);
    ASSERT_EQ(flat.stimuli.size(), kNumRows);
    ASSERT_EQ(flat.eegIndices.size(), kNumRows);
    EXPECT_EQ(flat.stimuli[0], 1);
    EXPECT_EQ(flat.stimuli[1], 2);
    EXPECT_EQ(flat.eegIndices[0], 42);
    EXPECT_EQ(flat.eegIndices[1], 43);
}

TEST_F(AudioLoaderTest, SessionHandlesCacheAndRangeValidation)
{
    nn::dataLoaders::AudioMatSession session(testFile);

    auto first = session.readRows(0, 2);
    auto second = session.readRows(0, 2);
    ASSERT_EQ(first.size(), second.size());

    auto [s1, stimulus1, eeg1] = first[1];
    auto [s2, stimulus2, eeg2] = second[1];
    EXPECT_EQ(stimulus1, stimulus2);
    EXPECT_EQ(eeg1, eeg2);
    EXPECT_EQ(s1.rows(), s2.rows());

    auto empty_rows = session.readRows(0, 0);
    EXPECT_TRUE(empty_rows.empty());

    auto empty_flat = session.readRowsFlat(0, 0);
    EXPECT_TRUE(empty_flat.samples.empty());
    EXPECT_TRUE(empty_flat.stimuli.empty());
    EXPECT_TRUE(empty_flat.eegIndices.empty());

    EXPECT_THROW((void) session.readRows(1, 2), std::runtime_error);
    EXPECT_THROW((void) session.readRowsFlat(1, 2), std::runtime_error);
}

TEST(AudioLoaderStandaloneTest, SessionConstructorRejectsMissingVariable)
{
    const auto path =
        std::filesystem::temp_directory_path().string() + "/audio_missing_var_test.mat";
    std::filesystem::remove(path);

    std::vector<double> data(4, 1.0);
    matioCpp::File file = matioCpp::File::Create(path);
    matioCpp::MultiDimensionalArray<double> wrong_var("NotAudio", {2, 2}, data.data());
    file.write(wrong_var);
    file.close();

    EXPECT_THROW((void) nn::dataLoaders::AudioMatSession(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(AudioLoaderStandaloneTest, SessionConstructorRejectsWrongDimensions)
{
    const auto path = std::filesystem::temp_directory_path().string() + "/audio_bad_dims_test.mat";
    std::filesystem::remove(path);

    constexpr size_t rows = 2;
    constexpr size_t cols = kNumCols - 1;
    std::vector<double> data(rows * cols, 0.0);

    matioCpp::File file = matioCpp::File::Create(path);
    matioCpp::MultiDimensionalArray<double> audio_data(
        nn::dataLoaders::kAudioMatVariableName, {rows, cols}, data.data());
    file.write(audio_data);
    file.close();

    EXPECT_THROW((void) nn::dataLoaders::AudioMatSession(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(AudioLoaderStandaloneTest, SessionConstructorRejectsWrongType)
{
    const auto path = std::filesystem::temp_directory_path().string() + "/audio_bad_type_test.mat";
    std::filesystem::remove(path);

    constexpr size_t rows = 2;
    constexpr size_t cols = kNumCols;
    std::vector<float> data(rows * cols, 0.0F);

    matioCpp::File file = matioCpp::File::Create(path);
    matioCpp::MultiDimensionalArray<float> audio_data(
        nn::dataLoaders::kAudioMatVariableName, {rows, cols}, data.data());
    file.write(audio_data);
    file.close();

    EXPECT_THROW((void) nn::dataLoaders::AudioMatSession(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(AudioLoaderStandaloneTest, SessionCacheEvictionPath)
{
    const auto path =
        std::filesystem::temp_directory_path().string() + "/audio_cache_evict_test.mat";
    std::filesystem::remove(path);

    constexpr size_t rows = 12;
    constexpr size_t cols = kNumCols;
    std::vector<double> data(rows * cols, 0.0);
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t i = 0; i < kNumAudioSamples; ++i)
        {
            data[(i * rows) + r] = static_cast<double>((r + i) % 1024U) / 1024.0;
        }
        data[(kStimulusCol * rows) + r] = static_cast<double>((r % 5U) + 1U);
        data[(kEEGIndexCol * rows) + r] = static_cast<double>(r);
    }

    matioCpp::File file = matioCpp::File::Create(path);
    matioCpp::MultiDimensionalArray<double> audio_data(
        nn::dataLoaders::kAudioMatVariableName, {rows, cols}, data.data());
    file.write(audio_data);
    file.close();

    nn::dataLoaders::AudioMatSession session(path);
    for (size_t r = 0; r < rows; ++r)
    {
        auto [samples, stimulus, eeg] = session.readRow(r);
        EXPECT_EQ(samples.rows(), static_cast<int>(kNumAudioSamples));
        EXPECT_EQ(samples.cols(), 1);
        EXPECT_EQ(eeg, static_cast<int>(r));
        EXPECT_EQ(stimulus, static_cast<int>((r % 5U) + 1U));
    }

    auto [samples0, stimulus0, eeg0] = session.readRow(0);
    EXPECT_EQ(samples0.rows(), static_cast<int>(kNumAudioSamples));
    EXPECT_EQ(stimulus0, 1);
    EXPECT_EQ(eeg0, 0);

    std::remove(path.c_str());
}