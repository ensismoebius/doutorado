/**
 * @file mock_dataset_generator_gtest.cpp
 * @brief Tests for deterministic MAT mock generation used by 10.1117 loader tests.
 */

#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <string>

#include "nn/dataLoaders/10.1117/AudioLoader.h"
#include "nn/dataLoaders/10.1117/EEGLoader.h"
#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "utils/MockImaginedSpeechDatasetGenerator.hpp"

namespace
{
class MockDatasetGeneratorTest : public ::testing::Test
{
   protected:
    std::filesystem::path tmp_dir_;

    void SetUp() override
    {
        tmp_dir_ = std::filesystem::temp_directory_path() /
                   ("mock_10_1117_dataset_" + std::to_string(getpid()));
        std::filesystem::remove_all(tmp_dir_);
        std::filesystem::create_directories(tmp_dir_);
    }

    void TearDown() override
    {
        std::filesystem::remove_all(tmp_dir_);
    }
};
} // namespace

TEST_F(MockDatasetGeneratorTest, GenerateDatasetProducesLoaderCompatibleFiles)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
    nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateDataset(tmp_dir_, 3U);

    const auto eeg_path = tmp_dir_ / "eeg.mat";
    const auto audio_path = tmp_dir_ / "audio.mat";

    ASSERT_TRUE(std::filesystem::exists(eeg_path));
    ASSERT_TRUE(std::filesystem::exists(audio_path));

    auto [eeg_tensor, eeg_labels] = nn::dataLoaders::loadEEGFromMat(eeg_path.string(), 0U);
    auto [audio_tensor, audio_stimulus, eeg_index] =
        nn::dataLoaders::loadAudioFromMat(audio_path.string(), 0U);

    EXPECT_EQ(eeg_tensor.rows(), schema.eeg_channels);
    EXPECT_EQ(eeg_tensor.cols(), schema.eegSamplesPerChannel());

    EXPECT_EQ(audio_tensor.rows(), schema.audioSamples());
    EXPECT_EQ(audio_tensor.cols(), 1);

    EXPECT_EQ(audio_stimulus, eeg_labels[1]);
    EXPECT_EQ(eeg_index, 0);
}

TEST_F(MockDatasetGeneratorTest, CorruptedEEGWrongColumnsFailsLoader)
{
    const auto eeg_path = tmp_dir_ / "eeg_bad.mat";

    nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateCorruptedEEGMatFile(
        eeg_path, 2U, nn::dataLoaders::test::EEGCorruptionOptions{.wrong_column_count = true});

    EXPECT_THROW((void) nn::dataLoaders::loadEEGFromMat(eeg_path.string(), 0U), std::runtime_error);
}

TEST_F(MockDatasetGeneratorTest, CorruptedAudioMissingLabelsFailsLoader)
{
    const auto audio_path = tmp_dir_ / "audio_bad.mat";

    nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateCorruptedAudioMatFile(
        audio_path, 2U, nn::dataLoaders::test::AudioCorruptionOptions{.missing_labels = true});

    EXPECT_THROW((void) nn::dataLoaders::loadAudioFromMat(audio_path.string(), 0U),
                 std::runtime_error);
}
