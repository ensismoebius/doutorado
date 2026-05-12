/**
 * @file src/core/dataLoaders/10.1117/tests/fuzz/audio_loader_property_tests.cpp
 * @brief Implementation for Audio loader property tests.
 *

 */

#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <random>
#include <string>

#include "dataLoaders/10.1117/loaders/AudioLoader.h"
#include "dataLoaders/10.1117/loaders/EEGLoader.h"
#include "dataLoaders/10.1117/schema/METADATA.hpp"
#include "utils/MockImaginedSpeechDatasetGenerator.hpp"

namespace
{
class AudioLoaderPropertyTest : public ::testing::Test
{
   protected:
    std::filesystem::path tmp_dir_;

    void SetUp() override
    {
        tmp_dir_ = std::filesystem::temp_directory_path() /
                   ("audio_loader_prop_" + std::to_string(getpid()));
        std::filesystem::remove_all(tmp_dir_);
        std::filesystem::create_directories(tmp_dir_);
    }

    void TearDown() override
    {
        std::filesystem::remove_all(tmp_dir_);
    }
};
} // namespace

TEST_F(AudioLoaderPropertyTest, AudioTensorShapeHoldsAcrossRandomValidDatasets)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
    std::mt19937 rng(42U);
    std::uniform_int_distribution<std::size_t> trials_dist(1U, 16U);

    for (int round = 0; round < 20; ++round)
    {
        const std::size_t trials = trials_dist(rng);
        const auto audio_path = tmp_dir_ / ("audio_prop_" + std::to_string(round) + ".mat");

        nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateAudioMatFile(
            audio_path, trials);

        for (std::size_t i = 0; i < trials; ++i)
        {
            auto [audio_tensor, stimulus, eeg_idx] =
                nn::dataLoaders::loadAudioFromMat(audio_path.string(), i);
            (void) stimulus;
            (void) eeg_idx;

            EXPECT_EQ(audio_tensor.rows(), schema.audioSamples());
            EXPECT_EQ(audio_tensor.cols(), 1);
        }
    }
}

TEST_F(AudioLoaderPropertyTest, LabelConsistencyAndCrossModalityIndexRangeHold)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
    const std::size_t trials = 12U;
    const auto eeg_path = tmp_dir_ / "eeg_cross.mat";
    const auto audio_path = tmp_dir_ / "audio_cross.mat";

    nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateEEGMatFile(eeg_path, trials);
    nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateAudioMatFile(
        audio_path, trials);

    for (std::size_t i = 0; i < trials; ++i)
    {
        auto [audio_tensor, audio_stimulus, eeg_trial_index] =
            nn::dataLoaders::loadAudioFromMat(audio_path.string(), i);
        ASSERT_EQ(audio_tensor.rows(), schema.audioSamples());
        ASSERT_EQ(audio_tensor.cols(), 1);

        ASSERT_GE(eeg_trial_index, 0);
        ASSERT_LT(static_cast<std::size_t>(eeg_trial_index), trials);

        auto [eeg_tensor, eeg_labels] = nn::dataLoaders::loadEEGFromMat(
            eeg_path.string(), static_cast<std::size_t>(eeg_trial_index));

        EXPECT_EQ(eeg_tensor.rows(), schema.eeg_channels);
        EXPECT_EQ(eeg_tensor.cols(), schema.eegSamplesPerChannel());

        EXPECT_EQ(audio_stimulus, eeg_labels[1]);
    }
}

TEST_F(AudioLoaderPropertyTest, LoadingSameTrialTwiceIsDeterministic)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
    const auto audio_path = tmp_dir_ / "audio_deterministic.mat";

    nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateAudioMatFile(audio_path, 6U);

    for (std::size_t trial = 0; trial < 6U; ++trial)
    {
        auto [a, stim_a, idx_a] = nn::dataLoaders::loadAudioFromMat(audio_path.string(), trial);
        auto [b, stim_b, idx_b] = nn::dataLoaders::loadAudioFromMat(audio_path.string(), trial);

        ASSERT_EQ(a.rows(), b.rows());
        ASSERT_EQ(a.cols(), b.cols());
        EXPECT_EQ(stim_a, stim_b);
        EXPECT_EQ(idx_a, idx_b);

        for (std::size_t s = 0; s < schema.audioSamples(); ++s)
        {
            EXPECT_FLOAT_EQ(a.at(s, 0), b.at(s, 0));
        }
    }
}
