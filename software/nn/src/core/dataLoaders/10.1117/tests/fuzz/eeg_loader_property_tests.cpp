#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <random>
#include <string>

#include "nn/dataLoaders/10.1117/EEGLoader.h"
#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "utils/MockImaginedSpeechDatasetGenerator.hpp"

namespace
{
class EEGLoaderPropertyTest : public ::testing::Test
{
   protected:
    std::filesystem::path tmp_dir_;

    void SetUp() override
    {
        tmp_dir_ = std::filesystem::temp_directory_path() /
                   ("eeg_loader_prop_" + std::to_string(getpid()));
        std::filesystem::remove_all(tmp_dir_);
        std::filesystem::create_directories(tmp_dir_);
    }

    void TearDown() override
    {
        std::filesystem::remove_all(tmp_dir_);
    }
};
} // namespace

TEST_F(EEGLoaderPropertyTest, TensorDimensionsAndFlatteningHoldAcrossRandomTrials)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
    std::mt19937 rng(42U);
    std::uniform_int_distribution<std::size_t> trials_dist(1U, 12U);

    for (int round = 0; round < 20; ++round)
    {
        const std::size_t trials = trials_dist(rng);
        const auto eeg_path = tmp_dir_ / ("eeg_prop_" + std::to_string(round) + ".mat");

        nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateEEGMatFile(
            eeg_path, trials);

        for (std::size_t i = 0; i < trials; ++i)
        {
            auto [eeg_tensor, labels] = nn::dataLoaders::loadEEGFromMat(eeg_path.string(), i);
            (void) labels;

            EXPECT_EQ(eeg_tensor.rows(), schema.eeg_channels);
            EXPECT_EQ(eeg_tensor.cols(), schema.eegSamplesPerChannel());

            const std::size_t flattened = static_cast<std::size_t>(eeg_tensor.rows()) *
                                          static_cast<std::size_t>(eeg_tensor.cols());
            EXPECT_EQ(flattened, schema.eegSignalColumns());
        }
    }
}

TEST_F(EEGLoaderPropertyTest, LoadingSameTrialTwiceIsDeterministic)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
    const auto eeg_path = tmp_dir_ / "eeg_deterministic.mat";

    nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateEEGMatFile(eeg_path, 6U);

    for (std::size_t trial = 0; trial < 6U; ++trial)
    {
        auto [a, labels_a] = nn::dataLoaders::loadEEGFromMat(eeg_path.string(), trial);
        auto [b, labels_b] = nn::dataLoaders::loadEEGFromMat(eeg_path.string(), trial);

        ASSERT_EQ(a.rows(), b.rows());
        ASSERT_EQ(a.cols(), b.cols());

        EXPECT_EQ(labels_a[0], labels_b[0]);
        EXPECT_EQ(labels_a[1], labels_b[1]);
        EXPECT_EQ(labels_a[2], labels_b[2]);

        for (std::size_t ch = 0; ch < schema.eeg_channels; ++ch)
        {
            for (std::size_t s = 0; s < schema.eegSamplesPerChannel(); ++s)
            {
                EXPECT_FLOAT_EQ(a.at(ch, s), b.at(ch, s));
            }
        }
    }
}
