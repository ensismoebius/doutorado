/**
 * @file audio_loader_article_spec_gtest.cpp
 * @brief Article-spec verification tests for audio loader behavior.
 */

#include <gtest/gtest.h>
#include <matio.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <string>

#include "data_loaders/10.1117/loaders/AudioLoader.hpp"
#include "data_loaders/10.1117/loaders/EEGLoader.hpp"
#include "data_loaders/10.1117/schema/Metadata.hpp"
#include "data_loaders/10.1117/schema/Names.hpp"
#include "utils/MockImaginedSpeechDatasetGenerator.hpp"

namespace
{
class AudioLoaderArticleSpecTest : public ::testing::Test
{
   protected:
    std::string audio_file_;
    std::string eeg_file_;
    std::size_t audio_rows_ = 4;
    std::size_t eeg_rows_ = 4;

    void SetUp() override
    {
        const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;

        audio_file_ = std::filesystem::temp_directory_path().string() +
                      "/audio_loader_article_spec_" + std::to_string(getpid()) + ".mat";
        eeg_file_ = std::filesystem::temp_directory_path().string() +
                    "/audio_loader_article_spec_ref_eeg_" + std::to_string(getpid()) + ".mat";

        std::filesystem::remove(audio_file_);
        std::filesystem::remove(eeg_file_);

        (void) schema;
        nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateEEGMatFile(
            eeg_file_, eeg_rows_);
        nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateAudioMatFile(
            audio_file_, audio_rows_);
    }

    void TearDown() override
    {
        std::remove(audio_file_.c_str());
        std::remove(eeg_file_.c_str());
    }
};
} // namespace

TEST_F(AudioLoaderArticleSpecTest, RawMatrixColumnCountMatchesSchema)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;

    mat_t* mat = Mat_Open(audio_file_.c_str(), MAT_ACC_RDONLY);
    ASSERT_NE(mat, nullptr);

    matvar_t* var = Mat_VarRead(mat, nn::dataLoaders::kAudioMatVariableName.c_str());
    ASSERT_NE(var, nullptr);

    EXPECT_EQ(var->rank, 2);
    EXPECT_EQ(var->dims[1], schema.audioTotalColumns());

    Mat_VarFree(var);
    Mat_Close(mat);
}

TEST_F(AudioLoaderArticleSpecTest, ReturnsExpectedTensorShapeAndLabelColumns)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
    const std::size_t row = 2;

    auto [audio_tensor, stimulus, eeg_index] = nn::dataLoaders::loadAudioFromMat(audio_file_, row);

    EXPECT_EQ(audio_tensor.rows(), schema.audioSamples());
    EXPECT_EQ(audio_tensor.cols(), 1);

    EXPECT_EQ(stimulus, static_cast<int>((row % 5U) + 1U));
    EXPECT_EQ(eeg_index, static_cast<int>(row % eeg_rows_));
}

TEST_F(AudioLoaderArticleSpecTest, AudioEEGIndexReferencesValidEEGTrial)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;

    for (std::size_t row = 0; row < audio_rows_; ++row)
    {
        auto [audio_tensor, audio_stimulus, eeg_index] =
            nn::dataLoaders::loadAudioFromMat(audio_file_, row);
        (void) audio_tensor;

        ASSERT_EQ(eeg_index, static_cast<int>(row % eeg_rows_));

        auto [eeg_tensor, eeg_labels] =
            nn::dataLoaders::loadEEGFromMat(eeg_file_, static_cast<std::size_t>(eeg_index));
        EXPECT_EQ(eeg_tensor.rows(), schema.eeg_channels);
        EXPECT_EQ(eeg_tensor.cols(), schema.eegSamplesPerChannel());

        // Cross-modal linkage in this fixture enforces matching stimulus labels.
        EXPECT_EQ(audio_stimulus, eeg_labels[1]);
    }
}
