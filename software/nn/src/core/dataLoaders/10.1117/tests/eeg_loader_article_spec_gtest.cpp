/**
 * @file eeg_loader_article_spec_gtest.cpp
 * @brief Article-spec verification tests for EEG loader behavior.
 */

#include <gtest/gtest.h>
#include <matio.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <string>

#include "nn/dataLoaders/10.1117/EEGLoader.h"
#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "nn/dataLoaders/10.1117/NAMES.hpp"
#include "utils/MockImaginedSpeechDatasetGenerator.hpp"

namespace
{
class EEGLoaderArticleSpecTest : public ::testing::Test
{
   protected:
    std::string eeg_file_;
    std::size_t eeg_rows_ = 3;

    void SetUp() override
    {
        eeg_file_ = std::filesystem::temp_directory_path().string() + "/eeg_loader_article_spec_" +
                    std::to_string(getpid()) + ".mat";

        std::filesystem::remove(eeg_file_);

        nn::dataLoaders::test::MockImaginedSpeechDatasetGenerator::generateEEGMatFile(eeg_file_,
                                                                                      eeg_rows_);
    }

    void TearDown() override
    {
        std::remove(eeg_file_.c_str());
    }
};
} // namespace

TEST_F(EEGLoaderArticleSpecTest, RawMatrixColumnCountMatchesSchema)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;

    mat_t* mat = Mat_Open(eeg_file_.c_str(), MAT_ACC_RDONLY);
    ASSERT_NE(mat, nullptr);

    matvar_t* var = Mat_VarRead(mat, nn::dataLoaders::EEG_MAT_VARIABLE_NAME.c_str());
    ASSERT_NE(var, nullptr);

    EXPECT_EQ(var->rank, 2);
    EXPECT_EQ(var->dims[1], schema.eegTotalColumns());

    Mat_VarFree(var);
    Mat_Close(mat);
}

TEST_F(EEGLoaderArticleSpecTest, ReturnsChannelBySamplesTensorWithExpectedLabels)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
    const std::size_t row = 1;

    auto [eeg_tensor, labels] = nn::dataLoaders::loadEEGFromMat(eeg_file_, row);

    EXPECT_EQ(eeg_tensor.rows(), schema.eeg_channels);
    EXPECT_EQ(eeg_tensor.cols(), schema.eegSamplesPerChannel());
    EXPECT_EQ(schema.eeg_channels * schema.eegSamplesPerChannel(), schema.eegSignalColumns());

    // Generated labels follow deterministic ranges used by the test generator.
    EXPECT_EQ(labels[0], static_cast<int>((row % 5U) + 1U));
    EXPECT_EQ(labels[1], static_cast<int>((row % 5U) + 1U));
    EXPECT_EQ(labels[2], static_cast<int>((row % 10U) + 1U));
}
