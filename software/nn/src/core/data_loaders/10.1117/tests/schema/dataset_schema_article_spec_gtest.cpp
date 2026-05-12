/**
 * @file dataset_schema_article_spec_gtest.cpp
 * @brief Verifies 10.1117 schema constants against article-defined dimensions.
 */

#include <gtest/gtest.h>

#include "data_loaders/10.1117/schema/Metadata.hpp"

TEST(DatasetSchemaArticleSpecTest, EEGDimensionsMatchArticle)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;

    EXPECT_EQ(schema.duration_seconds, 4U);
    EXPECT_EQ(schema.eeg_sampling_rate, 1024U);
    EXPECT_EQ(schema.eeg_channels, 6U);

    EXPECT_EQ(schema.eegSamplesPerChannel(), 4096U);
    EXPECT_EQ(schema.eegSignalColumns(), 24576U);
    EXPECT_EQ(schema.eegTotalColumns(), 24579U);

    EXPECT_EQ(schema.eeg_channels * schema.eegSamplesPerChannel(), schema.eegSignalColumns());

    EXPECT_EQ(schema.eegModeColumn(), 24576U);
    EXPECT_EQ(schema.eegStimulusColumn(), 24577U);
    EXPECT_EQ(schema.eegBlinkColumn(), 24578U);
}

TEST(DatasetSchemaArticleSpecTest, AudioDimensionsMatchArticle)
{
    const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;

    EXPECT_EQ(schema.duration_seconds, 4U);
    EXPECT_EQ(schema.audio_sampling_rate, 44100U);

    EXPECT_EQ(schema.audioSamples(), 176400U);
    EXPECT_EQ(schema.audioTotalColumns(), 176402U);

    EXPECT_EQ(schema.audioStimulusColumn(), 176400U);
    EXPECT_EQ(schema.audioEEGIndexColumn(), 176401U);
}
