/**
 * @file src/core/dataLoaders/10.1117/tests/codec/batch_target_formatter_gtest.cpp
 * @brief Implementation for Batch target formatter gtest.
 *

 */

#include <gtest/gtest.h>

#include "dataLoaders/10.1117/codec/BatchTargetFormatter.hpp"

TEST(BatchTargetFormatterTest, FormatsKnownLabels)
{
    Batch batch;
    batch.targets = nn::Tensor(2, 5);

    batch.targets.at(0, 0) = 10.0F;
    batch.targets.at(0, 1) = 1.0F;
    batch.targets.at(0, 2) = 2.0F;
    batch.targets.at(0, 3) = 1.0F;
    batch.targets.at(0, 4) = 15.0F;

    batch.targets.at(1, 0) = 11.0F;
    batch.targets.at(1, 1) = 2.0F;
    batch.targets.at(1, 2) = 6.0F;
    batch.targets.at(1, 3) = 2.0F;
    batch.targets.at(1, 4) = 30.0F;

    const std::string formatted = nn::dataLoaders::formatProtocol101117BatchTargets(batch);

    EXPECT_NE(formatted.find("Sample 0:"), std::string::npos);
    EXPECT_NE(formatted.find("Target modality: Imagined"), std::string::npos);
    EXPECT_NE(formatted.find("Target stimulus: E"), std::string::npos);
    EXPECT_NE(formatted.find("Target artifact: No blink"), std::string::npos);

    EXPECT_NE(formatted.find("Sample 1:"), std::string::npos);
    EXPECT_NE(formatted.find("Target modality: Pronounced"), std::string::npos);
    EXPECT_NE(formatted.find("Target stimulus: Arriba"), std::string::npos);
    EXPECT_NE(formatted.find("Target artifact: Blink"), std::string::npos);
}

TEST(BatchTargetFormatterTest, UsesUnknownFallbackForOutOfRangeLabels)
{
    Batch batch;
    batch.targets = nn::Tensor(1, 5);

    batch.targets.at(0, 0) = 20.0F;
    batch.targets.at(0, 1) = 99.0F;
    batch.targets.at(0, 2) = 42.0F;
    batch.targets.at(0, 3) = 8.0F;
    batch.targets.at(0, 4) = 4.0F;

    const std::string formatted = nn::dataLoaders::formatProtocol101117BatchTargets(batch);

    EXPECT_NE(formatted.find("Target modality: Unknown(99)"), std::string::npos);
    EXPECT_NE(formatted.find("Target stimulus: Unknown(42)"), std::string::npos);
    EXPECT_NE(formatted.find("Target artifact: Unknown(8)"), std::string::npos);
}
