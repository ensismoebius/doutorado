/**
 * @file DatasetBuilder_gtest.cpp
 * @brief Unit tests for the Experiment03 DatasetBuilder.
 */

#include <gtest/gtest.h>

#include "DatasetBuilder.hpp"
#include "cli.hpp"

TEST(DatasetBuilder, BuildsEegWindowDatasetWithEmptyDiscovery)
{
    Config cfg;
    cfg.dataset_type = Experiment03DatasetType::EegWindow;

    std::vector<SubjectFiles> discovered; // empty discovery

    auto ds = experiment03::DatasetBuilder().with_discovered(discovered).with_config(cfg).build();

    ASSERT_NE(ds, nullptr);
    EXPECT_EQ(ds->size(), 0u);
}
