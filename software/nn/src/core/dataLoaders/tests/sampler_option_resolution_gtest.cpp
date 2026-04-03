/**
 * @file src/core/dataLoaders/tests/sampler_option_resolution_gtest.cpp
 * @brief Implementation for Sampler option resolution gtest.
 *

 */

#include <gtest/gtest.h>

#include <stdexcept>

#include "nn/dataLoaders/SamplerOptionResolution.hpp"

TEST(SamplerOptionResolutionTest, EmptySamplerTypeUsesLegacyShuffleBehavior)
{
    SamplerOptionSelection selection{};
    selection.sampler_type = "";
    selection.shuffle = false;

    const auto options = resolveDefaultSamplerOptions(selection);
    EXPECT_EQ(options.type, DataLoader::DefaultSamplerType::Sequential);

    selection.shuffle = true;
    const auto shuffled_options = resolveDefaultSamplerOptions(selection);
    EXPECT_EQ(shuffled_options.type, DataLoader::DefaultSamplerType::Random);
}

TEST(SamplerOptionResolutionTest, WeightedSamplerCopiesWeightedArgs)
{
    SamplerOptionSelection selection{};
    selection.sampler_type = "WeIgHtEd";
    selection.weights = {0.2, 0.3, 0.5};
    selection.weighted_num_samples = 10U;

    const auto options = resolveDefaultSamplerOptions(selection);
    EXPECT_EQ(options.type, DataLoader::DefaultSamplerType::WeightedRandom);
    EXPECT_EQ(options.weights.size(), 3);
    EXPECT_DOUBLE_EQ(options.weights[0], 0.2);
    EXPECT_EQ(options.weighted_num_samples, 10U);
}

TEST(SamplerOptionResolutionTest, DistributedSamplerCopiesDistributedArgs)
{
    SamplerOptionSelection selection{};
    selection.sampler_type = "distributed";
    selection.distributed_num_replicas = 4;
    selection.distributed_rank = 2;
    selection.distributed_shuffle = false;
    selection.distributed_drop_last = true;

    const auto options = resolveDefaultSamplerOptions(selection);
    EXPECT_EQ(options.type, DataLoader::DefaultSamplerType::Distributed);
    EXPECT_EQ(options.num_replicas, 4U);
    EXPECT_EQ(options.rank, 2U);
    EXPECT_FALSE(options.distributed_shuffle);
    EXPECT_TRUE(options.distributed_drop_last);
}

TEST(SamplerOptionResolutionTest, UnknownSamplerThrows)
{
    SamplerOptionSelection selection{};
    selection.sampler_type = "unsupported";
    EXPECT_THROW((void) resolveDefaultSamplerOptions(selection), std::runtime_error);
}
