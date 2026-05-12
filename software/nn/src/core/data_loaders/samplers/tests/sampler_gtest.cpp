/**
 * @file src/core/dataLoaders/samplers/tests/sampler_gtest.cpp
 * @brief Implementation for Sampler gtest.
 *

 */

#include <algorithm>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "data_loaders/samplers/DistributedSampler.hpp"
#include "data_loaders/samplers/FoldSampler.hpp"
#include "data_loaders/samplers/RandomSampler.hpp"
#include "data_loaders/samplers/SequentialSampler.hpp"
#include "data_loaders/samplers/WeightedRandomSampler.hpp"

TEST(SamplerTest, SequentialSamplerProducesOrderedIndices)
{
    SequentialSampler sampler(6);
    std::vector<std::size_t> out(sampler.index_count());
    sampler.sample_into(out);

    const std::vector<std::size_t> expected = {0, 1, 2, 3, 4, 5};
    EXPECT_EQ(out, expected);
}

TEST(SamplerTest, RandomSamplerDeterministicPerEpochWithSeed)
{
    RandomSampler sampler_a(10, 123u);
    RandomSampler sampler_b(10, 123u);

    std::vector<std::size_t> a0(10);
    std::vector<std::size_t> b0(10);
    sampler_a.set_epoch(0);
    sampler_b.set_epoch(0);
    sampler_a.sample_into(a0);
    sampler_b.sample_into(b0);
    EXPECT_EQ(a0, b0);

    std::vector<std::size_t> a1(10);
    std::vector<std::size_t> b1(10);
    sampler_a.set_epoch(1);
    sampler_b.set_epoch(1);
    sampler_a.sample_into(a1);
    sampler_b.sample_into(b1);
    EXPECT_EQ(a1, b1);
    EXPECT_NE(a0, a1);
}

TEST(SamplerTest, WeightedSamplerProducesInRangeIndices)
{
    WeightedRandomSampler sampler({0.1, 0.1, 0.8}, 200, 42u);
    std::vector<std::size_t> out(sampler.index_count());
    sampler.set_epoch(0);
    sampler.sample_into(out);

    EXPECT_EQ(out.size(), 200U);
    for (const auto idx : out)
    {
        EXPECT_LT(idx, 3U);
    }

    const auto count_two = std::count(out.begin(), out.end(), static_cast<std::size_t>(2));
    EXPECT_GT(count_two, 80);
}

TEST(SamplerTest, DistributedSamplerPartitionsDataset)
{
    // dummy stub — real tests for distributed sampler are in SamplerThrowTest below
}

TEST(SamplerThrowTest, SequentialSamplerSpanMismatch)
{
    SequentialSampler sampler(5);
    std::vector<std::size_t> out(3); // wrong size
    EXPECT_THROW(sampler.sample_into(out), std::invalid_argument);
}

TEST(SamplerThrowTest, RandomSamplerSpanMismatch)
{
    RandomSampler sampler(5, 42u);
    std::vector<std::size_t> out(3); // wrong size
    EXPECT_THROW(sampler.sample_into(out), std::invalid_argument);
}

TEST(SamplerThrowTest, WeightedRandomSamplerEmptyWeights)
{
    EXPECT_THROW((WeightedRandomSampler({}, 5, 1u)), std::invalid_argument);
}

TEST(SamplerThrowTest, WeightedRandomSamplerZeroNumSamples)
{
    EXPECT_THROW((WeightedRandomSampler({0.5, 0.5}, 0, 1u)), std::invalid_argument);
}

TEST(SamplerThrowTest, WeightedRandomSamplerAllZeroWeights)
{
    EXPECT_THROW((WeightedRandomSampler({0.0, 0.0}, 5, 1u)), std::invalid_argument);
}

TEST(SamplerThrowTest, WeightedRandomSamplerSpanMismatch)
{
    WeightedRandomSampler sampler({0.3, 0.7}, 4, 1u);
    std::vector<std::size_t> out(3); // wrong size (expected 4)
    EXPECT_THROW(sampler.sample_into(out), std::invalid_argument);
}

TEST(SamplerThrowTest, DistributedSamplerZeroReplicas)
{
    EXPECT_THROW((DistributedSampler(10, 0, 0, false, false, 1u)), std::invalid_argument);
}

TEST(SamplerThrowTest, DistributedSamplerRankTooLarge)
{
    EXPECT_THROW((DistributedSampler(10, 2, 2, false, false, 1u)), std::invalid_argument);
}

TEST(SamplerTest, DistributedSamplerDropLast)
{
    // drop_last=true: num_samples = 10 / 3 = 3
    DistributedSampler rank0(10, 3, 0, false, true, 7u);
    EXPECT_EQ(rank0.index_count(), 3u);

    std::vector<std::size_t> out(rank0.index_count());
    rank0.sample_into(out);
    for (auto idx : out)
    {
        EXPECT_LT(idx, 10u);
    }
}

TEST(SamplerTest, DistributedSamplerSetEpochNoSeed)
{
    // No seed: set_epoch should be a no-op without crashing
    DistributedSampler sampler(6, 2, 0, true, false, std::nullopt);
    EXPECT_NO_THROW(sampler.set_epoch(5));
    std::vector<std::size_t> out(sampler.index_count());
    EXPECT_NO_THROW(sampler.sample_into(out));
}

TEST(SamplerTest, DistributedSamplerShuffleNoPadding)
{
    // drop_last=true with shuffle: all indices should be in range
    DistributedSampler rank0(9, 3, 0, true, true, 42u);
    rank0.set_epoch(0);
    std::vector<std::size_t> out(rank0.index_count());
    rank0.sample_into(out);
    for (auto idx : out)
    {
        EXPECT_LT(idx, 9u);
    }
}

TEST(SamplerThrowTest, DistributedSamplerSpanMismatch)
{
    DistributedSampler sampler(10, 2, 0, false, false, 1u);
    std::vector<std::size_t> out(3); // wrong size
    EXPECT_THROW(sampler.sample_into(out), std::invalid_argument);
}

TEST(SamplerTest, DistributedSamplerPartitionsDatasetFully)
{
    DistributedSampler rank0(10, 2, 0, false, false, 7u);
    DistributedSampler rank1(10, 2, 1, false, false, 7u);

    std::vector<std::size_t> out0(rank0.index_count());
    std::vector<std::size_t> out1(rank1.index_count());

    rank0.set_epoch(0);
    rank1.set_epoch(0);
    rank0.sample_into(out0);
    rank1.sample_into(out1);

    EXPECT_EQ(out0.size(), 5U);
    EXPECT_EQ(out1.size(), 5U);

    std::vector<std::size_t> merged;
    merged.reserve(out0.size() + out1.size());
    merged.insert(merged.end(), out0.begin(), out0.end());
    merged.insert(merged.end(), out1.begin(), out1.end());
    std::sort(merged.begin(), merged.end());

    std::vector<std::size_t> expected(10);
    std::iota(expected.begin(), expected.end(), static_cast<std::size_t>(0));
    EXPECT_EQ(merged, expected);
}

TEST(SamplerTest, DistributedSamplerEmptyDatasetFillsZeros)
{
    DistributedSampler sampler(0, 2, 0, false, false, 7u);
    std::vector<std::size_t> out(sampler.index_count());
    EXPECT_NO_THROW(sampler.sample_into(out));
    for (auto v : out)
    {
        EXPECT_EQ(v, 0U);
    }
}

TEST(SamplerTest, FoldSamplerTrainPartition)
{
    const statistics::FoldSplit split{
        .train_indices = {0, 2, 4, 6},
        .test_indices = {1, 3, 5},
    };

    FoldSampler sampler(split, FoldPartition::Train);
    std::vector<std::size_t> out(sampler.index_count());
    sampler.sample_into(out);

    const std::vector<std::size_t> expected = {0, 2, 4, 6};
    EXPECT_EQ(out, expected);
}

TEST(SamplerTest, FoldSamplerValidationPartition)
{
    const statistics::FoldSplit split{
        .train_indices = {0, 2, 4, 6},
        .test_indices = {1, 3, 5},
    };

    FoldSampler sampler(split, FoldPartition::Validation);
    std::vector<std::size_t> out(sampler.index_count());
    sampler.sample_into(out);

    const std::vector<std::size_t> expected = {1, 3, 5};
    EXPECT_EQ(out, expected);
}

TEST(SamplerTest, DistributedSamplerNonDropLastPadsByReplication)
{
    // dataset_size=5, replicas=2 => num_samples=3, total_size=6, one replicated element needed.
    DistributedSampler rank0(5, 2, 0, false, false, 11u);
    DistributedSampler rank1(5, 2, 1, false, false, 11u);

    std::vector<std::size_t> out0(rank0.index_count());
    std::vector<std::size_t> out1(rank1.index_count());
    rank0.sample_into(out0);
    rank1.sample_into(out1);

    ASSERT_EQ(out0.size(), 3U);
    ASSERT_EQ(out1.size(), 3U);

    std::vector<std::size_t> merged;
    merged.reserve(out0.size() + out1.size());
    merged.insert(merged.end(), out0.begin(), out0.end());
    merged.insert(merged.end(), out1.begin(), out1.end());

    std::sort(merged.begin(), merged.end());
    EXPECT_EQ(merged.size(), 6U);
    EXPECT_EQ(std::count(merged.begin(), merged.end(), 0U), 2);
}

// RandomSampler::index_count() — covers RandomSampler.cpp lines 22, 24
TEST(SamplerTest, RandomSamplerIndexCount)
{
    RandomSampler sampler(7, 42u);
    EXPECT_EQ(sampler.index_count(), 7U);
}

// SequentialSampler::set_epoch() — covers SequentialSampler.cpp lines 20, 23
TEST(SamplerTest, SequentialSamplerSetEpochIsNoOp)
{
    SequentialSampler sampler(4);
    sampler.set_epoch(3); // should not throw or change behavior
    std::vector<std::size_t> out(sampler.index_count());
    sampler.sample_into(out);
    EXPECT_EQ(out, (std::vector<std::size_t>{0, 1, 2, 3}));
}

// FoldSampler::set_epoch() — covers FoldSampler.cpp lines 28, 31
// FoldSampler::sample_into() throw — covers FoldSampler.cpp line 37
TEST(SamplerThrowTest, FoldSamplerSetEpochAndSizeMismatch)
{
    statistics::FoldSplit split;
    split.train_indices = {0, 2, 4};
    split.test_indices = {1, 3};
    FoldSampler sampler(split, FoldPartition::Train);

    // set_epoch is a no-op for FoldSampler
    EXPECT_NO_THROW(sampler.set_epoch(5));

    // sample_into with wrong size should throw
    std::vector<std::size_t> out(1); // wrong size (expected 3)
    EXPECT_THROW(sampler.sample_into(out), std::invalid_argument);
}
