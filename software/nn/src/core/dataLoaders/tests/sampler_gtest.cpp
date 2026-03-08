#include <algorithm>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "nn/dataLoaders/samplers/DistributedSampler.hpp"
#include "nn/dataLoaders/samplers/RandomSampler.hpp"
#include "nn/dataLoaders/samplers/SequentialSampler.hpp"
#include "nn/dataLoaders/samplers/WeightedRandomSampler.hpp"

TEST(SamplerTest, SequentialSamplerProducesOrderedIndices)
{
    SequentialSampler sampler(6);
    std::vector<std::size_t> out(sampler.index_count());
    sampler.sample_into(out);

    std::vector<std::size_t> expected = {0, 1, 2, 3, 4, 5};
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
