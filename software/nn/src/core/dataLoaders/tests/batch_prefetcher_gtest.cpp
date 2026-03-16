/**
 * @file batch_prefetcher_gtest.cpp
 * @brief Unit tests for BatchPrefetcher lookahead/max-batch behavior.
 */

#include <vector>

#include "gtest/gtest.h"
#include "nn/dataLoaders/BatchPrefetcher.hpp"
#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/dataLoaders/TensorDataset.hpp"

static auto make_sequential_tensor(std::size_t rows, std::size_t cols) -> nn::Tensor
{
    nn::Tensor t(rows, cols);
    for (std::size_t r = 0; r < rows; ++r)
    {
        for (std::size_t c = 0; c < cols; ++c)
        {
            t.at(r, c) = static_cast<float>((r * cols) + c);
        }
    }
    return t;
}

TEST(BatchPrefetcherTest, RespectsMaxBatches)
{
    auto inputs = make_sequential_tensor(20, 2);
    auto targets = make_sequential_tensor(20, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader(dataset, 3, false);
    BatchPrefetcher prefetcher(loader, 4, 4);

    int batches = 0;
    int rows = 0;
    while (prefetcher.hasNext())
    {
        auto maybe = prefetcher.next();
        ASSERT_TRUE(maybe.has_value());
        ++batches;
        rows += maybe->inputs.rows();
    }

    EXPECT_EQ(batches, 4);
    EXPECT_EQ(rows, 12);
    EXPECT_EQ(prefetcher.seenBatches(), 4u);
}

TEST(BatchPrefetcherTest, PreservesOrderAgainstDataLoaderWhenNotShuffled)
{
    auto inputs = make_sequential_tensor(15, 2);
    auto targets = make_sequential_tensor(15, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader baseline_loader(dataset, 4, false);
    std::vector<int> baseline_ids;
    for (const auto& b : baseline_loader)
    {
        for (int r = 0; r < b.inputs.rows(); ++r)
        {
            baseline_ids.push_back(static_cast<int>(b.inputs.at(r, 0) / 2.0f));
        }
    }

    DataLoader pref_loader(dataset, 4, false);
    BatchPrefetcher prefetcher(pref_loader, 100, 3);
    std::vector<int> prefetched_ids;
    while (prefetcher.hasNext())
    {
        auto maybe = prefetcher.next();
        ASSERT_TRUE(maybe.has_value());
        for (int r = 0; r < maybe->inputs.rows(); ++r)
        {
            prefetched_ids.push_back(static_cast<int>(maybe->inputs.at(r, 0) / 2.0f));
        }
    }

    EXPECT_EQ(prefetched_ids, baseline_ids);
}

TEST(BatchPrefetcherTest, ZeroLookaheadFallsBackToOne)
{
    auto inputs = make_sequential_tensor(9, 2);
    auto targets = make_sequential_tensor(9, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    DataLoader loader(dataset, 2, false);
    BatchPrefetcher prefetcher(loader, 10, 0);

    int batches = 0;
    while (prefetcher.hasNext())
    {
        auto maybe = prefetcher.next();
        ASSERT_TRUE(maybe.has_value());
        ++batches;
    }

    EXPECT_EQ(batches, 5);
    EXPECT_EQ(prefetcher.seenBatches(), 5u);
}
