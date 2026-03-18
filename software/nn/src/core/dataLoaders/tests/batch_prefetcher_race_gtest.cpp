/**
 * @file batch_prefetcher_race_gtest.cpp
 * @brief Stress-test to exercise potential hasNext()/next() races in the
 * BatchPrefetcher.
 */

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

TEST(BatchPrefetcherRaceTest, RepeatedPreservesOrderAgainstDataLoader)
{
    // Repeat the scenario multiple times to increase chance of catching races.
    for (int iter = 0; iter < 100; ++iter)
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
        BatchPrefetcher prefetcher(pref_loader, 100, 1);
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
}
