/**
 * @file batch_prefetcher_gtest.cpp
 * @brief Targeted tests for BatchPrefetcher RAM-cap and fast-path behavior.
 */

#include <chrono>
#include <functional>
#include <thread>

#include "gtest/gtest.h"
#include "nn/dataLoaders/BatchPrefetcher.hpp"
#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/dataLoaders/TensorDataset.hpp"

static auto make_sequential_tensor(std::size_t rows, std::size_t cols) -> nn::Tensor
{
    nn::Tensor t(static_cast<size_t>(rows), static_cast<size_t>(cols));
    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            t.at(static_cast<size_t>(i), static_cast<size_t>(j)) =
                static_cast<float>((i * cols) + j);
        }
    }
    return t;
}

static auto wait_until(const std::function<bool()>& predicate, std::chrono::milliseconds timeout)
    -> bool
{
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout)
    {
        if (predicate())
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return predicate();
}

TEST(BatchPrefetcherRamCapTest, OversizedBatchStillMakesProgress)
{
    auto inputs = make_sequential_tensor(4, 64);
    auto targets = make_sequential_tensor(4, 16);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);
    DataLoader loader(dataset, 4, false);

    // Much smaller than one full batch to exercise the oversized fallback path.
    BatchPrefetcher prefetcher(loader, 1, 1, false, "", 64);

    ASSERT_TRUE(wait_until([&]() { return prefetcher.diagnostics().push_successes >= 1; },
        std::chrono::milliseconds(500)));

    auto batch = prefetcher.next();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->inputs.rows(), 4);
    EXPECT_EQ(batch->inputs.cols(), 64);
    EXPECT_EQ(batch->targets.rows(), 4);
    EXPECT_EQ(batch->targets.cols(), 16);

    const auto d = prefetcher.diagnostics();
    EXPECT_GE(d.fast_path_hits + d.slow_path_hits, 1U);
}

TEST(BatchPrefetcherRamCapTest, OneBatchCapAllowsSequentialProgress)
{
    auto inputs = make_sequential_tensor(4, 32);
    auto targets = make_sequential_tensor(4, 8);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);
    DataLoader loader(dataset, 2, false);

    // Exactly one 2-sample batch in bytes: (2*32 + 2*8) floats * 4 bytes.
    const std::size_t one_batch_cap_bytes = (2U * 32U + 2U * 8U) * sizeof(float);
    BatchPrefetcher prefetcher(loader, 2, 2, false, "", one_batch_cap_bytes);

    ASSERT_TRUE(wait_until([&]() { return prefetcher.diagnostics().push_successes >= 1; },
        std::chrono::milliseconds(500)));

    auto first = prefetcher.next();
    ASSERT_TRUE(first.has_value());

    ASSERT_TRUE(wait_until([&]() { return prefetcher.diagnostics().push_successes >= 2; },
        std::chrono::milliseconds(1000)));

    ASSERT_TRUE(wait_until(
        [&]() { return prefetcher.diagnostics().ring_size > 0; }, std::chrono::milliseconds(500)));

    auto second = prefetcher.next();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(prefetcher.seenBatches(), 2U);
}

TEST(BatchPrefetcherFastPathTest, BufferedBatchesHitFastPath)
{
    auto inputs = make_sequential_tensor(6, 24);
    auto targets = make_sequential_tensor(6, 4);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);
    DataLoader loader(dataset, 2, false);

    BatchPrefetcher prefetcher(loader, 3, 3, false, "", 0);

    ASSERT_TRUE(wait_until(
        [&]() { return prefetcher.diagnostics().ring_size >= 2; }, std::chrono::milliseconds(500)));

    auto first = prefetcher.next();
    ASSERT_TRUE(first.has_value());

    const auto d = prefetcher.diagnostics();
    EXPECT_GE(d.fast_path_hits, 1U);
}

TEST(BatchPrefetcherFastPathTest, FastDominateSlowInSteadyState)
{
    auto inputs = make_sequential_tensor(20, 24);
    auto targets = make_sequential_tensor(20, 4);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);
    DataLoader loader(dataset, 2, false);

    // Provide plenty of lookahead to ensure fast paths are hit
    BatchPrefetcher prefetcher(loader, 10, 5, false, "", 0);

    // Let producer fill up the queue initially
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    int consumed = 0;
    while(auto b = prefetcher.next()) {
        consumed++;
        // short simulated processing delay: queue should remain non-empty
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto d = prefetcher.diagnostics();
    std::cout << "\n[   METRICS] Total batches consumed: " << consumed << "\n";
    std::cout << "[   METRICS] Fast path hits: " << d.fast_path_hits << "\n";
    std::cout << "[   METRICS] Slow path hits: " << d.slow_path_hits << "\n";
    
    EXPECT_GT(d.fast_path_hits, d.slow_path_hits) << "Fast path should dominate in steady state!";
}
