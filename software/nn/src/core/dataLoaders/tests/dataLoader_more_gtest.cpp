/**
 * @file dataLoader_more_gtest.cpp
 * @brief Additional tests for dataset collation and edge cases.
 */

#include <numeric>

#include "gtest/gtest.h"
#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/dataLoaders/TensorDataset.hpp"

// Reuse helper from existing tests: build a Tensor with sequential rows (N x D)
static auto make_sequential_tensor(std::size_t N, std::size_t D) -> nn::Tensor
{
    nn::Tensor t(static_cast<size_t>(N), static_cast<size_t>(D));
    for (std::size_t i = 0; i < N; ++i)
    {
        for (std::size_t j = 0; j < D; ++j)
        {
            t.at(static_cast<size_t>(i), static_cast<size_t>(j)) = static_cast<float>((i * D) + j);
        }
    }
    return t;
}

TEST(DataLoaderMoreTest, CollateProducesCorrectShapesAndValues)
{
    auto inputs = make_sequential_tensor(6, 3);
    auto targets = make_sequential_tensor(6, 2);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    // Ask collate for indices {1,4}
    std::vector<std::size_t> indices = {1, 4};
    Batch b = dataset->collate(indices);

    EXPECT_EQ(b.inputs.rows(), 2);
    EXPECT_EQ(b.inputs.cols(), 3);
    EXPECT_EQ(b.targets.rows(), 2);
    EXPECT_EQ(b.targets.cols(), 2);

    // Check values: row 0 should equal original row 1
    for (int c = 0; c < b.inputs.cols(); ++c)
    {
        EXPECT_FLOAT_EQ(b.inputs.at(0, c), inputs.at(1, c));
    }
    // row 1 equals original row 4
    for (int c = 0; c < b.inputs.cols(); ++c)
    {
        EXPECT_FLOAT_EQ(b.inputs.at(1, c), inputs.at(4, c));
    }
}

TEST(DataLoaderMoreTest, MismatchedInputTargetColumnsDetected)
{
    // Create inputs and targets with different column counts and ensure
    // collate still produces outputs but with the expected column dims.
    auto inputs = make_sequential_tensor(4, 5);
    auto targets = make_sequential_tensor(4, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    std::vector<std::size_t> indices = {0, 2};
    Batch b = dataset->collate(indices);
    EXPECT_EQ(b.inputs.cols(), 5);
    EXPECT_EQ(b.targets.cols(), 1);
    EXPECT_EQ(b.inputs.rows(), 2);
    EXPECT_EQ(b.targets.rows(), 2);
}

TEST(DataLoaderMoreTest, DifferentSeedsChangeOrder)
{
    auto inputs = make_sequential_tensor(12, 2);
    auto targets = make_sequential_tensor(12, 1);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    // Same seed should be deterministic: two loaders with same seed -> same order
    DataLoader dl_seedA_1(dataset, 3, true, 123u);
    DataLoader dl_seedA_2(dataset, 3, true, 123u);
    std::vector<int> orderA1, orderA2;
    for (const auto& b : dl_seedA_1)
    {
        for (int r = 0; r < b.inputs.rows(); ++r)
        {
            orderA1.push_back(static_cast<int>(b.inputs.at(r, 0) / 2));
        }
    }
    for (const auto& b : dl_seedA_2)
    {
        for (int r = 0; r < b.inputs.rows(); ++r)
        {
            orderA2.push_back(static_cast<int>(b.inputs.at(r, 0) / 2));
        }
    }
    EXPECT_EQ(orderA1, orderA2);

    // Different seed should produce a different ordering (very likely). Compare
    // the ordering from seed A with ordering from seed B and require they differ.
    DataLoader dl_seedB(dataset, 3, true, 456u);
    std::vector<int> orderB;
    for (const auto& b : dl_seedB)
    {
        for (int r = 0; r < b.inputs.rows(); ++r)
        {
            orderB.push_back(static_cast<int>(b.inputs.at(r, 0) / 2));
        }
    }
    // Expect different ordering for different seeds. If orders are identical
    // that's unexpected and the test will fail (extremely unlikely).
    EXPECT_NE(orderA1, orderB);
}

// Basic concurrency test: spawn multiple threads that iterate over separate
// DataLoader instances (same dataset) to ensure no shared-state races in the
// loader's per-instance state. This is intentionally light-weight and only
// serves as a smoke test (not a full thread-safety guarantee).
#include <thread>

TEST(DataLoaderMoreTest, ConcurrencySmokeTest)
{
    auto inputs = make_sequential_tensor(100, 3);
    auto targets = make_sequential_tensor(100, 3);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);

    const int num_threads = 4;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    std::atomic<int> total_batches{0};

    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(
            [&, t]()
            {
                DataLoader loader(dataset, 7, true, static_cast<unsigned int>(100 + t));
                int local = std::accumulate(loader.begin(),
                                            loader.end(),
                                            0,
                                            [](int sum, const auto& b)
                                            { return sum + static_cast<int>(b.inputs.rows()); });
                total_batches.fetch_add(local, std::memory_order_relaxed);
            });
    }

    for (auto& th : threads) th.join();

    // Each sample should be visited exactly once per loader run; there are 100
    // rows per loader, and num_threads loaders ran, so total should be
    // 100*num_threads
    EXPECT_EQ(total_batches.load(std::memory_order_relaxed), 100 * num_threads);
}
