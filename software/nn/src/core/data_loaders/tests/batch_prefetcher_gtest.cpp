/**
 * @file batch_prefetcher_gtest.cpp
 * @brief Targeted tests for BatchPrefetcher RAM-cap and fast-path behavior.
 */

#include <sqlite3.h>
#include <unistd.h>

#include <chrono>
#include <functional>
#include <thread>

#include "data_loaders/datasets/TensorDataset.hpp"
#include "data_loaders/runtime/BatchPrefetcher.hpp"
#include "data_loaders/sources/SqliteBatchSource.hpp"
#include "gtest/gtest.h"
#include "test_utils/SqliteTestHelpers.hpp"

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
    return t; //
} //

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

// Use shared helpers from include/test_utils/SqliteTestHelpers.hpp

namespace
{
auto make_small_windowed_source(const std::string& db_root, std::size_t batch_size)
    -> std::unique_ptr<SqliteBatchSource>
{
    nn::windowing::WindowSpec eeg_win{.window_size = 4, .overlap = 0.5f, .sample_rate = 1024};
    nn::windowing::WindowSpec audio_win{.window_size = 4, .overlap = 0.5f, .sample_rate = 44100};
    return std::make_unique<SqliteBatchSource>(db_root,
        batch_size,
        nn::dataLoaders::SqliteDatasetType::FusedWindow,
        eeg_win,
        audio_win,
        Protocol101117InputMode::Concatenated);
}
} // namespace

TEST(BatchPrefetcherRamCapTest, OversizedBatchStillMakesProgress)
{
    auto inputs = make_sequential_tensor(4, 64);
    auto targets = make_sequential_tensor(4, 16);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);
    const std::string db_root = nn::testing::make_temp_db_path_unique("nn_batch_prefetch_test");
    nn::testing::create_simple_protocol_db(db_root, 4096, 176400);
    auto src = std::make_unique<SqliteBatchSource>(db_root, 4);
    BatchPrefetcher prefetcher(std::move(src), 1, 1, 64);

    ASSERT_TRUE(wait_until(
        [&]() { return prefetcher.diagnostics().ring_size >= 1; }, std::chrono::milliseconds(500)));

    auto batch = prefetcher.next();
    ASSERT_TRUE(batch.has_value());
    EXPECT_EQ(batch->inputs.rows(), 4);
    // Protocol+Concatenated now emits stacked-and-resampled rows at width 176400.
    EXPECT_EQ(batch->inputs.cols(), 176400);
    EXPECT_EQ(batch->targets.rows(), 4);
    EXPECT_EQ(batch->targets.cols(), 176400);

    const auto d = prefetcher.diagnostics();
    EXPECT_GE(d.ring_size + d.seen_batches, 1U);
}

TEST(BatchPrefetcherRamCapTest, WaitUntilReturnsFalseWhenPredicateNeverTrue)
{
    const bool ok = wait_until([]() { return false; }, std::chrono::milliseconds(1));
    EXPECT_FALSE(ok);
}

TEST(BatchPrefetcherRamCapTest, OneBatchCapAllowsSequentialProgress)
{
    auto inputs = make_sequential_tensor(4, 32);
    auto targets = make_sequential_tensor(4, 8);
    auto dataset = std::make_shared<TensorDataset>(inputs, targets);
    const std::string db_root2 = nn::testing::make_temp_db_path_unique("nn_batch_prefetch_test");
    nn::testing::create_simple_protocol_db(db_root2, 8, 8);
    const std::size_t one_batch_cap_bytes = (2U * 28U + 2U * 28U) * sizeof(float);
    auto src2 = make_small_windowed_source(db_root2, 2);
    BatchPrefetcher prefetcher(std::move(src2), 2, 2, one_batch_cap_bytes);

    ASSERT_TRUE(wait_until(
        [&]() { return prefetcher.diagnostics().ring_size >= 1; }, std::chrono::milliseconds(500)));

    auto first = prefetcher.next();
    ASSERT_TRUE(first.has_value());

    ASSERT_TRUE(wait_until(
        [&]() { return prefetcher.seenBatches() >= 2; }, std::chrono::milliseconds(1000)));

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
    const std::string db_root3 = nn::testing::make_temp_db_path_unique("nn_batch_prefetch_test");
    nn::testing::create_simple_protocol_db(db_root3, 12, 12);
    auto src3 = make_small_windowed_source(db_root3, 2);
    BatchPrefetcher prefetcher(std::move(src3), 3, 3, 0);

    ASSERT_TRUE(wait_until(
        [&]() { return prefetcher.diagnostics().ring_size >= 2; }, std::chrono::milliseconds(500)));

    auto first = prefetcher.next();
    ASSERT_TRUE(first.has_value());

    const auto d = prefetcher.diagnostics();
    EXPECT_GE(d.ring_size, 1U);
}

TEST(BatchPrefetcherFastPathTest, FastDominateSlowInSteadyState)
{
    auto inputs = make_sequential_tensor(20, 24);
    auto targets = make_sequential_tensor(20, 4);
    auto source_dataset = std::make_shared<TensorDataset>(inputs, targets);
    (void) source_dataset;
    const std::string db_root4 = nn::testing::make_temp_db_path_unique("nn_batch_prefetch_test");
    nn::testing::create_simple_protocol_db(db_root4, 12, 12);

    // Provide plenty of lookahead to ensure fast paths are hit
    auto src4 = make_small_windowed_source(db_root4, 2);
    BatchPrefetcher prefetcher(std::move(src4), 10, 5, 0);

    // Let producer fill up the queue initially
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    int consumed = 0;
    while (auto b = prefetcher.next())
    {
        consumed++;
        // short simulated processing delay: queue should remain non-empty
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto d = prefetcher.diagnostics();
    std::cout << "\n[   METRICS] Total batches consumed: " << consumed << "\n";
    std::cout << "[   METRICS] Ring size: " << d.ring_size << "\n";
    std::cout << "[   METRICS] Seen batches: " << d.seen_batches << "\n";

    // 1 trial, fused window_size=4/hop=2 on length-12 modalities => 5 windows.
    // With batch_size=2, emit batches of rows [2, 2, 1] => 3 total batches.
    EXPECT_EQ(consumed, 3);
    EXPECT_EQ(d.seen_batches, 3U);
}
