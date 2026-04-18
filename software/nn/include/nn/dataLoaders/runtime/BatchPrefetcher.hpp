/**
 * @file include/nn/dataLoaders/runtime/BatchPrefetcher.hpp
 * @brief Batchprefetcher.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_DATALOADERS_BATCHPREFETCHER_HPP
#define NN_DATALOADERS_BATCHPREFETCHER_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>

#include "nn/dataLoaders/interfaces/IBatchSource.hpp"
#include "nn/utility/BufferPool.hpp"
#include "nn/utility/HighPerfSpscQueue.hpp"

class BatchPrefetcher
{
   public:
    struct Diagnostics
    {
        std::size_t inflight_prefetch_bytes = 0;
        std::size_t ring_size = 0;
        std::size_t seen_batches = 0;
        bool producer_done = false;
        bool stop_requested = false;
    };

    BatchPrefetcher(std::unique_ptr<IBatchSource> source,
        std::size_t max_batches,
        std::size_t lookahead = 1,
        std::size_t max_prefetch_ram_bytes = 0);
    ~BatchPrefetcher();

    auto next() -> std::optional<Batch>;
    [[nodiscard]] auto hasNext() const -> bool;
    [[nodiscard]] auto seenBatches() const -> std::size_t;
    [[nodiscard]] auto diagnostics() const -> Diagnostics;

   private:
    struct PrefetchedBatch
    {
        Batch batch;
        std::size_t bytes = 0;
    };

    static auto estimate_batch_bytes(const Batch& batch) -> std::size_t;

    void producerLoop();

    std::unique_ptr<IBatchSource> source_;
    std::size_t max_batches_;
    std::atomic<std::size_t> seen_batches_;
    std::size_t lookahead_;
    std::size_t max_prefetch_ram_bytes_;
    std::atomic<std::size_t> inflight_prefetch_bytes_{0};

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::unique_ptr<HighPerfSpscQueue<PrefetchedBatch>> prefetched_ring_;
    std::unique_ptr<BufferPool<Batch>> prefetched_pool_;
    std::thread producer_thread_;
    bool producer_done_;
    std::atomic<bool> stop_requested_{false};
    std::exception_ptr producer_error_;
};

#endif // NN_DATALOADERS_BATCHPREFETCHER_HPP