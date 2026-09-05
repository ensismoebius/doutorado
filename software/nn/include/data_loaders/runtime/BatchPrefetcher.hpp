/**
 * @file include/data_loaders/runtime/BatchPrefetcher.hpp
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

#include "data_loaders/interfaces/IBatchSource.hpp"
#include "utility/BufferPool.hpp"
#include "utility/HighPerfSpscQueue.hpp"

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

    // Wait predicate for the producer's backpressure wait in producerLoop(). Must be called
    // with `mutex_` held (as cv_.wait's predicate does). Returns true when the producer may
    // push `batch_bytes` worth of data (or must wake up to re-check stop/limit conditions),
    // false to keep waiting.
    [[nodiscard]] auto shouldProceedProducing(std::size_t batch_bytes) const -> bool;

    // Pushes `prefetched` into prefetched_ring_, retrying (with a yield) until it succeeds or
    // stop is requested; on failure to push, returns the batch to the pool. Extracted from
    // producerLoop() with no behavior change.
    void pushPrefetchedOrRelease(PrefetchedBatch prefetched, std::size_t batch_bytes);

    // Logs and stashes an in-flight producer-thread exception into producer_error_ so the
    // consumer thread can rethrow it deterministically from next().
    void handleProducerException();

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