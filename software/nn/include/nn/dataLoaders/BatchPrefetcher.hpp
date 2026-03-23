#ifndef NN_DATALOADERS_BATCHPREFETCHER_HPP
#define NN_DATALOADERS_BATCHPREFETCHER_HPP

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/utility/BufferPool.hpp"
#include "nn/utility/HighPerfSpscQueue.hpp"

class BatchPrefetcher
{
   public:
    struct Diagnostics
    {
        std::size_t fast_path_hits = 0;
        std::size_t slow_path_hits = 0;
        std::size_t push_successes = 0;
        std::size_t push_retries = 0;
        std::size_t inflight_prefetch_bytes = 0;
        std::size_t ring_size = 0;
        std::size_t seen_batches = 0;
        bool producer_done = false;
        bool stop_requested = false;
    };

    BatchPrefetcher(DataLoader& loader,
        std::size_t max_batches,
        std::size_t lookahead = 1,
        bool use_shards = false,
        const std::string& dataset_root = "",
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

    DataLoader::Iterator it_;
    DataLoader::Iterator end_;
    bool use_shards_ = false;
    std::string dataset_root_;
    std::size_t max_batches_;
    std::atomic<std::size_t> seen_batches_;
    std::size_t lookahead_;
    std::size_t max_prefetch_ram_bytes_;
    std::atomic<std::size_t> inflight_prefetch_bytes_{0};

    // Lightweight diagnostics
    std::atomic<std::size_t> fast_path_hits_{0};
    std::atomic<std::size_t> slow_path_hits_{0};
    std::atomic<std::size_t> push_successes_{0};
    std::atomic<std::size_t> push_retries_{0};

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::unique_ptr<HighPerfSpscQueue<PrefetchedBatch>> prefetched_ring_;
    std::unique_ptr<BufferPool<Batch>> prefetched_pool_;
    std::thread producer_thread_;
    bool producer_done_;
    bool stop_requested_;
    std::exception_ptr producer_error_;
};

#endif // NN_DATALOADERS_BATCHPREFETCHER_HPP