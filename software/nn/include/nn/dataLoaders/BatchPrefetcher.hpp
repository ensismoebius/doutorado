#ifndef NN_DATALOADERS_BATCHPREFETCHER_HPP
#define NN_DATALOADERS_BATCHPREFETCHER_HPP

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <filesystem>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/dataLoaders/ShardReader.hpp"
#include "nn/utility/BufferPool.hpp"
#include "nn/utility/SpscRingBuffer.hpp"

class BatchPrefetcher
{
   public:
    BatchPrefetcher(DataLoader& loader,
                    std::size_t max_batches,
                    std::size_t lookahead = 1,
                    bool use_shards = false,
                    const std::string& dataset_root = "");
    ~BatchPrefetcher();

    auto next() -> std::optional<Batch>;
    [[nodiscard]] auto hasNext() const -> bool;
    [[nodiscard]] auto seenBatches() const -> std::size_t;

   private:
    void producerLoop();

    DataLoader::Iterator it_;
    DataLoader::Iterator end_;
    DataLoader* loader_ptr_ = nullptr;
    bool use_shards_ = false;
    std::string dataset_root_;
    std::vector<std::unique_ptr<nn::dataLoaders::ShardReader>> shard_readers_cache_;
    std::size_t max_batches_;
    std::atomic<std::size_t> seen_batches_;
    std::size_t lookahead_;

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::unique_ptr<SpscRingBuffer<Batch>> prefetched_ring_;
    std::unique_ptr<BufferPool<Batch>> prefetched_pool_;
    std::thread producer_thread_;
    bool producer_done_;
    bool stop_requested_;
    std::exception_ptr producer_error_;
};

#endif // NN_DATALOADERS_BATCHPREFETCHER_HPP