#ifndef NN_DATALOADERS_BATCHPREFETCHER_HPP
#define NN_DATALOADERS_BATCHPREFETCHER_HPP

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>

#include "nn/dataLoaders/DataLoader.hpp"

class BatchPrefetcher
{
   public:
    BatchPrefetcher(DataLoader& loader, std::size_t max_batches, std::size_t lookahead = 1);
    ~BatchPrefetcher();

    auto next() -> std::optional<Batch>;
    [[nodiscard]] auto hasNext() const -> bool;
    [[nodiscard]] auto seenBatches() const -> std::size_t;

   private:
    void producerLoop();

    DataLoader::Iterator it_;
    DataLoader::Iterator end_;
    std::size_t max_batches_;
    std::size_t seen_batches_;
    std::size_t lookahead_;

    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    std::deque<Batch> prefetched_batches_;
    std::thread producer_thread_;
    bool producer_done_;
    bool stop_requested_;
    std::exception_ptr producer_error_;
};

#endif // NN_DATALOADERS_BATCHPREFETCHER_HPP