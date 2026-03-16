#ifndef NN_DATALOADERS_BATCHPREFETCHER_HPP
#define NN_DATALOADERS_BATCHPREFETCHER_HPP

#include <cstddef>
#include <deque>
#include <future>
#include <optional>

#include "nn/dataLoaders/DataLoader.hpp"

class BatchPrefetcher
{
   public:
    BatchPrefetcher(DataLoader& loader, std::size_t max_batches, std::size_t lookahead = 1);

    auto next() -> std::optional<Batch>;
    [[nodiscard]] auto hasNext() const -> bool;
    [[nodiscard]] auto seenBatches() const -> std::size_t;

   private:
    DataLoader::Iterator it_;
    DataLoader::Iterator end_;
    std::size_t max_batches_;
    std::size_t seen_batches_;
    std::size_t lookahead_;
    DataLoader::Iterator schedule_cursor_;
    std::deque<std::future<Batch>> next_batch_futures_;
};

#endif // NN_DATALOADERS_BATCHPREFETCHER_HPP