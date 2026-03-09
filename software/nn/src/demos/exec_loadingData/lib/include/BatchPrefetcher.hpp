#ifndef EXEC_LOADINGDATA_BATCHPREFETCHER_HPP
#define EXEC_LOADINGDATA_BATCHPREFETCHER_HPP

#include <cstddef>
#include <future>
#include <optional>

#include "nn/dataLoaders/DataLoader.hpp"

class BatchPrefetcher
{
   public:
    BatchPrefetcher(DataLoader& loader, std::size_t max_batches);

    [[nodiscard]] auto hasNext() const -> bool;
    auto next() -> std::optional<Batch>;
    [[nodiscard]] auto seenBatches() const -> std::size_t;

   private:
    DataLoader::Iterator it_;
    DataLoader::Iterator end_;
    std::size_t max_batches_;
    std::size_t seen_batches_;
    std::optional<std::future<Batch>> next_batch_future_;
};

#endif // EXEC_LOADINGDATA_BATCHPREFETCHER_HPP
