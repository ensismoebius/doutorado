#ifndef NN_DATALOADERS_DATALOADERITERATOR_HPP
#define NN_DATALOADERS_DATALOADERITERATOR_HPP

#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

#include "nn/utility/batching.hpp"

class DataLoader;

class DataLoaderIterator
{
   public:
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = Batch;
    using pointer = Batch*;
    using reference = Batch&;

    DataLoaderIterator(DataLoader& loader,
        std::size_t current_batch,
        std::shared_ptr<std::vector<std::size_t>> indices);

    [[nodiscard]] auto operator*() const -> const Batch&;
    auto operator++() -> DataLoaderIterator&;
    auto operator!=(const DataLoaderIterator& other) const -> bool;
    auto operator==(const DataLoaderIterator& other) const -> bool;

   private:
    DataLoader& loader_;
    std::size_t current_batch_;
    std::shared_ptr<std::vector<std::size_t>> indices_;
    mutable Batch current_batch_data_;
    mutable bool batch_valid_{false};
    void fetch_batch() const;
};

#endif // NN_DATALOADERS_DATALOADERITERATOR_HPP
