#include "nn/dataLoaders/DataLoaderIterator.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "nn/dataLoaders/DataLoader.hpp"

using std::size_t;
using std::vector;

DataLoaderIterator::DataLoaderIterator(     //
    DataLoader& loader,                     //
    size_t current_batch,                   //
    std::shared_ptr<vector<size_t>> indices //
    )
    : loader_(loader),               //
      current_batch_(current_batch), //
      indices_(std::move(indices)    //
      )
{
}

auto DataLoaderIterator::operator*() const -> Batch
{
    size_t start_index = current_batch_ * loader_.batch_size_;
    const auto& indices = *indices_;
    size_t end_index = std::min(start_index + loader_.batch_size_, indices.size());

    vector<size_t> idxs;
    idxs.reserve(end_index - start_index);
    for (size_t i = start_index; i < end_index; ++i)
    {
        idxs.emplace_back(indices.at(i));
    }

    return loader_.dataset_->collate(idxs);
}

auto DataLoaderIterator::operator++() -> DataLoaderIterator&
{
    ++current_batch_;
    return *this;
}

auto DataLoaderIterator::operator!=(const DataLoaderIterator& other) const -> bool
{
    return current_batch_ != other.current_batch_;
}

auto DataLoaderIterator::operator==(const DataLoaderIterator& other) const -> bool
{
    return current_batch_ == other.current_batch_;
}
