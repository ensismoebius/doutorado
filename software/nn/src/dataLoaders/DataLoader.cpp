#include "DataLoader.h"

#include <random>
#include <utility>

DataLoader::DataLoader(  //
    Dataset dataset,     //
    int batch_size,      //
    bool shuffle         //
    )
    :                                // Initialize members
      dataset_(std::move(dataset)),  //
      batch_size_(batch_size),       //
      shuffle_(shuffle)              //
{
  num_batches_ =
      (dataset_.inputs.get_shape()[0] + batch_size_ - 1) / batch_size_;

  indices_.resize(dataset_.inputs.get_shape()[0]);

  std::iota(indices_.begin(), indices_.end(), 0);

  if (shuffle_) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices_.begin(), indices_.end(), g);
  }
}

auto DataLoader::begin() const -> DataLoader::Iterator { return {*this, 0}; }

auto DataLoader::end() const -> DataLoader::Iterator {
  return {*this, num_batches_};
}

DataLoader::Iterator::Iterator(const DataLoader& loader, int current_batch)
    : loader_(loader), current_batch_(current_batch) {}

auto DataLoader::Iterator::operator*() const -> Batch {
  int start_index = current_batch_ * loader_.batch_size_;
  int end_index =
      std::min(start_index + loader_.batch_size_, (int)loader_.indices_.size());

  std::vector<int> batch_indices(loader_.indices_.begin() + start_index,
                                 loader_.indices_.begin() + end_index);

  Tensor batch_inputs = loader_.dataset_.inputs.slice(batch_indices);
  Tensor batch_targets = loader_.dataset_.targets.slice(batch_indices);

  return {.inputs = batch_inputs, .targets = batch_targets};
}

auto DataLoader::Iterator::operator++() -> DataLoader::Iterator& {
  current_batch_++;
  return *this;
}

auto DataLoader::Iterator::operator!=(const Iterator& other) const -> bool {
  return current_batch_ != other.current_batch_;
}
