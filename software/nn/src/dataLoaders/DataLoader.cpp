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
  // get_shape() returns vector<long>, so do arithmetic in long to avoid
  // implementation-defined narrowing when assigning to an int.
  long n_samples = dataset_.inputs.get_shape()[0];
  num_batches_ = (n_samples + static_cast<long>(batch_size_) - 1) /
                 static_cast<long>(batch_size_);

  indices_.resize(static_cast<size_t>(dataset_.inputs.get_shape()[0]));

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

DataLoader::Iterator::Iterator(const DataLoader& loader, long current_batch)
    : loader_(loader), current_batch_(current_batch) {}

auto DataLoader::Iterator::operator*() const -> Batch {
  long start_index = current_batch_ * static_cast<long>(loader_.batch_size_);
  long end_index =
      std::min(start_index + static_cast<long>(loader_.batch_size_),
               static_cast<long>(loader_.indices_.size()));

  // Tensor::slice expects std::vector<int>, but indices_ is std::vector<int>
  // and start/end are long. Copy the relevant range into a vector<int>.
  std::vector<int> batch_indices;
  batch_indices.reserve(static_cast<size_t>(end_index - start_index));
  for (long i = start_index; i < end_index; ++i) {
    batch_indices.push_back(loader_.indices_.at(static_cast<size_t>(i)));
  }

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
