/**
 * @file DataLoader.cpp
 * @brief Implementation of the `DataLoader` batching/iteration utilities.
 */

#include "nn/dataLoaders/DataLoader.hpp"

#include <algorithm>
#include <random>
#include <utility>

// Implementation notes:
// - The `DataLoader` owns a base `indices_` vector and produces per-iterator
//   snapshots to avoid iterator interference.
// - When `seed_` is present, `begin()` adds `epoch_` to the seed to produce a
//   deterministic-but-different shuffle each epoch.
// - `Iterator::operator*()` delegates batching to `Dataset::collate()`, which
//   enables datasets to implement fast slicing/gather.

DataLoader::DataLoader(               //
    std::shared_ptr<Dataset> dataset, //
    std::size_t batch_size,           //
    bool do_shuffle,                  //
    std::optional<unsigned int> seed  //
    )
    : dataset_(std::move(dataset)), batch_size_(batch_size), shuffle_(do_shuffle), seed_(seed)
{
    if (!dataset_)
    {
        throw std::invalid_argument("DataLoader: dataset cannot be null.");
    }
    if (batch_size == 0)
    {
        throw std::invalid_argument("DataLoader: batch size cannot be zero.");
    }
    // Check for implicitly converted negative values (wrapped to very large size_t)
    // When -1 is passed to size_t, it becomes SIZE_MAX (typically 2^64-1 or 2^32-1)
    // Any batch_size > 1 billion is suspicious and likely a wrapped negative
    constexpr std::size_t MAX_REASONABLE_BATCH_SIZE = 1'000'000'000;
    if (batch_size > MAX_REASONABLE_BATCH_SIZE)
    {
        throw std::invalid_argument(
            "DataLoader: batch size is unreasonably large (possible negative value).");
    }

    // TODO - Stopped here - Im checking _indices values and batches calculation. I want to make
    // sure the logic is correct before proceeding to iterators.

    // Precompute number of batches and initialize indices
    const std::size_t n_samples = dataset_->size();
    num_batches_ = (n_samples + batch_size_ - 1) / batch_size_;
    indices_.resize(n_samples);

    std::iota(indices_.begin(), indices_.end(), 0);
    if (shuffle_ && seed_)
    {
        std::mt19937 g(*seed_);
        std::shuffle(indices_.begin(), indices_.end(), g);
        return;
    }

    if (shuffle_)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices_.begin(), indices_.end(), g);
    }
}

auto DataLoader::begin() -> DataLoader::Iterator
{
    // Create a snapshot of indices for this iterator
    std::vector<std::size_t> snapshot = indices_;

    // Shuffle the snapshot if requested
    if (shuffle_)
    {
        if (seed_)
        {
            // use epoch to vary shuffle when seed is provided
            std::mt19937 g(*seed_ + static_cast<unsigned int>(epoch_));
            std::shuffle(snapshot.begin(), snapshot.end(), g);
        }
        else
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(snapshot.begin(), snapshot.end(), g);
        }
    }
    ++epoch_;
    return {*this, 0, std::move(snapshot)};
}

auto DataLoader::end() -> DataLoader::Iterator
{
    // End iterator doesn't need valid indices, just position
    return {*this, num_batches_, {}};
}

DataLoader::Iterator::Iterator(      //
    DataLoader& loader,              //
    std::size_t current_batch,       //
    std::vector<std::size_t> indices //
    )
    : loader_(loader),               //
      current_batch_(current_batch), //
      indices_(std::move(indices)    //
      )
{
}

auto DataLoader::Iterator::operator*() const -> Batch
{
    std::size_t start_index = current_batch_ * loader_.batch_size_;
    std::size_t end_index = std::min(start_index + loader_.batch_size_, indices_.size());

    // build indices (size_t -> int) for Dataset::collate
    std::vector<std::size_t> idxs;
    idxs.reserve(end_index - start_index);
    for (std::size_t i = start_index; i < end_index; ++i)
    {
        idxs.push_back(indices_.at(i));
    }

    // Delegate collation to dataset (allows custom collate behavior)
    return loader_.dataset_->collate(idxs);
}

auto DataLoader::Iterator::operator++() -> DataLoader::Iterator&
{
    ++current_batch_;
    return *this;
}

auto DataLoader::Iterator::operator!=(const Iterator& other) const -> bool
{
    return current_batch_ != other.current_batch_;
}

auto DataLoader::Iterator::operator==(const Iterator& other) const -> bool
{
    return current_batch_ == other.current_batch_;
}
