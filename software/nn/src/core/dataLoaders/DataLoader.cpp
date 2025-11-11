#include "DataLoader.h"

#include <random>
#include <utility>

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, std::size_t batch_size, bool shuffle,
                       std::optional<unsigned int> seed)
    : dataset_(std::move(dataset)), batch_size_(batch_size), shuffle_(shuffle), seed_(seed)
{
    std::size_t n_samples = dataset_->size();
    num_batches_ = (n_samples + batch_size_ - 1) / batch_size_;

    indices_.resize(n_samples);
    std::iota(indices_.begin(), indices_.end(), 0);
    if (shuffle_ && seed_)
    {
        std::mt19937 g(*seed_);
        std::shuffle(indices_.begin(), indices_.end(), g);
    }
    else if (shuffle_)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices_.begin(), indices_.end(), g);
    }
}

auto DataLoader::begin() -> DataLoader::Iterator
{
    // Re-shuffle at each epoch if requested
    if (shuffle_)
    {
        if (seed_)
        {
            // use epoch to vary shuffle when seed is provided
            std::mt19937 g(*seed_ + static_cast<unsigned int>(epoch_));
            std::shuffle(indices_.begin(), indices_.end(), g);
        }
        else
        {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }
    }
    ++epoch_;
    return {*this, 0};
}

auto DataLoader::end() -> DataLoader::Iterator
{
    return {*this, num_batches_};
}

DataLoader::Iterator::Iterator(DataLoader& loader, std::size_t current_batch)
    : loader_(loader), current_batch_(current_batch)
{
}

auto DataLoader::Iterator::operator*() const -> Batch
{
    std::size_t start_index = current_batch_ * loader_.batch_size_;
    std::size_t end_index = std::min(start_index + loader_.batch_size_, loader_.indices_.size());

    // build indices (size_t -> int) for Dataset::collate
    std::vector<std::size_t> idxs;
    idxs.reserve(end_index - start_index);
    for (std::size_t i = start_index; i < end_index; ++i)
    {
        idxs.push_back(loader_.indices_.at(i));
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
