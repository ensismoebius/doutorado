#include "../include/BatchPrefetcher.hpp"

BatchPrefetcher::BatchPrefetcher(DataLoader& loader, std::size_t max_batches)
    : it_(loader.begin()), end_(loader.end()), max_batches_(max_batches), seen_batches_(0)
{
    if (it_ != end_)
    {
        auto it_snapshot = it_;
        next_batch_future_ =
            std::async(std::launch::async, [it_snapshot]() mutable { return *it_snapshot; });
    }
}

[[nodiscard]] auto BatchPrefetcher::hasNext() const -> bool
{
    return (it_ != end_) && (seen_batches_ < max_batches_) && next_batch_future_.has_value();
}

auto BatchPrefetcher::next() -> std::optional<Batch>
{
    if (!hasNext())
    {
        return std::nullopt;
    }

    Batch batch = next_batch_future_->get();

    ++it_;
    if (it_ != end_ && (seen_batches_ + 1) < max_batches_)
    {
        auto it_snapshot = it_;
        next_batch_future_ =
            std::async(std::launch::async, [it_snapshot]() mutable { return *it_snapshot; });
    }
    else
    {
        next_batch_future_.reset();
    }

    ++seen_batches_;
    return batch;
}

[[nodiscard]] auto BatchPrefetcher::seenBatches() const -> std::size_t
{
    return seen_batches_;
}
