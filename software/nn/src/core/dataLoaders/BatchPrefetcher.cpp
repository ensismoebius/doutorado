#include "nn/dataLoaders/BatchPrefetcher.hpp"

using std::async;

BatchPrefetcher::BatchPrefetcher( //
    DataLoader& loader,           //
    std::size_t max_batches       //
    )
    : it_(loader.begin()),       //
      end_(loader.end()),        //
      max_batches_(max_batches), //
      seen_batches_(0)           //
{
    if (it_ != end_)
    {
        // Capture a copy of the current iterator for the async task. If we
        // captured the member iterator (`it_`) by reference, the main thread
        // might advance it before the async task dereferences it, producing
        // the wrong batch or creating a data race. Making a local snapshot
        // and capturing it by value guarantees the background task sees the
        // exact iterator state intended at the time the prefetch was
        // scheduled.
        auto it_snapshot = it_;
        next_batch_future_ = async(     //
            std::launch::async,         //
            [it_snapshot]() mutable {   //
                return *it_snapshot;    //
            }                           //
        );
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
        // Take a snapshot of the iterator and capture it by value for
        // the async prefetch task so the produced Batch corresponds
        // to the intended position even if the main thread advances
        // `it_` concurrently.
        auto it_snapshot = it_;
        next_batch_future_ =
            async(                          //
                std::launch::async,         //
                [it_snapshot]() mutable {   //
                    return *it_snapshot;    //
                }                           //
            );
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