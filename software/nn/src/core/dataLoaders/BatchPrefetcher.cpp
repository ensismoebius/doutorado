#include "nn/dataLoaders/BatchPrefetcher.hpp"

using std::async;

using std::move;

static void schedule_prefetch_if_needed(DataLoader::Iterator& schedule_cursor,
                                        DataLoader::Iterator const& end, std::size_t lookahead,
                                        std::size_t max_batches, std::size_t seen_batches,
                                        std::deque<std::future<Batch>>& queue)
{
    while (queue.size() < static_cast<size_t>(lookahead) && schedule_cursor != end &&
           (seen_batches + queue.size()) < max_batches)
    {
        auto snap = schedule_cursor;
        queue.push_back(async(std::launch::async, [snap]() mutable { return *snap; }));
        ++schedule_cursor;
    }
}

BatchPrefetcher::BatchPrefetcher( //
    DataLoader& loader,           //
    std::size_t max_batches,      //
    std::size_t lookahead         //
    )
    : it_(loader.begin()),       //
      end_(loader.end()),        //
      max_batches_(max_batches), //
      seen_batches_(0),          //
      lookahead_(lookahead),     //
      schedule_cursor_(it_)
{
    // Prime the prefetch queue up to `lookahead_` or until we hit EOF /
    // max_batches_. We schedule using a separate cursor so `it_` remains the
    // canonical iterator for the next-to-consume batch.
    schedule_prefetch_if_needed(
        schedule_cursor_, end_, lookahead_, max_batches_, seen_batches_, next_batch_futures_);
}

[[nodiscard]] auto BatchPrefetcher::hasNext() const -> bool
{
    return (!next_batch_futures_.empty()) && (seen_batches_ < max_batches_);
}

auto BatchPrefetcher::next() -> std::optional<Batch>
{
    if (!hasNext())
    {
        return std::nullopt;
    }

    // Pop the oldest prefetched batch and obtain it.
    auto fut = std::move(next_batch_futures_.front());
    next_batch_futures_.pop_front();
    Batch batch = fut.get();

    // Advance the canonical iterator to reflect consumption.
    ++it_;
    ++seen_batches_;

    // Refill the prefetch queue to maintain the lookahead.
    schedule_prefetch_if_needed(
        schedule_cursor_, end_, lookahead_, max_batches_, seen_batches_, next_batch_futures_);

    return batch;
}

[[nodiscard]] auto BatchPrefetcher::seenBatches() const -> std::size_t
{
    return seen_batches_;
}