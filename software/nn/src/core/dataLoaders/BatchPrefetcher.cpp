#include "nn/dataLoaders/BatchPrefetcher.hpp"

#include <utility>

BatchPrefetcher::BatchPrefetcher( //
    DataLoader& loader,           //
    std::size_t max_batches,      //
    std::size_t lookahead         //
    )
    : it_(loader.begin()),
      end_(loader.end()),
      max_batches_(max_batches),
      seen_batches_(0),
      lookahead_(lookahead == 0 ? 1 : lookahead),
      producer_done_(false),
      stop_requested_(false),
      producer_error_(nullptr)
{
    producer_thread_ = std::thread([this]() { producerLoop(); });
}

BatchPrefetcher::~BatchPrefetcher()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_requested_ = true;
    }
    cv_.notify_all();

    if (producer_thread_.joinable())
    {
        producer_thread_.join();
    }
}

void BatchPrefetcher::producerLoop()
{
    try
    {
        while (it_ != end_)
        {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock,
                         [this]()
                         { return stop_requested_ || prefetched_batches_.size() < lookahead_; });

                if (stop_requested_)
                {
                    break;
                }

                if (seen_batches_ + prefetched_batches_.size() >= max_batches_)
                {
                    break;
                }
            }

            Batch batch = *it_;
            ++it_;

            {
                std::lock_guard<std::mutex> lock(mutex_);
                prefetched_batches_.push_back(std::move(batch));
            }
            cv_.notify_all();
        }
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        producer_error_ = std::current_exception();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        producer_done_ = true;
    }
    cv_.notify_all();
}

[[nodiscard]] auto BatchPrefetcher::hasNext() const -> bool
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Keep reporting data availability while there are buffered batches or the
    // producer can still produce more. If a producer-side exception happened,
    // return true so next() can surface it deterministically.
    return !prefetched_batches_.empty() || !producer_done_ || static_cast<bool>(producer_error_);
}

auto BatchPrefetcher::next() -> std::optional<Batch>
{
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock,
             [this]()
             {
                 return stop_requested_ || !prefetched_batches_.empty() || producer_done_ ||
                        static_cast<bool>(producer_error_);
             });

    if (producer_error_ != nullptr)
    {
        std::rethrow_exception(producer_error_);
    }

    if (prefetched_batches_.empty())
    {
        return std::nullopt;
    }

    Batch batch = std::move(prefetched_batches_.front());
    prefetched_batches_.pop_front();
    ++seen_batches_;
    lock.unlock();
    cv_.notify_all();

    return batch;
}

[[nodiscard]] auto BatchPrefetcher::seenBatches() const -> std::size_t
{
    std::lock_guard<std::mutex> lock(mutex_);
    return seen_batches_;
}
