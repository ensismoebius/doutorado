/**
 * @file src/core/dataLoaders/BatchPrefetcher.cpp
 * @brief Implementation for Batchprefetcher.
 *

 */

#include "dataLoaders/runtime/BatchPrefetcher.hpp"

#include <chrono>
#include <cstring>
#include <utility>

#include "logging/Logger.hpp"

using std::make_unique;

BatchPrefetcher::BatchPrefetcher(         //
    std::unique_ptr<IBatchSource> source, //
    std::size_t max_batches,              //
    std::size_t lookahead,                //
    std::size_t max_prefetch_ram_bytes)
    : source_(std::move(source)),

      max_batches_(max_batches),
      seen_batches_(0),
      lookahead_(lookahead == 0 ? 1 : lookahead),
      max_prefetch_ram_bytes_(max_prefetch_ram_bytes),
      producer_done_(false),
      stop_requested_(false),
      producer_error_(nullptr)
{
    // Construct the SPSC ring buffer with capacity=max_batches
    prefetched_ring_ =
        std::make_unique<HighPerfSpscQueue<PrefetchedBatch>>(max_batches_ > 0 ? max_batches_ : 1);
    // Construct a buffer pool prefilled with lookahead buffers to reduce allocations
    prefetched_pool_ = std::make_unique<BufferPool<Batch>>(lookahead_);

    producer_thread_ = std::thread([this]() { producerLoop(); });
}

auto BatchPrefetcher::estimate_batch_bytes(const Batch& batch) -> std::size_t
{
    const auto inputs_count = static_cast<std::size_t>(batch.inputs.size());
    const auto targets_count = static_cast<std::size_t>(batch.targets.size());
    return (inputs_count + targets_count) * sizeof(float);
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

    // Storage cleanup is handled by IBatchSource implementations if needed.
}

void BatchPrefetcher::producerLoop()
{
    // legacy shard flag removed; shard detection is automatic elsewhere

    try
    {
        while (true)
        {
            // Acquire a reusable buffer and ask iterator to fill it.
            Batch batch = prefetched_pool_->acquire();
            bool ok = source_->next(batch);
            if (!ok)
            {
                prefetched_pool_->release(std::move(batch));
                break;
            }
            const std::size_t batch_bytes = estimate_batch_bytes(batch);

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock,
                    [this, batch_bytes]()
                    {
                        if (stop_requested_)
                        {
                            return true;
                        }

                        if (seen_batches_.load(std::memory_order_relaxed) +
                                prefetched_ring_->size() >=
                            max_batches_)
                        {
                            return true;
                        }

                        if (prefetched_ring_->size() >= lookahead_)
                        {
                            return false;
                        }

                        if (max_prefetch_ram_bytes_ == 0)
                        {
                            return true;
                        }

                        const std::size_t inflight =
                            inflight_prefetch_bytes_.load(std::memory_order_relaxed);

                        if (inflight + batch_bytes <= max_prefetch_ram_bytes_)
                        {
                            return true;
                        }

                        return inflight == 0 && prefetched_ring_->empty();
                    });

                if (seen_batches_.load(std::memory_order_relaxed) + prefetched_ring_->size() >=
                    max_batches_)
                {
                    prefetched_pool_->release(std::move(batch));
                    break;
                }

                if (stop_requested_)
                {
                    prefetched_pool_->release(std::move(batch));
                    break;
                }
            }

            inflight_prefetch_bytes_.fetch_add(batch_bytes, std::memory_order_relaxed);

            // Storage persistence (e.g., SQLite) is handled by the source implementation.

            PrefetchedBatch prefetched{std::move(batch), batch_bytes};

            // Try to push into ring; if full, yield briefly and retry.
            bool pushed = false;
            while (!stop_requested_.load(std::memory_order_relaxed))
            {
                if (prefetched_ring_->try_push(std::move(prefetched)))
                {
                    pushed = true;
                    break;
                }
                std::this_thread::yield();
            }

            if (!pushed)
            {
                inflight_prefetch_bytes_.fetch_sub(batch_bytes, std::memory_order_relaxed);
                if (prefetched.batch.inputs.size() > 0 || prefetched.batch.targets.size() > 0)
                {
                    prefetched_pool_->release(std::move(prefetched.batch));
                }
            }

            cv_.notify_all();
        }
    }
    catch (...)
    {
        // Log the exception message (if available) to help debugging producer
        // thread failures (e.g., dataset or source errors). Store the exception
        // so main thread can rethrow it later.
        auto ep = std::current_exception();
        try
        {
            std::rethrow_exception(ep);
        }
        catch (const std::exception& ex)
        {
            NN_LOG_ERROR(std::string("Producer thread exception: ") + ex.what());
        }
        catch (...)
        {
            NN_LOG_ERROR("Producer thread exception: <non-std>");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        producer_error_ = ep;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        producer_done_ = true;
    }
    cv_.notify_all();
}

[[nodiscard]] auto BatchPrefetcher::hasNext() const -> bool
{
    std::unique_lock<std::mutex> lock(mutex_);

    // If the producer failed, let next() surface the exception deterministically.
    if (producer_error_)
    {
        return true;
    }

    if (!prefetched_ring_->empty())
    {
        return true;
    }

    // If the producer hasn't yet produced a batch, wait a short time to avoid
    // a race where `hasNext()` returns true but `next()` immediately finds
    // the queue empty because the producer finished between the calls.
    if (!producer_done_ &&
        (seen_batches_.load(std::memory_order_relaxed) + prefetched_ring_->size() < max_batches_))
    {
        cv_.wait_for(lock,
            std::chrono::milliseconds(10),
            [this]() { return !prefetched_ring_->empty() || producer_done_ || producer_error_; });

        if (!prefetched_ring_->empty())
        {
            return true;
        }
    }

    // Avoid a race where producer hasn't marked done yet, but the max-batch
    // budget is already exhausted and no more batches can ever be produced.
    if (seen_batches_.load(std::memory_order_relaxed) + prefetched_ring_->size() >= max_batches_)
    {
        return false;
    }

    return !producer_done_;
}

auto BatchPrefetcher::next() -> std::optional<Batch>
{
    // Fast path: try to pop without locking. This avoids mutex contention
    // when the ring already has data (common case).
    PrefetchedBatch prefetched;
    if (prefetched_ring_->try_pop(prefetched))
    {
        inflight_prefetch_bytes_.fetch_sub(prefetched.bytes, std::memory_order_relaxed);

        // Take ownership of the produced batch to return to the caller.
        // Return an empty buffer to the pool for reuse.
        Batch ret = std::move(prefetched.batch);
        prefetched_pool_->release(Batch{});
        seen_batches_.fetch_add(1, std::memory_order_relaxed);
        cv_.notify_all();
        return ret;
    }

    // Slow path: wait for producer to signal availability or completion.
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock,
        [this]()
        {
            return stop_requested_ || !prefetched_ring_->empty() || producer_done_ ||
                   static_cast<bool>(producer_error_);
        });

    // If a producer error occurred, rethrow it.
    if (producer_error_ != nullptr)
    {
        std::rethrow_exception(producer_error_);
    }

    if (prefetched_ring_->empty())
    {
        return std::nullopt;
    }

    bool ok = prefetched_ring_->try_pop(prefetched);
    if (!ok)
    {
        return std::nullopt;
    }

    (void) 0;
    inflight_prefetch_bytes_.fetch_sub(prefetched.bytes, std::memory_order_relaxed);

    // Move the stored batch out and return an empty buffer to the pool.
    Batch ret = std::move(prefetched.batch);
    prefetched_pool_->release(Batch{});
    seen_batches_.fetch_add(1, std::memory_order_relaxed);
    lock.unlock();
    cv_.notify_all();

    return ret;
}

[[nodiscard]] auto BatchPrefetcher::seenBatches() const -> std::size_t
{
    std::lock_guard<std::mutex> lock(mutex_);
    // Return total batches produced so far: consumed (seen_batches_) plus
    // those still present in the ring buffer. This matches callers that
    // expect a view of progress including buffered batches.
    return seen_batches_.load(std::memory_order_relaxed) + prefetched_ring_->size();
}

[[nodiscard]] auto BatchPrefetcher::diagnostics() const -> Diagnostics
{
    std::lock_guard<std::mutex> lock(mutex_);
    Diagnostics d{};
    d.inflight_prefetch_bytes = inflight_prefetch_bytes_.load(std::memory_order_relaxed);
    d.ring_size = prefetched_ring_->size();
    d.seen_batches = seen_batches_.load(std::memory_order_relaxed);
    d.producer_done = producer_done_;
    d.stop_requested = stop_requested_;
    return d;
}
