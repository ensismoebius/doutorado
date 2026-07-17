/**
 * @file include/utility/BufferPool.hpp
 * @brief Bufferpool.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#pragma once

#include <mutex>
#include <utility>
#include <vector>

// Simple thread-safe pool for reusable objects. Not lock-free but lightweight.
template <typename T>
class BufferPool
{
   public:
    explicit BufferPool(std::size_t initial = 0)
    {
        pool_.reserve(initial);
        for (std::size_t i = 0; i < initial; ++i)
        {
            pool_.emplace_back();
        }
    }

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;

    // Acquire an object (by move) from the pool, or default-construct one.
    T acquire()
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!pool_.empty())
        {
            T t = std::move(pool_.back());
            pool_.pop_back();
            return t;
        }
        return T{};
    }

    // Return an object to the pool (by move).
    void release(T&& t)
    {
        std::lock_guard<std::mutex> lk(mutex_);
        pool_.emplace_back(std::move(t));
    }

    [[nodiscard]] std::size_t size() const
    {
        std::lock_guard<std::mutex> lk(mutex_);
        return pool_.size();
    }

   private:
    mutable std::mutex mutex_;
    std::vector<T> pool_;
};
