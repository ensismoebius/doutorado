/**
 * @file include/utility/SpscRingBuffer.hpp
 * @brief Spscringbuffer.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

// Single-producer single-consumer ring buffer.
// Not thread-safe for multiple producers or multiple consumers.
template <typename T>
class SpscRingBuffer
{
   public:
    explicit SpscRingBuffer(std::size_t capacity)
    {
        // Round up to next power of two
        std::size_t cap = 1;
        while (cap < capacity) cap <<= 1;
        capacity_ = cap;
        mask_ = capacity_ - 1;
        buffer_.resize(capacity_);
        head_.store(0);
        tail_.store(0);
    }

    SpscRingBuffer(const SpscRingBuffer&) = delete;
    SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;

    // Try to push, return false if full.
    bool try_push(T&& item)
    {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        if ((head - tail) >= capacity_)
        {
            return false; // full
        }

        buffer_[head & mask_] = std::move(item);
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // Try to pop, return false if empty.
    bool try_pop(T& out)
    {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t head = head_.load(std::memory_order_acquire);
        if (head == tail)
        {
            return false; // empty
        }

        out = std::move(buffer_[tail & mask_]);
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool empty() const
    {
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        return head == tail;
    }

    [[nodiscard]] std::size_t size() const
    {
        const std::size_t head = head_.load(std::memory_order_acquire);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        return head - tail;
    }

    [[nodiscard]] std::size_t capacity() const
    {
        return capacity_;
    }

   private:
    std::size_t capacity_{0};
    std::size_t mask_{0};
    std::vector<T> buffer_;
    std::atomic<std::size_t> head_;
    std::atomic<std::size_t> tail_;
};
