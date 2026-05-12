/**
 * @file include/nn/utility/HighPerfSpscQueue.hpp
 * @brief Highperfspscqueue.
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
#include <utility>
#include <vector>

// High-performance single-producer single-consumer ring buffer.
// - Placement-new storage to avoid default-constructing T for every slot.
// - Power-of-two capacity with mask indexing.
// - Cache-line padded atomics to reduce false sharing.
// - Minimal memory ordering: acquire on loads that synchronize with release stores.

template <typename T>
class HighPerfSpscQueue
{
   public:
    explicit HighPerfSpscQueue(std::size_t capacity)
    {
        std::size_t cap = 1;
        while (cap < capacity) cap <<= 1;
        capacity_ = cap;
        mask_ = capacity_ - 1;

        storage_.resize(capacity_);

        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
    }

    ~HighPerfSpscQueue()
    {
        // Destroy any remaining constructed elements
        std::size_t tail = tail_.load(std::memory_order_relaxed);
        std::size_t head = head_.load(std::memory_order_relaxed);
        for (std::size_t i = tail; i != head; ++i)
        {
            auto ptr = reinterpret_cast<T*>(slot_ptr(i & mask_));
            ptr->~T();
        }
    }

    HighPerfSpscQueue(const HighPerfSpscQueue&) = delete;
    HighPerfSpscQueue& operator=(const HighPerfSpscQueue&) = delete;

    bool try_push(T&& item)
    {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        if ((head - tail) >= capacity_)
        {
            return false; // full
        }

        void* p = slot_ptr(head & mask_);
        ::new (p) T(std::move(item));
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    bool try_pop(T& out)
    {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t head = head_.load(std::memory_order_acquire);
        if (head == tail)
        {
            return false; // empty
        }

        T* p = reinterpret_cast<T*>(slot_ptr(tail & mask_));
        out = std::move(*p);
        p->~T();
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
    // Storage slot for placement-new alignment
    struct Storage
    {
        alignas(alignof(T)) unsigned char data[sizeof(T)];
    };

    void* slot_ptr(std::size_t idx)
    {
        return static_cast<void*>(storage_[idx].data);
    }

    std::size_t capacity_{0};
    std::size_t mask_{0};
    std::vector<Storage> storage_; // aligned slots for placement-new

    alignas(64) std::atomic<std::size_t> head_;
    alignas(64) std::atomic<std::size_t> tail_;
};
