/**
 * @file GPUBufferPool.hpp
 * @brief GPU device memory pool for reusing allocations across tensor operations.
 *
 * Provides a thread-safe pool of pre-allocated GPU buffers to reduce allocation
 * overhead and enable persistent data staging on the device. Buffers are acquired
 * for operations and returned to the pool for reuse.
 */

#pragma once

#include <CL/cl.h>

#include <cstddef>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace nn::tensor
{

/**
 * GPUBuffer wraps a single OpenCL device buffer with size metadata.
 */
struct GPUBuffer
{
    cl_mem buffer = nullptr;
    size_t size_bytes = 0;

    GPUBuffer() = default;
    explicit GPUBuffer(cl_mem buf, size_t sz) : buffer(buf), size_bytes(sz) {}

    ~GPUBuffer()
    {
        if (buffer)
        {
            clReleaseMemObject(buffer);
        }
    }

    // Move-only semantics
    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;
    GPUBuffer(GPUBuffer&& other) noexcept : buffer(other.buffer), size_bytes(other.size_bytes)
    {
        other.buffer = nullptr;
        other.size_bytes = 0;
    }
    GPUBuffer& operator=(GPUBuffer&& other) noexcept
    {
        if (this != &other)
        {
            if (buffer) clReleaseMemObject(buffer);
            buffer = other.buffer;
            size_bytes = other.size_bytes;
            other.buffer = nullptr;
            other.size_bytes = 0;
        }
        return *this;
    }
};

/**
 * GPUBufferPoolHandle is an RAII handle that holds a buffer from the pool
 * and returns it on destruction.
 */
class GPUBufferPoolHandle
{
   public:
    GPUBufferPoolHandle() = default;
    explicit GPUBufferPoolHandle(class GPUBufferPool* pool, GPUBuffer buf)
        : pool_(pool), buffer_(std::move(buf))
    {
    }

    ~GPUBufferPoolHandle();

    GPUBuffer* operator->()
    {
        return &buffer_;
    }
    const GPUBuffer* operator->() const
    {
        return &buffer_;
    }
    GPUBuffer& operator*()
    {
        return buffer_;
    }
    const GPUBuffer& operator*() const
    {
        return buffer_;
    }

    explicit operator bool() const
    {
        return buffer_.buffer != nullptr;
    }

    // Move-only
    GPUBufferPoolHandle(const GPUBufferPoolHandle&) = delete;
    GPUBufferPoolHandle& operator=(const GPUBufferPoolHandle&) = delete;
    GPUBufferPoolHandle(GPUBufferPoolHandle&&) = default;
    GPUBufferPoolHandle& operator=(GPUBufferPoolHandle&&) = default;

   private:
    GPUBufferPool* pool_ = nullptr;
    GPUBuffer buffer_;
};

/**
 * GPUBufferPool manages a collection of pre-allocated GPU buffers organized
 * by size. Buffers are acquired for use and returned to the pool.
 *
 * Thread-safe via internal mutex.
 */
class GPUBufferPool
{
   public:
    // Cache ceiling: once cached (idle, available-for-reuse) buffers reach this many
    // bytes, further releases are dropped instead of pooled. Without this, a run that
    // touches many distinct tensor shapes accumulates buffers forever — each bucket is
    // capped at 20 buffers individually, but nothing capped the bucket count, so pinned
    // (real host RAM) memory could grow unbounded over a long training run.
    static constexpr size_t kDefaultMaxPoolBytes = 1ull << 30; // 1 GiB

    /**
     * Initialize the pool with a context and command queue.
     * @param context OpenCL context for allocation
     * @param queue OpenCL command queue (used for synchronization if needed)
     * @param use_pinned If true, use CL_MEM_ALLOC_HOST_PTR for pinned memory (faster transfers)
     * @param max_pool_bytes Ceiling on total bytes held in idle/cached buffers
     */
    explicit GPUBufferPool(cl_context context,
        cl_command_queue queue,
        bool use_pinned = true,
        size_t max_pool_bytes = kDefaultMaxPoolBytes);

    ~GPUBufferPool();

    /**
     * Acquire a buffer of at least `size_bytes` from the pool.
     * If no suitable buffer exists, allocates a new one.
     * @param size_bytes Requested buffer size
     * @return RAII handle that returns buffer to pool on destruction
     */
    auto acquire(size_t size_bytes) -> GPUBufferPoolHandle;

    /**
     * Return a buffer to the pool for reuse.
     * Called automatically by GPUBufferPoolHandle destructor.
     */
    void release(GPUBuffer buffer);

    /**
     * Clear all pooled buffers and release GPU memory.
     */
    void clear();

    /**
     * Get statistics about pool state.
     * @return {total_allocated_bytes, num_buffers, num_available}
     */
    auto get_stats() const -> std::tuple<size_t, size_t, size_t>;

   private:
    cl_context context_;
    cl_command_queue queue_;
    bool use_pinned_memory_;
    size_t max_pool_bytes_;

    mutable std::mutex mutex_;
    // Buffers organized by size ranges: 64B, 256B, 1KB, 4KB, etc.
    std::unordered_map<size_t, std::deque<GPUBuffer>> pools_;
    size_t cached_bytes_ = 0; // sum of pool_size * count over all buckets

    static size_t get_pool_size(size_t requested);
};

} // namespace nn::tensor
