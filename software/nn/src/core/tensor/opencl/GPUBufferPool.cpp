/**
 * @file GPUBufferPool.cpp
 * @brief GPU device memory pool implementation.
 */

#include "tensor/opencl/GPUBufferPool.hpp"

#include <cassert>

namespace nn::tensor
{

GPUBufferPoolHandle::~GPUBufferPoolHandle()
{
    if (pool_ && buffer_.buffer != nullptr)
    {
        pool_->release(std::move(buffer_));
    }
}

GPUBufferPool::GPUBufferPool(
    cl_context context, cl_command_queue queue, bool use_pinned, size_t max_pool_bytes)
    : context_(context),
      queue_(queue),
      use_pinned_memory_(use_pinned),
      max_pool_bytes_(max_pool_bytes)
{
    clRetainContext(context);
    clRetainCommandQueue(queue);
}

GPUBufferPool::~GPUBufferPool()
{
    clear();
    clReleaseCommandQueue(queue_);
    clReleaseContext(context_);
}

size_t GPUBufferPool::get_pool_size(size_t requested)
{
    // Optimized bucket sizes for neural network workloads
    // NN typically uses: 1KB-256KB for activations, 256KB-16MB for weights
    if (requested <= 1024) return 1024;         // 1KB - small activations
    if (requested <= 4096) return 4096;         // 4KB
    if (requested <= 16384) return 16384;       // 16KB
    if (requested <= 65536) return 65536;       // 64KB - common layer output
    if (requested <= 262144) return 262144;     // 256KB - medium weights
    if (requested <= 1048576) return 1048576;   // 1MB - large activations
    if (requested <= 4194304) return 4194304;   // 4MB
    if (requested <= 16777216) return 16777216; // 16MB - big layers

    // For very large, round up to nearest 64MB
    const size_t bucket_size = 67108864;
    return ((requested + bucket_size - 1) / bucket_size) * bucket_size;
}

auto GPUBufferPool::acquire(size_t size_bytes) -> GPUBufferPoolHandle
{
    if (size_bytes == 0) return GPUBufferPoolHandle(nullptr, GPUBuffer());

    std::lock_guard<std::mutex> lock(mutex_);

    size_t pool_size = get_pool_size(size_bytes);

    auto& pool = pools_[pool_size];
    if (!pool.empty())
    {
        // Reuse existing buffer
        GPUBuffer buf = std::move(pool.front());
        pool.pop_front();
        cached_bytes_ -= pool_size;
        return GPUBufferPoolHandle(this, std::move(buf));
    }

    // Allocate new buffer
    cl_int err = CL_SUCCESS;
    cl_mem_flags flags = CL_MEM_READ_WRITE;
    if (use_pinned_memory_)
    {
        flags |= CL_MEM_ALLOC_HOST_PTR; // Pinned memory for faster transfers
    }
    cl_mem mem = clCreateBuffer(context_, flags, pool_size, nullptr, &err);

    if (err != CL_SUCCESS || !mem)
    {
        return GPUBufferPoolHandle(nullptr, GPUBuffer());
    }

    return GPUBufferPoolHandle(this, GPUBuffer(mem, pool_size));
}

void GPUBufferPool::release(GPUBuffer buffer)
{
    if (!buffer.buffer || buffer.size_bytes == 0) return;

    std::lock_guard<std::mutex> lock(mutex_);

    size_t pool_size = get_pool_size(buffer.size_bytes);
    auto& pool = pools_[pool_size];

    // Reuse if pool is not too large (max 20 buffers per size for NN workloads)
    // and the global cache ceiling has room. Otherwise buffer is dropped here
    // (destructor releases GPU memory) instead of being cached forever.
    if (pool.size() < 20 && cached_bytes_ + pool_size <= max_pool_bytes_)
    {
        cached_bytes_ += pool_size;
        pool.push_back(std::move(buffer));
    }
}

void GPUBufferPool::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    pools_.clear();
    cached_bytes_ = 0;
}

auto GPUBufferPool::get_stats() const -> std::tuple<size_t, size_t, size_t>
{
    std::lock_guard<std::mutex> lock(mutex_);

    size_t total_bytes = 0;
    size_t num_buffers = 0;
    size_t num_available = 0;

    for (const auto& [pool_size, deque] : pools_)
    {
        num_buffers += deque.size();
        num_available += deque.size();
        total_bytes += pool_size * deque.size();
    }

    return {total_bytes, num_buffers, num_available};
}

} // namespace nn::tensor
