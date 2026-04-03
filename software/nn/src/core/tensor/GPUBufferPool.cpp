/**
 * @file GPUBufferPool.cpp
 * @brief GPU device memory pool implementation.
 */

#include "nn/tensor/GPUBufferPool.hpp"

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

GPUBufferPool::GPUBufferPool(cl_context context, cl_command_queue queue)
    : context_(context), queue_(queue)
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

size_t GPUBufferPool::get_pool_size(size_t requested) const
{
    // Round up to nearest power of 2, min 64 bytes
    if (requested <= 64) return 64;
    if (requested <= 256) return 256;
    if (requested <= 1024) return 1024;
    if (requested <= 4096) return 4096;

    // For larger sizes, round up to nearest 64KB
    const size_t bucket_size = 65536;
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
        return GPUBufferPoolHandle(this, std::move(buf));
    }

    // Allocate new buffer
    cl_int err = CL_SUCCESS;
    cl_mem mem = clCreateBuffer(context_, CL_MEM_READ_WRITE, pool_size, nullptr, &err);

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

    // Reuse if pool is not too large (max 10 buffers per size)
    if (pool.size() < 10)
    {
        pool.push_back(std::move(buffer));
    }
    // Otherwise buffer is dropped (destructor releases GPU memory)
}

void GPUBufferPool::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);
    pools_.clear();
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
