/**
 * @file include/nn/tensor/DeviceMemory.hpp
 * @brief GPU device memory management and host<->device transfers.
 *
 * Provides RAII-based GPU buffer management with automatic synchronization.
 *
 * **Design Principles:**
 * - No implicit data transfers (explicit copy_to_device/from_device)
 * - RAII pattern: buffer allocated on construction, freed on destruction
 * - Lazy host-side cache: device is source of truth, host copy on-demand
 * - Thread-safe for single command queue (AMD Renoir single-queue model)
 *
 * **Memory Layout:**
 * - Row-major dense storage (compatibility with Eigen, BLAS)
 * - No padding or alignment overhead for AMD Renoir
 */

#ifndef DEVICE_MEMORY_HPP
#define DEVICE_MEMORY_HPP

#include <CL/cl.h>

#include <cstring>

namespace nn::opencl
{

using Index = std::size_t;

/**
 * @brief Managed GPU buffer with optional host-side shadow copy.
 *
 * Owns a cl_mem buffer on the device and optionally caches the host copy.
 * Lifecycle is bound to this object: allocation on construction, deallocation on destruction.
 */
class DeviceMemory
{
   public:
    /**
     * @brief Construct an unallocated buffer (default state).
     *
     * Call allocate() or move-assign from another DeviceMemory to initialize.
     */
    DeviceMemory() noexcept;

    /**
     * @brief Allocate GPU buffer of the given size (bytes).
     *
     * @param num_bytes Size in bytes
     * @throws std::runtime_error if allocation fails (device memory exhausted)
     */
    explicit DeviceMemory(Index num_bytes);

    /**
     * @brief Move constructor: transfer GPU buffer ownership.
     */
    DeviceMemory(DeviceMemory&& other) noexcept;

    /**
     * @brief Move assignment: release old buffer, take ownership of other's buffer.
     */
    DeviceMemory& operator=(DeviceMemory&& other) noexcept;

    // Deleted: no shallow copies to avoid use-after-free
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    /**
     * @brief Destructor: release GPU buffer (async-safe via OpenCL ref counting).
     */
    ~DeviceMemory() noexcept;

    /**
     * @brief Check if buffer is allocated.
     */
    bool is_allocated() const noexcept
    {
        return m_device_buffer != nullptr;
    }

    /**
     * @brief Get allocated size in bytes.
     */
    Index size() const noexcept
    {
        return m_size_bytes;
    }

    /**
     * @brief Get underlying cl_mem handle (device buffer).
     *
     * Valid only if is_allocated() returns true.
     */
    cl_mem get_device_buffer() const noexcept
    {
        return m_device_buffer;
    }

    /**
     * @brief Copy host data to device (write device).
     *
     * Asynchronously enqueues a host->device copy. Use sync() to ensure completion.
     *
     * @param host_data Pointer to host memory, must be at least size() bytes
     * @throws std::runtime_error if enqueue fails
     */
    void copy_to_device(const void* host_data);

    /**
     * @brief Copy device data to host (read device).
     *
     * Synchronously reads device->host (blocks until copy completes).
     *
     * @param out_host_data Pointer to output host memory, must be at least size() bytes
     * @throws std::runtime_error if read fails
     */
    void copy_from_device(void* out_host_data) const;

    /**
     * @brief Synchronize: wait for all pending operations on this buffer to complete.
     *
     * Ensures copy_to_device and kernels using this buffer have completed.
     */
    void sync() const;

    /**
     * @brief Re-allocate to a new size, discarding old contents.
     *
     * Releases existing buffer and allocates a new one.
     *
     * @param new_size_bytes New size in bytes
     * @throws std::runtime_error if allocation fails
     */
    void reallocate(Index new_size_bytes);

   private:
    cl_mem m_device_buffer{nullptr};
    Index m_size_bytes{0};
};

/**
 * @brief RAII host-side staging buffer for temporary data transfers.
 *
 * Allocates page-locked memory suitable for DMA transfers.
 * Useful for non-blocking I/O patterns.
 */
class StagingBuffer
{
   public:
    /**
     * @brief Allocate pinned host memory for transfers.
     *
     * @param num_bytes Size in bytes
     * @throws std::runtime_error if allocation fails
     */
    explicit StagingBuffer(Index num_bytes);

    StagingBuffer(StagingBuffer&&) noexcept;
    StagingBuffer& operator=(StagingBuffer&&) noexcept;

    StagingBuffer(const StagingBuffer&) = delete;
    StagingBuffer& operator=(const StagingBuffer&) = delete;

    /**
     * @brief Free pinned memory.
     */
    ~StagingBuffer() noexcept;

    /**
     * @brief Get pointer to pinned host memory.
     */
    void* data() noexcept
    {
        return m_host_data;
    }
    const void* data() const noexcept
    {
        return m_host_data;
    }

    /**
     * @brief Size of allocated memory in bytes.
     */
    Index size() const noexcept
    {
        return m_size_bytes;
    }

   private:
    void* m_host_data{nullptr};
    Index m_size_bytes{0};
};

} // namespace nn::opencl

#endif // DEVICE_MEMORY_HPP
