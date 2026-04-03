/**
 * @file include/nn/tensor/CLBuffer.hpp
 * @brief GPU buffer management with explicit host/device synchronization.
 *
 * Manages OpenCL buffer allocation, device transfer, and memory tracking.
 * Supports explicit sync strategy for non-UMA devices (AMD Renoir APU).
 */

#ifndef CL_BUFFER_HPP
#define CL_BUFFER_HPP

#include <CL/cl.h>

#include <vector>

namespace nn::opencl
{

/**
 * @brief GPU-resident buffer with host mirror for explicit synchronization.
 *
 * Does NOT use mapping (clEnqueueMapBuffer). Instead maintains:
 * - `gpu_buffer`: OpenCL GPU-resident buffer
 * - `host_copy`: CPU-side shadow copy
 * - Dirty flags to track sync state
 *
 * Rationale: On AMD Renoir (non-UMA with coarse-grained SVM only),
 * mapping is inefficient. Explicit `WriteBuffer`/`ReadBuffer` is clearer.
 */
class CLBuffer
{
   public:
    /**
     * @brief Create an uninitialized buffer of the given size.
     *
     * @param size_elements Number of float elements
     */
    explicit CLBuffer(size_t size_elements);

    // Move-only (no copy to avoid accidental buffer duplication)
    CLBuffer(const CLBuffer&) = delete;
    CLBuffer& operator=(const CLBuffer&) = delete;

    CLBuffer(CLBuffer&& other) noexcept
        : m_gpu_buffer(other.m_gpu_buffer),
          m_host_copy(std::move(other.m_host_copy)),
          m_size_elements(other.m_size_elements),
          m_is_dirty_gpu(other.m_is_dirty_gpu),
          m_is_dirty_host(other.m_is_dirty_host)
    {
        other.m_gpu_buffer = nullptr;
        other.m_size_elements = 0;
    }

    CLBuffer& operator=(CLBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            m_gpu_buffer = other.m_gpu_buffer;
            m_host_copy = std::move(other.m_host_copy);
            m_size_elements = other.m_size_elements;
            m_is_dirty_gpu = other.m_is_dirty_gpu;
            m_is_dirty_host = other.m_is_dirty_host;

            other.m_gpu_buffer = nullptr;
            other.m_size_elements = 0;
        }
        return *this;
    }

    ~CLBuffer();

    /**
     * @brief Get the OpenCL buffer object.
     *
     * Use this when passing to kernel arguments.
     */
    auto get_gpu_buffer() const -> cl_mem
    {
        return m_gpu_buffer;
    }

    /**
     * @brief Get mutable pointer to CPU-side mirror.
     *
     * Modifying this data marks the buffer as dirty for GPU transfer.
     */
    auto get_host_ptr() -> float*
    {
        return m_host_copy.data();
    }

    /**
     * @brief Get const pointer to CPU-side mirror.
     */
    auto get_host_ptr() const -> const float*
    {
        return m_host_copy.data();
    }

    /**
     * @brief Get buffer size in float elements.
     */
    auto size_elements() const -> size_t
    {
        return m_size_elements;
    }

    /**
     * @brief Get buffer size in bytes.
     */
    auto size_bytes() const -> size_t
    {
        return m_size_elements * sizeof(float);
    }

    /**
     * @brief Write CPU data to GPU buffer.
     *
     * Async operation (CL_FALSE). Call flush() to ensure completion.
     *
     * @param queue OpenCL command queue
     */
    void write_to_device(cl_command_queue queue);

    /**
     * @brief Read GPU buffer back to CPU.
     *
     * **Expensive operation**—avoid unless necessary (e.g., validation, output).
     *
     * Blocking operation (CL_TRUE).
     *
     * @param queue OpenCL command queue
     */
    void read_from_device(cl_command_queue queue);

    /**
     * @brief Mark buffer as modified on GPU (internal use).
     *
     * Called after GPU kernels that modify this buffer.
     * Indicates next CPU access requires a read.
     */
    void mark_gpu_dirty()
    {
        m_is_dirty_host = true;
    }

    /**
     * @brief Mark buffer as modified on CPU (internal use).
     *
     * Called after CPU-side modifications.
     * Indicates next GPU kernel needs an updated copy.
     */
    void mark_host_dirty()
    {
        m_is_dirty_gpu = true;
    }

    /**
     * @brief Check if GPU buffer is out-of-sync with CPU.
     */
    auto is_gpu_dirty() const -> bool
    {
        return m_is_dirty_gpu;
    }

    /**
     * @brief Check if CPU buffer is out-of-sync with GPU.
     */
    auto is_host_dirty() const -> bool
    {
        return m_is_dirty_host;
    }

   private:
    void release();

    cl_mem m_gpu_buffer = nullptr;
    std::vector<float> m_host_copy;
    size_t m_size_elements = 0;

    // Sync flags
    bool m_is_dirty_gpu = true;   // GPU buffer needs update from host
    bool m_is_dirty_host = false; // Host buffer is stale (GPU modified)
};

} // namespace nn::opencl

#endif // CL_BUFFER_HPP
