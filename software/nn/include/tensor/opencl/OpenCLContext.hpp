/**
 * @file include/tensor/opencl/OpenCLContext.hpp
 * @brief OpenCL device and context management for the OpenCL backend.
 *
 * Manages device detection, context creation, and command queue handling.
 * Implements a singleton pattern to maintain a persistent OpenCL context
 * across training epochs, minimizing allocation overhead.
 *
 * **Hardware Notes:**
 * - Targets AMD Renoir APU (7 compute units, 64 KiB LDS, no UMA)
 * - Explicit buffer synchronization required (no fine-grained SVM)
 * - Kernel launch overhead is critical; batching recommended
 */

#ifndef OPENCL_CONTEXT_HPP
#define OPENCL_CONTEXT_HPP

#include <CL/cl.h>

#include <string>

namespace nn::opencl
{

/**
 * @brief Manages OpenCL device, context, and command queue.
 *
 * Singleton that persists for the application lifetime, avoiding repeated
 * device initialization and context destruction costs.
 */
class OpenCLContext
{
   public:
    /**
     * @brief RAII scope for batch mode.
     *
     * Starts batch mode on construction when OpenCL is available and
     * synchronizes exactly once on destruction.
     */
    struct BatchScope
    {
        bool active = false;

        BatchScope();
        ~BatchScope();

        BatchScope(const BatchScope&) = delete;
        BatchScope& operator=(const BatchScope&) = delete;
        BatchScope(BatchScope&& other) noexcept;
        BatchScope& operator=(BatchScope&& other) noexcept;
    };

    /**
     * @brief Get the singleton OpenCL context instance.
     *
     * First call initializes the context (device detection, context creation).
     * Subsequent calls return the existing instance.
     */
    static OpenCLContext& instance();

    // Deleted copy/move to enforce singleton semantics
    OpenCLContext(const OpenCLContext&) = delete;
    OpenCLContext& operator=(const OpenCLContext&) = delete;
    OpenCLContext(OpenCLContext&&) = delete;
    OpenCLContext& operator=(OpenCLContext&&) = delete;

    /**
     * @brief Check if OpenCL device is available and initialized.
     */
    auto is_available() const -> bool
    {
        return m_is_available;
    }

    /**
     * @brief Get the OpenCL device ID.
     *
     * @return cl_device_id, valid only if is_available() returns true
     */
    auto get_device() const -> cl_device_id
    {
        return m_device;
    }

    /**
     * @brief Get the OpenCL context.
     *
     * @return cl_context, valid only if is_available() returns true
     */
    auto get_context() const -> cl_context
    {
        return m_context;
    }

    /**
     * @brief Get the OpenCL command queue.
     *
     * Used for all kernel launches and buffer operations.
     *
     * @return cl_command_queue, valid only if is_available() returns true
     */
    auto get_queue() const -> cl_command_queue
    {
        return m_queue;
    }

    /**
     * @brief Get the device name as a human-readable string.
     *
     * @return Device name (e.g., "AMD Radeon Graphics (radeonsi, renoir, ACO)")
     */
    auto get_device_name() const -> const std::string&
    {
        return m_device_name;
    }

    /**
     * @brief Get platform name.
     *
     * @return Platform name (e.g., "rusticl", "NVIDIA CUDA", "Intel")
     */
    auto get_platform_name() const -> const std::string&
    {
        return m_platform_name;
    }

    /**
     * @brief Get the number of compute units on the device.
     *
     * @return Number of CUs (e.g., 7 for Renoir APU)
     */
    auto get_compute_units() const -> cl_uint
    {
        return m_compute_units;
    }

    /**
     * @brief Get local memory size in bytes.
     *
     * @return Local (shared/LDS) memory per workgroup in bytes (e.g., 65536 for Renoir)
     */
    auto get_local_memory_size() const -> cl_ulong
    {
        return m_local_memory_size;
    }

    /**
     * @brief Finish all pending operations in the command queue.
     *
     * Blocks until all enqueued commands complete. Use judiciously to avoid
     * serializing GPU/CPU work; prefer batching operations before a single flush.
     */
    void flush() const;

    /**
     * @brief Enable batch mode: defer queue synchronization after each kernel.
     *
     * When enabled, OpenCLTensorBackend operations skip per-kernel clFinish.
     * Use with end_batch() to amortize synchronization overhead across
     * multiple operations (e.g., one forward pass).
     *
     * Example:
     *   OpenCLContext::instance().begin_batch();
     *   output = layer1.forward(input);
     *   output = layer2.forward(output);
     *   // ... more layers
     *   OpenCLContext::instance().end_batch();  // single GPU sync
     */
    static void begin_batch();

    /**
     * @brief End batch mode and synchronize the GPU.
     *
     * If batching is active, calls queue.flush() to wait for all pending operations.
     * Safe to call even if batching wasn't active.
     */
    static void end_batch();

    /**
     * @brief Query current batch mode status.
     */
    static bool is_batching();

   private:
    OpenCLContext();
    ~OpenCLContext();

    void initialize_device();
    void query_device_info();

    bool m_is_available = false;

    cl_platform_id m_platform = nullptr;
    cl_device_id m_device = nullptr;
    cl_context m_context = nullptr;
    cl_command_queue m_queue = nullptr;

    std::string m_device_name;
    std::string m_platform_name;
    cl_uint m_compute_units = 0;
    cl_ulong m_local_memory_size = 0;

    // Reference-counted batch mode: depth > 0 → no per-kernel clFinish.
    // Nesting allows a full-network forward scope to absorb all per-layer scopes.
    static int s_batch_depth;
};

} // namespace nn::opencl

#endif // OPENCL_CONTEXT_HPP
