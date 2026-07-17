/**
 * @file include/tensor/opencl/KernelManager.hpp
 * @brief OpenCL kernel compilation, caching, and execution.
 *
 * Manages kernel compilation with caching to avoid recompilation.
 * Handles kernel linking and program creation for efficiency.
 *
 * **Design:**
 * - Per-kernel lazy compilation (first use creates kernel)
 * - In-memory kernel source (embedded in binary)
 * - No file I/O overhead for kernel loading
 * - Thread-safe singleton for production use
 */

#ifndef KERNEL_MANAGER_HPP
#define KERNEL_MANAGER_HPP

#include <CL/cl.h>

#include <map>
#include <string>

namespace nn::opencl
{

/**
 * @brief Manages OpenCL kernel compilation and execution.
 *
 * Singleton that caches compiled kernels to avoid repeated compilation overhead.
 * Supports both single-kernel programs and multi-kernel programs for batch compilation.
 */
class KernelManager
{
   public:
    /**
     * @brief Get the singleton kernel manager instance.
     */
    static KernelManager& instance();

    KernelManager(const KernelManager&) = delete;
    KernelManager& operator=(const KernelManager&) = delete;
    KernelManager(KernelManager&&) = delete;
    KernelManager& operator=(KernelManager&&) = delete;

    /**
     * @brief Get or compile a kernel by name.
     *
     * First call compiles the kernel from source.
     * Subsequent calls return the cached kernel.
     *
     * @param kernel_name Name of the kernel (e.g., "matmul_kernel")
     * @return cl_kernel handle, valid until context cleanup
     * @throws std::runtime_error if compilation fails
     */
    cl_kernel get_kernel(const std::string& kernel_name);

    /**
     * @brief Get a compiled program by name (for inspection or multi-kernel batching).
     *
     * @param program_name Name of the program (e.g., "linear_algebra")
     * @return cl_program handle
     * @throws std::runtime_error if compilation fails
     */
    cl_program get_program(const std::string& program_name);

    /**
     * @brief Release all cached kernels and programs (cleanup).
     *
     * Called on shutdown or to free GPU memory.
     */
    void release_all();

    /**
     * @brief Query if a kernel is compiled and cached.
     *
     * @param kernel_name Name of the kernel
     * @return true if cached, false otherwise
     */
    bool has_kernel(const std::string& kernel_name) const;

   private:
    KernelManager();
    ~KernelManager();

    // Kernel source code for various operations
    // Embedded as strings to avoid file I/O
    static std::string get_kernel_source(const std::string& program_name);

    // Cache: program_name -> compiled cl_program
    std::map<std::string, cl_program> m_programs;

    // Cache: kernel_name -> compiled cl_kernel
    std::map<std::string, cl_kernel> m_kernels;
};

} // namespace nn::opencl

#endif // KERNEL_MANAGER_HPP
