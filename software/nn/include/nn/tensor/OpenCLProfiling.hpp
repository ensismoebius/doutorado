/**
 * @file include/nn/tensor/OpenCLProfiling.hpp
 * @brief Simple OpenCL event profiling helpers.
 */

#ifndef OPENCL_PROFILING_HPP
#define OPENCL_PROFILING_HPP

#include <CL/cl.h>

#include <cstddef>
#include <cstdint>

namespace nn::opencl::profiling
{
/**
 * Enqueue a kernel and collect profiling information. This blocks until
 * completion (calls clFinish) and logs timing via the project's logger.
 *
 * Parameters mirror clEnqueueNDRangeKernel except `event_name` is used for
 * logging.
 */
void enqueue_and_profile(cl_command_queue queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_size,
    const size_t* local_work_size,
    const char* event_name);

/**
 * Enqueue a kernel and collect profiling information. This respects batch mode:
 * if batching is active, defers queue synchronization; otherwise blocks.
 *
 * Parameters mirror clEnqueueNDRangeKernel except `event_name` is used for
 * logging.
 */
void enqueue_and_profile_batched(cl_command_queue queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_size,
    const size_t* local_work_size,
    const char* event_name);

/**
 * Return the kernel execution duration in nanoseconds for a completed event.
 * The event must be a profiling-enabled event and be in a complete state.
 */
uint64_t event_duration_ns(cl_event evt);

/**
 * Enable or disable profiling at runtime. When disabled, `enqueue_and_profile`
 * performs a plain enqueue+finish without collecting profiling events.
 */
void set_enabled(bool enabled);

/**
 * Query whether profiling is currently enabled.
 */
bool is_enabled();

/**
 * Enqueue a kernel and return the measured duration in nanoseconds.
 * Returns 0 on error or if profiling is disabled.
 */
uint64_t profile_kernel_and_get_ns(cl_command_queue queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_size,
    const size_t* local_work_size);

/**
 * Enqueue a kernel WITHOUT blocking (deferred synchronization).
 * For batching multiple kernels before synchronizing.
 * Returns the event, or nullptr on error. Caller must release event.
 */
cl_event enqueue_no_finish(cl_command_queue queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_size,
    const size_t* local_work_size);

/**
 * Synchronize the command queue (blocks until all enqueued commands complete).
 */
void finish_queue(cl_command_queue queue);

} // namespace nn::opencl::profiling

#endif // OPENCL_PROFILING_HPP
