/**
 * @file src/core/tensor/opencl/OpenCLProfiling.cpp
 * @brief OpenCL profiling helper implementations.
 */

#include "nn/tensor/opencl/OpenCLProfiling.hpp"

#include <CL/cl.h>

#include <atomic>
#include <cstdint>

#include "nn/logging/Logger.hpp"
#include "nn/tensor/opencl/OpenCLContext.hpp"

namespace nn::opencl::profiling
{
static std::atomic<bool> g_enabled{false};

void set_enabled(bool enabled)
{
    g_enabled.store(enabled, std::memory_order_relaxed);
}

bool is_enabled()
{
    return g_enabled.load(std::memory_order_relaxed);
}

uint64_t profile_kernel_and_get_ns(cl_command_queue queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_size,
    const size_t* local_work_size)
{
    if (!queue || !kernel) return 0;
    if (!is_enabled())
    {
        // Enqueue kernel WITHOUT synchronization when profiling is disabled
        // This allows kernels to pipeline and reduces per-kernel overhead
        cl_int err = clEnqueueNDRangeKernel(queue,
            kernel,
            work_dim,
            nullptr,
            global_work_size,
            local_work_size,
            0,
            nullptr,
            nullptr);
        if (err != CL_SUCCESS) return 0;
        // NO clFinish here - let copy_from_device handle implicit synchronization
        return 0;
    }

    cl_event evt = nullptr;
    cl_int err = clEnqueueNDRangeKernel(
        queue, kernel, work_dim, nullptr, global_work_size, local_work_size, 0, nullptr, &evt);
    if (err != CL_SUCCESS)
    {
        return 0;
    }
    err = clFinish(queue);
    if (err != CL_SUCCESS)
    {
        if (evt) clReleaseEvent(evt);
        return 0;
    }

    const uint64_t ns = event_duration_ns(evt);
    if (evt) clReleaseEvent(evt);
    return ns;
}
uint64_t event_duration_ns(cl_event evt)
{
    if (!evt) return 0;
    cl_int err = CL_SUCCESS;
    cl_ulong start = 0;
    cl_ulong end = 0;
    err = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
    if (err != CL_SUCCESS)
    {
        NN_LOG_DEBUG(
            "OpenCL profiling: clGetEventProfilingInfo start failed: " + std::to_string(err));
        return 0;
    }
    err = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
    if (err != CL_SUCCESS)
    {
        NN_LOG_DEBUG(
            "OpenCL profiling: clGetEventProfilingInfo end failed: " + std::to_string(err));
        return 0;
    }
    if (end < start) return 0;
    return static_cast<uint64_t>(end - start);
}

void enqueue_and_profile(cl_command_queue queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_size,
    const size_t* local_work_size,
    const char* event_name)
{
    if (!queue || !kernel)
    {
        NN_LOG_DEBUG("OpenCL profiling: invalid queue or kernel");
        return;
    }

    const uint64_t ns =
        profile_kernel_and_get_ns(queue, kernel, work_dim, global_work_size, local_work_size);
    if (ns > 0)
    {
        NN_LOG_DEBUG(std::string("OpenCL kernel profile: ") + event_name +
                     " duration_ns=" + std::to_string(ns));
    }
}

void enqueue_and_profile_batched(cl_command_queue queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_size,
    const size_t* local_work_size,
    const char* event_name)
{
    if (!queue || !kernel)
    {
        NN_LOG_DEBUG("OpenCL profiling: invalid queue or kernel");
        return;
    }

    // If batch mode is active, defer synchronization to allow kernel pipelining
    if (opencl::OpenCLContext::is_batching())
    {
        // Enqueue without synchronization
        cl_event evt =
            enqueue_no_finish(queue, kernel, work_dim, global_work_size, local_work_size);
        if (evt)
        {
            clReleaseEvent(evt);
        }
    }
    else
    {
        // Normal blocking enqueue with profiling
        enqueue_and_profile(queue, kernel, work_dim, global_work_size, local_work_size, event_name);
    }
}

cl_event enqueue_no_finish(cl_command_queue queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t* global_work_size,
    const size_t* local_work_size)
{
    if (!queue || !kernel)
    {
        NN_LOG_DEBUG("OpenCL: invalid queue or kernel in enqueue_no_finish");
        return nullptr;
    }

    cl_event evt = nullptr;
    cl_int err = clEnqueueNDRangeKernel(
        queue, kernel, work_dim, nullptr, global_work_size, local_work_size, 0, nullptr, &evt);
    if (err != CL_SUCCESS)
    {
        NN_LOG_DEBUG(std::string("OpenCL enqueue_no_finish failed: ") + std::to_string(err));
        return nullptr;
    }

    return evt;
}

void finish_queue(cl_command_queue queue)
{
    if (!queue)
    {
        return;
    }
    cl_int err = clFinish(queue);
    if (err != CL_SUCCESS)
    {
        NN_LOG_DEBUG(std::string("OpenCL finish_queue failed: ") + std::to_string(err));
    }
}

} // namespace nn::opencl::profiling
