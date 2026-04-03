/**
 * @file src/core/tensor/DeviceMemory.cpp
 * @brief GPU device memory management implementation.
 */

#include "nn/tensor/DeviceMemory.hpp"

#include <cassert>
#include <stdexcept>

#include "nn/logging/Logger.hpp"
#include "nn/tensor/OpenCLContext.hpp"

namespace nn::opencl
{

// Helper: throw on OpenCL error
static void check_cl_error(cl_int err, const std::string& context)
{
    if (err != CL_SUCCESS)
    {
        std::string msg = "OpenCL error [" + context + "]: " + std::to_string(err);
        throw std::runtime_error(msg);
    }
}

// === DeviceMemory ===

DeviceMemory::DeviceMemory() noexcept : m_device_buffer(nullptr), m_size_bytes(0) {}

DeviceMemory::DeviceMemory(Index num_bytes) : m_size_bytes(num_bytes)
{
    if (num_bytes == 0)
    {
        throw std::runtime_error("DeviceMemory: cannot allocate 0 bytes");
    }

    const auto& ctx = OpenCLContext::instance();
    if (!ctx.is_available())
    {
        throw std::runtime_error("DeviceMemory: OpenCL context not available");
    }

    cl_int err = CL_SUCCESS;
    m_device_buffer =
        clCreateBuffer(ctx.get_context(), CL_MEM_READ_WRITE, num_bytes, nullptr, &err);

    if (err != CL_SUCCESS)
    {
        NN_LOG_ERROR("DeviceMemory allocation failed for " + std::to_string(num_bytes) + " bytes");
        check_cl_error(err, "clCreateBuffer");
    }

    NN_LOG_DEBUG("DeviceMemory allocated: " + std::to_string(num_bytes) + " bytes");
}

DeviceMemory::DeviceMemory(DeviceMemory&& other) noexcept
    : m_device_buffer(other.m_device_buffer), m_size_bytes(other.m_size_bytes)
{
    other.m_device_buffer = nullptr;
    other.m_size_bytes = 0;
}

DeviceMemory& DeviceMemory::operator=(DeviceMemory&& other) noexcept
{
    if (this != &other)
    {
        // Release existing buffer
        if (m_device_buffer != nullptr)
        {
            clReleaseMemObject(m_device_buffer);
        }

        // Take ownership
        m_device_buffer = other.m_device_buffer;
        m_size_bytes = other.m_size_bytes;

        other.m_device_buffer = nullptr;
        other.m_size_bytes = 0;
    }
    return *this;
}

DeviceMemory::~DeviceMemory() noexcept
{
    if (m_device_buffer != nullptr)
    {
        clReleaseMemObject(m_device_buffer);
        m_device_buffer = nullptr;
    }
}

void DeviceMemory::copy_to_device(const void* host_data)
{
    if (!is_allocated())
    {
        throw std::runtime_error("DeviceMemory: cannot copy to unallocated buffer");
    }

    const auto& ctx = OpenCLContext::instance();
    if (!ctx.is_available())
    {
        throw std::runtime_error("DeviceMemory: OpenCL context not available");
    }

    cl_int err = clEnqueueWriteBuffer(ctx.get_queue(),
        m_device_buffer,
        CL_FALSE, // non-blocking
        0,
        m_size_bytes,
        host_data,
        0,
        nullptr,
        nullptr);

    check_cl_error(err, "clEnqueueWriteBuffer");
}

void DeviceMemory::copy_from_device(void* out_host_data) const
{
    if (!is_allocated())
    {
        throw std::runtime_error("DeviceMemory: cannot copy from unallocated buffer");
    }

    const auto& ctx = OpenCLContext::instance();
    if (!ctx.is_available())
    {
        throw std::runtime_error("DeviceMemory: OpenCL context not available");
    }

    cl_int err = clEnqueueReadBuffer(ctx.get_queue(),
        m_device_buffer,
        CL_TRUE, // blocking
        0,
        m_size_bytes,
        out_host_data,
        0,
        nullptr,
        nullptr);

    check_cl_error(err, "clEnqueueReadBuffer");
}

void DeviceMemory::sync() const
{
    const auto& ctx = OpenCLContext::instance();
    if (!ctx.is_available())
    {
        return;
    }

    cl_int err = clFinish(ctx.get_queue());
    check_cl_error(err, "clFinish");
}

void DeviceMemory::reallocate(Index new_size_bytes)
{
    if (new_size_bytes == 0)
    {
        throw std::runtime_error("DeviceMemory: cannot reallocate to 0 bytes");
    }

    // Release existing buffer
    if (m_device_buffer != nullptr)
    {
        clReleaseMemObject(m_device_buffer);
    }

    // Allocate new buffer
    const auto& ctx = OpenCLContext::instance();
    if (!ctx.is_available())
    {
        throw std::runtime_error("DeviceMemory: OpenCL context not available");
    }

    cl_int err = CL_SUCCESS;
    m_device_buffer =
        clCreateBuffer(ctx.get_context(), CL_MEM_READ_WRITE, new_size_bytes, nullptr, &err);

    if (err != CL_SUCCESS)
    {
        m_device_buffer = nullptr;
        m_size_bytes = 0;
        check_cl_error(err, "clCreateBuffer (reallocate)");
    }

    m_size_bytes = new_size_bytes;
}

// === StagingBuffer ===

StagingBuffer::StagingBuffer(Index num_bytes) : m_size_bytes(num_bytes)
{
    if (num_bytes == 0)
    {
        throw std::runtime_error("StagingBuffer: cannot allocate 0 bytes");
    }

    const auto& ctx = OpenCLContext::instance();
    if (!ctx.is_available())
    {
        throw std::runtime_error("StagingBuffer: OpenCL context not available");
    }

    // For now, use regular malloc (not pinned memory on all platforms)
    // TODO: Use clCreateBuffer with CL_MEM_ALLOC_HOST_PTR for true pinned memory
    m_host_data = std::malloc(num_bytes);

    if (m_host_data == nullptr)
    {
        throw std::runtime_error(
            "StagingBuffer: malloc failed for " + std::to_string(num_bytes) + " bytes");
    }

    NN_LOG_DEBUG("StagingBuffer allocated: " + std::to_string(num_bytes) + " bytes");
}

StagingBuffer::StagingBuffer(StagingBuffer&& other) noexcept
    : m_host_data(other.m_host_data), m_size_bytes(other.m_size_bytes)
{
    other.m_host_data = nullptr;
    other.m_size_bytes = 0;
}

StagingBuffer& StagingBuffer::operator=(StagingBuffer&& other) noexcept
{
    if (this != &other)
    {
        if (m_host_data != nullptr)
        {
            std::free(m_host_data);
        }

        m_host_data = other.m_host_data;
        m_size_bytes = other.m_size_bytes;

        other.m_host_data = nullptr;
        other.m_size_bytes = 0;
    }
    return *this;
}

StagingBuffer::~StagingBuffer() noexcept
{
    if (m_host_data != nullptr)
    {
        std::free(m_host_data);
        m_host_data = nullptr;
    }
}

} // namespace nn::opencl
