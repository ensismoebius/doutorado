/**
 * @file src/core/tensor/opencl/CLBuffer.cpp
 * @brief GPU buffer management implementation.
 */

#include "tensor/opencl/CLBuffer.hpp"

#include <cassert>
#include <stdexcept>

#include "tensor/opencl/OpenCLContext.hpp"

namespace nn::opencl
{

namespace
{
void check_cl_error(cl_int err, const std::string& context)
{
    if (err != CL_SUCCESS)
    {
        throw std::runtime_error("OpenCL error [" + context + "]: " + std::to_string(err));
    }
}
} // namespace

CLBuffer::CLBuffer(size_t size_elements) : m_size_elements(size_elements)
{
    m_host_copy.resize(size_elements, 0.0f);

    const auto& ctx = OpenCLContext::instance();
    if (!ctx.is_available())
    {
        m_gpu_buffer = nullptr;
        m_is_dirty_gpu = true; // No GPU; CPU is the source
        m_is_dirty_host = false;
        return;
    }

    cl_int err = CL_SUCCESS;
    m_gpu_buffer = clCreateBuffer(
        ctx.get_context(), CL_MEM_READ_WRITE, size_elements * sizeof(float), nullptr, &err);
    check_cl_error(err, "clCreateBuffer");

    m_is_dirty_gpu = true;   // GPU buffer needs data from host
    m_is_dirty_host = false; // Host is current
}

CLBuffer::~CLBuffer()
{
    release();
}

void CLBuffer::release()
{
    if (m_gpu_buffer)
    {
        clReleaseMemObject(m_gpu_buffer);
        m_gpu_buffer = nullptr;
    }
}

void CLBuffer::write_to_device(cl_command_queue queue)
{
    if (!m_gpu_buffer)
    {
        // No GPU; nothing to do
        m_is_dirty_gpu = false;
        return;
    }

    if (!m_is_dirty_gpu)
    {
        // GPU already has current data
        return;
    }

    cl_int err = clEnqueueWriteBuffer(queue, // command queue
        m_gpu_buffer,                        // destination buffer
        CL_FALSE,                            // non-blocking
        0,                                   // offset
        size_bytes(),                        // size
        m_host_copy.data(),                  // source data
        0,                                   // num events in wait list
        nullptr,                             // wait list
        nullptr                              // event (for profiling)
    );
    check_cl_error(err, "clEnqueueWriteBuffer");

    m_is_dirty_gpu = false; // GPU is now current
}

void CLBuffer::read_from_device(cl_command_queue queue)
{
    if (!m_gpu_buffer)
    {
        // No GPU; nothing to do
        m_is_dirty_host = false;
        return;
    }

    if (!m_is_dirty_host)
    {
        // Host already has current data
        return;
    }

    cl_int err = clEnqueueReadBuffer(queue, // command queue
        m_gpu_buffer,                       // source buffer
        CL_TRUE,                            // blocking (wait for completion)
        0,                                  // offset
        size_bytes(),                       // size
        m_host_copy.data(),                 // destination
        0,                                  // num events in wait list
        nullptr,                            // wait list
        nullptr                             // event
    );
    check_cl_error(err, "clEnqueueReadBuffer");

    m_is_dirty_host = false; // Host is now current
}

} // namespace nn::opencl
