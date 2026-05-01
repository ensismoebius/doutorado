/**
 * @file src/core/tensor/opencl/OpenCLContext.cpp
 * @brief OpenCL device and context management implementation.
 */

#include "nn/tensor/opencl/OpenCLContext.hpp"

#include <cassert>
#include <stdexcept>

#include "nn/logging/Logger.hpp"

namespace nn::opencl
{

namespace
{
/**
 * @brief Wrap OpenCL error code into runtime_error with context.
 */
void check_cl_error(cl_int err, const std::string& context)
{
    if (err != CL_SUCCESS)
    {
        std::string msg = "OpenCL error [" + context + "]: " + std::to_string(err);
        throw std::runtime_error(msg);
    }
}
} // namespace

OpenCLContext& OpenCLContext::instance()
{
    static OpenCLContext context;
    return context;
}

OpenCLContext::BatchScope::BatchScope()
{
    OpenCLContext& ctx = OpenCLContext::instance();
    if (!ctx.is_available())
    {
        return;
    }
    OpenCLContext::begin_batch();
    active = true;
}

OpenCLContext::BatchScope::~BatchScope()
{
    if (active)
    {
        OpenCLContext::end_batch();
    }
}

OpenCLContext::BatchScope::BatchScope(BatchScope&& other) noexcept
    : active(other.active)
{
    other.active = false;
}

auto OpenCLContext::BatchScope::operator=(BatchScope&& other) noexcept -> BatchScope&
{
    if (this != &other)
    {
        if (active)
        {
            OpenCLContext::end_batch();
        }
        active = other.active;
        other.active = false;
    }
    return *this;
}

OpenCLContext::OpenCLContext()
{
    initialize_device();
    if (m_is_available)
    {
        query_device_info();
        NN_LOG_INFO("OpenCL backend initialized: " + m_device_name);
    }
    else
    {
        NN_LOG_WARN("OpenCL not available; falling back to CPU backend");
    }
}

OpenCLContext::~OpenCLContext()
{
    if (m_queue) clReleaseCommandQueue(m_queue);
    if (m_context) clReleaseContext(m_context);
}

void OpenCLContext::initialize_device()
{
    cl_int err = CL_SUCCESS;

    // Get first platform
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0)
    {
        NN_LOG_DEBUG("No OpenCL platforms found");
        return;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    check_cl_error(err, "clGetPlatformIDs");

    m_platform = platforms[0];

    // Get platform name
    {
        char name_buf[256] = {0};
        err = clGetPlatformInfo(m_platform, CL_PLATFORM_NAME, sizeof(name_buf), name_buf, nullptr);
        check_cl_error(err, "clGetPlatformInfo CL_PLATFORM_NAME");
        m_platform_name = name_buf;
    }

    // Get GPU device first, fallback to CPU
    err = clGetDeviceIDs(m_platform, CL_DEVICE_TYPE_GPU, 1, &m_device, nullptr);
    if (err != CL_SUCCESS)
    {
        NN_LOG_DEBUG("No GPU devices found; trying CPU");
        err = clGetDeviceIDs(m_platform, CL_DEVICE_TYPE_CPU, 1, &m_device, nullptr);
        if (err != CL_SUCCESS)
        {
            NN_LOG_DEBUG("No CPU devices found either");
            return;
        }
    }

    // Get device name
    {
        char name_buf[256] = {0};
        err = clGetDeviceInfo(m_device, CL_DEVICE_NAME, sizeof(name_buf), name_buf, nullptr);
        check_cl_error(err, "clGetDeviceInfo CL_DEVICE_NAME");
        m_device_name = name_buf;
    }

    // Create context
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) m_platform, 0};
    m_context = clCreateContext(props, 1, &m_device, nullptr, nullptr, &err);
    check_cl_error(err, "clCreateContext");

    // Create command queue with profiling enabled.
    // Prefer OpenCL 2.0+ API and keep a compile-time fallback for older headers.
#if defined(CL_VERSION_2_0)
    const cl_queue_properties queue_props[] = {
        CL_QUEUE_PROPERTIES,
        static_cast<cl_queue_properties>(CL_QUEUE_PROFILING_ENABLE),
        0,
    };
    m_queue = clCreateCommandQueueWithProperties(m_context, m_device, queue_props, &err);
    check_cl_error(err, "clCreateCommandQueueWithProperties");
#else
    m_queue = clCreateCommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &err);
    check_cl_error(err, "clCreateCommandQueue");
#endif

    m_is_available = true;
}

void OpenCLContext::query_device_info()
{
    cl_int err = CL_SUCCESS;

    // Compute units
    err = clGetDeviceInfo(
        m_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(m_compute_units), &m_compute_units, nullptr);
    check_cl_error(err, "clGetDeviceInfo CL_DEVICE_MAX_COMPUTE_UNITS");

    // Local memory
    err = clGetDeviceInfo(m_device,
        CL_DEVICE_LOCAL_MEM_SIZE,
        sizeof(m_local_memory_size),
        &m_local_memory_size,
        nullptr);
    check_cl_error(err, "clGetDeviceInfo CL_DEVICE_LOCAL_MEM_SIZE");

    std::string log = "  Platform: " + m_platform_name + "\n";
    log += "  Device: " + m_device_name + "\n";
    log += "  Compute Units: " + std::to_string(m_compute_units) + "\n";
    log += "  Local Memory: " + std::to_string(m_local_memory_size / 1024) + " KiB";
    NN_LOG_INFO(log);
}

void OpenCLContext::flush() const
{
    if (!m_is_available) return;
    cl_int err = clFinish(m_queue);
    if (err != CL_SUCCESS)
    {
        throw std::runtime_error("clFinish failed: " + std::to_string(err));
    }
}

// Static batch mode control
bool OpenCLContext::s_batching = false;

void OpenCLContext::begin_batch()
{
    s_batching = true;
}

void OpenCLContext::end_batch()
{
    if (s_batching)
    {
        OpenCLContext::instance().flush();
        s_batching = false;
    }
}

bool OpenCLContext::is_batching()
{
    return s_batching;
}

} // namespace nn::opencl
