/**
 * @file src/core/tensor/OpenCLTensorBackend.cpp
 * @brief OpenCL tensor backend implementation (Phase 1: CPU fallback).
 *
 * Phase 1 delegates all operations to EigenTensorBackend for correctness.
 * GPU implementations will be added incrementally as needed.
 */

#include "nn/tensor/OpenCLTensorBackend.hpp"

#include <cassert>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "nn/logging/Logger.hpp"
#include "nn/tensor/DeviceMemory.hpp"
#include "nn/tensor/EigenTensorBackend.hpp"
#include "nn/tensor/KernelManager.hpp"
#include "nn/tensor/OpenCLContext.hpp"

namespace nn
{

namespace
{
void check_cl_error(cl_int err, const char* context)
{
    if (err != CL_SUCCESS)
    {
        throw std::runtime_error(
            std::string("OpenCL error in ") + context + ": " + std::to_string(err));
    }
}

bool can_use_opencl()
{
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
    return false;
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
    return false;
#endif
    return opencl::OpenCLContext::instance().is_available();
}

void warn_opencl_cpu_fallback_once(const std::string& operation, const std::string& reason)
{
    static std::mutex warned_mutex;
    static std::unordered_set<std::string> warned_messages;

    const std::string message =
        "OPENCL BACKEND FALLING BACK TO CPU for " + operation + ": " + reason;
    std::lock_guard<std::mutex> lock(warned_mutex);
    if (warned_messages.insert(message).second)
    {
        NN_LOG_WARN(message);
    }
}

bool can_use_opencl(const char* operation)
{
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
    warn_opencl_cpu_fallback_once(operation, "AddressSanitizer build disables OpenCL execution");
    return false;
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
    warn_opencl_cpu_fallback_once(operation, "AddressSanitizer build disables OpenCL execution");
    return false;
#endif
    if (!can_use_opencl())
    {
        warn_opencl_cpu_fallback_once(operation, "OpenCL runtime or device is not available");
        return false;
    }
    return true;
}

void warn_opencl_unimplemented_once(const char* operation)
{
    warn_opencl_cpu_fallback_once(
        operation, "operation is not implemented on the OpenCL backend yet");
}

std::size_t round_up(std::size_t global, std::size_t local)
{
    if (local == 0)
    {
        return global;
    }
    const std::size_t rem = global % local;
    return rem == 0 ? global : (global + (local - rem));
}

void copy_host_to_device(cl_command_queue queue,
    cl_mem device_buffer,
    const float* host_data,
    std::size_t bytes,
    const char* context)
{
    check_cl_error(clEnqueueWriteBuffer(
                       queue, device_buffer, CL_TRUE, 0, bytes, host_data, 0, nullptr, nullptr),
        context);
}

void copy_device_to_host(cl_command_queue queue,
    cl_mem device_buffer,
    float* host_data,
    std::size_t bytes,
    const char* context)
{
    check_cl_error(clEnqueueReadBuffer(
                       queue, device_buffer, CL_TRUE, 0, bytes, host_data, 0, nullptr, nullptr),
        context);
}
} // namespace

// Constructors
OpenCLTensorBackend::OpenCLTensorBackend(Index rows, Index cols)
    : m_backend(std::make_unique<EigenTensorBackend>(rows, cols))
{
}

OpenCLTensorBackend::OpenCLTensorBackend(Index d1, Index d2, Index d3, Index d4)
    : m_backend(std::make_unique<EigenTensorBackend>(d1, d2, d3, d4))
{
}

OpenCLTensorBackend::OpenCLTensorBackend(const std::vector<Index>& shape)
    : m_backend(std::make_unique<EigenTensorBackend>(shape))
{
}

OpenCLTensorBackend::OpenCLTensorBackend(const OpenCLTensorBackend& other)
{
    if (other.m_backend)
    {
        m_backend = std::make_unique<EigenTensorBackend>(*other.m_backend);
    }
    if (other.m_grad_backend)
    {
        m_grad_backend = std::make_unique<OpenCLTensorBackend>(*other.m_grad_backend);
    }
}

OpenCLTensorBackend& OpenCLTensorBackend::operator=(const OpenCLTensorBackend& other)
{
    if (this != &other)
    {
        m_backend =
            other.m_backend ? std::make_unique<EigenTensorBackend>(*other.m_backend) : nullptr;
        m_grad_backend = other.m_grad_backend
                             ? std::make_unique<OpenCLTensorBackend>(*other.m_grad_backend)
                             : nullptr;
    }
    return *this;
}

OpenCLTensorBackend::~OpenCLTensorBackend() = default;

// Static Factories
OpenCLTensorBackend OpenCLTensorBackend::zeros(Index rows, Index cols)
{
    OpenCLTensorBackend t(rows, cols);
    *t.m_backend = EigenTensorBackend::zeros(rows, cols);
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::ones(Index rows, Index cols)
{
    OpenCLTensorBackend t(rows, cols);
    *t.m_backend = EigenTensorBackend::ones(rows, cols);
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::random(Index rows, Index cols)
{
    OpenCLTensorBackend t(rows, cols);
    *t.m_backend = EigenTensorBackend::random(rows, cols);
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::random(Index rows, Index cols, std::mt19937& rng)
{
    OpenCLTensorBackend t(rows, cols);
    *t.m_backend = EigenTensorBackend::random(rows, cols, rng);
    return t;
}

// Shape & Access
const std::vector<Index>& OpenCLTensorBackend::shape() const
{
    return m_backend->shape();
}

void OpenCLTensorBackend::reshape(const std::vector<Index>& new_shape)
{
    m_backend->reshape(new_shape);
}

Index OpenCLTensorBackend::rows() const
{
    return m_backend->rows();
}

Index OpenCLTensorBackend::cols() const
{
    return m_backend->cols();
}

Index OpenCLTensorBackend::size() const
{
    return m_backend->size();
}

// N-D access
float& OpenCLTensorBackend::at(Index i)
{
    return m_backend->at(i);
}

const float& OpenCLTensorBackend::at(Index i) const
{
    return m_backend->at(i);
}

float& OpenCLTensorBackend::at(Index row, Index col)
{
    return m_backend->at(row, col);
}

const float& OpenCLTensorBackend::at(Index row, Index col) const
{
    return m_backend->at(row, col);
}

float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3, Index d4)
{
    return m_backend->at(d1, d2, d3, d4);
}

const float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3, Index d4) const
{
    return m_backend->at(d1, d2, d3, d4);
}

float& OpenCLTensorBackend::at(const std::vector<Index>& indices)
{
    return m_backend->at(indices);
}

const float& OpenCLTensorBackend::at(const std::vector<Index>& indices) const
{
    return m_backend->at(indices);
}

// In-place operations
void OpenCLTensorBackend::add_inplace(const OpenCLTensorBackend& other)
{
    warn_opencl_unimplemented_once("add_inplace");
    m_backend->add_inplace(*other.m_backend);
}

void OpenCLTensorBackend::subtract_inplace(const OpenCLTensorBackend& other)
{
    warn_opencl_unimplemented_once("subtract_inplace");
    m_backend->subtract_inplace(*other.m_backend);
}

void OpenCLTensorBackend::multiply_inplace(const OpenCLTensorBackend& other)
{
    warn_opencl_unimplemented_once("multiply_inplace");
    m_backend->multiply_inplace(*other.m_backend);
}

void OpenCLTensorBackend::divide_inplace(const OpenCLTensorBackend& other)
{
    warn_opencl_unimplemented_once("divide_inplace");
    m_backend->divide_inplace(*other.m_backend);
}

void OpenCLTensorBackend::add_scalar_inplace(float val)
{
    warn_opencl_unimplemented_once("add_scalar_inplace");
    m_backend->add_scalar_inplace(val);
}

void OpenCLTensorBackend::multiply_scalar_inplace(float val)
{
    warn_opencl_unimplemented_once("multiply_scalar_inplace");
    m_backend->multiply_scalar_inplace(val);
}

void OpenCLTensorBackend::divide_scalar_inplace(float val)
{
    warn_opencl_unimplemented_once("divide_scalar_inplace");
    m_backend->divide_scalar_inplace(val);
}

void OpenCLTensorBackend::sqrt_inplace()
{
    warn_opencl_unimplemented_once("sqrt_inplace");
    m_backend->sqrt_inplace();
}

void OpenCLTensorBackend::square_inplace()
{
    warn_opencl_unimplemented_once("square_inplace");
    m_backend->square_inplace();
}

void OpenCLTensorBackend::add_col_vector_to_rows_inplace(const OpenCLTensorBackend& col_vector)
{
    warn_opencl_unimplemented_once("add_col_vector_to_rows_inplace");
    m_backend->add_col_vector_to_rows_inplace(*col_vector.m_backend);
}

// Element-wise operations
OpenCLTensorBackend OpenCLTensorBackend::exp() const
{
    if (can_use_opencl("exp"))
    {
        try
        {
            const auto n = size();
            if (n == 0)
            {
                OpenCLTensorBackend empty(*this);
                return empty;
            }

            auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(exp, in)");

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("exp_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                        "clSetKernelArg(exp, in)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(exp, out)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(exp, size)");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);
                    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                       kernel,
                                       1,
                                       nullptr,
                                       &global,
                                       &local,
                                       0,
                                       nullptr,
                                       nullptr),
                        "clEnqueueNDRangeKernel(exp)");
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(exp)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(exp, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("exp_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "clSetKernelArg(exp, in)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "clSetKernelArg(exp, out)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "clSetKernelArg(exp, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(exp)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(exp)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL exp fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->exp());
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::sqrt() const
{
    warn_opencl_unimplemented_once("sqrt");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->sqrt());
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::square() const
{
    warn_opencl_unimplemented_once("square");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->square());
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::add(const OpenCLTensorBackend& other) const
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("add", "OpenCL path requires matching tensor shapes");
    }
    else if (can_use_opencl("add"))
    {
        try
        {
            auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (a_buf && b_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(add, a)");
                    copy_host_to_device(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(add, b)");

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("add_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                        "clSetKernelArg(add, a)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                        "clSetKernelArg(add, b)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(add, out)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(add, size)");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);
                    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                       kernel,
                                       1,
                                       nullptr,
                                       &global,
                                       &local,
                                       0,
                                       nullptr,
                                       nullptr),
                        "clEnqueueNDRangeKernel(add)");
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(add)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(add, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("add_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(add, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(add, b)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), "clSetKernelArg(add, out)");
            check_cl_error(
                clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "clSetKernelArg(add, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(add)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(add)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL add fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->add(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::subtract(const OpenCLTensorBackend& other) const
{
    warn_opencl_unimplemented_once("subtract");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->subtract(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::multiply(const OpenCLTensorBackend& other) const
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("multiply", "OpenCL path requires matching tensor shapes");
    }
    else if (can_use_opencl("multiply"))
    {
        try
        {
            auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (a_buf && b_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(multiply, a)");
                    copy_host_to_device(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(multiply, b)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("multiply_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                        "clSetKernelArg(multiply, a)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                        "clSetKernelArg(multiply, b)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(multiply, out)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(multiply, size)");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);
                    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                       kernel,
                                       1,
                                       nullptr,
                                       &global,
                                       &local,
                                       0,
                                       nullptr,
                                       nullptr),
                        "clEnqueueNDRangeKernel(multiply)");
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(multiply)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(multiply, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("multiply_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(multiply, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(multiply, b)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(multiply, out)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(multiply, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(multiply)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(multiply)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL multiply fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->multiply(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::divide(const OpenCLTensorBackend& other) const
{
    warn_opencl_unimplemented_once("divide");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->divide(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::add_scalar(float val) const
{
    if (can_use_opencl("add_scalar"))
    {
        try
        {
            auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(add_scalar, in)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("add_scalar_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                        "clSetKernelArg(add_scalar, in)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(add_scalar, out)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &val),
                        "clSetKernelArg(add_scalar, scalar)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(add_scalar, size)");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);
                    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                       kernel,
                                       1,
                                       nullptr,
                                       &global,
                                       &local,
                                       0,
                                       nullptr,
                                       nullptr),
                        "clEnqueueNDRangeKernel(add_scalar)");
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(add_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(add_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("add_scalar_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(add_scalar, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(add_scalar, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &val),
                "clSetKernelArg(add_scalar, scalar)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(add_scalar, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(add_scalar)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(add_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL add_scalar fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->add_scalar(val));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::multiply_scalar(float val) const
{
    if (can_use_opencl("multiply_scalar"))
    {
        try
        {
            auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(multiply_scalar, in)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("multiply_scalar_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                        "clSetKernelArg(multiply_scalar, in)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(multiply_scalar, out)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &val),
                        "clSetKernelArg(multiply_scalar, scalar)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(multiply_scalar, size)");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);
                    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                       kernel,
                                       1,
                                       nullptr,
                                       &global,
                                       &local,
                                       0,
                                       nullptr,
                                       nullptr),
                        "clEnqueueNDRangeKernel(multiply_scalar)");
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(multiply_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(multiply_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("multiply_scalar_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(multiply_scalar, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(multiply_scalar, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &val),
                "clSetKernelArg(multiply_scalar, scalar)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(multiply_scalar, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(multiply_scalar)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(multiply_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL multiply_scalar fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->multiply_scalar(val));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::divide_scalar(float val) const
{
    warn_opencl_unimplemented_once("divide_scalar");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->divide_scalar(val));
    return t;
}

// Reduction
OpenCLTensorBackend OpenCLTensorBackend::rowwise_sum() const
{
    if (shape().size() != 2)
    {
        warn_opencl_cpu_fallback_once("rowwise_sum", "OpenCL path requires rank-2 tensors");
    }
    else if (can_use_opencl("rowwise_sum"))
    {
        try
        {
            auto& ctx = opencl::OpenCLContext::instance();
            const Index num_rows = rows();
            const Index num_cols = cols();
            const std::size_t input_bytes = num_rows * num_cols * sizeof(float);
            const std::size_t output_bytes = num_rows * sizeof(float);

            EigenTensorBackend out(num_rows, 1);
            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto input_buf = pool->acquire(input_bytes);
                auto out_buf = pool->acquire(output_bytes);
                if (input_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        input_bytes,
                        "clEnqueueWriteBuffer(rowwise_sum, input)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("rowwise_sum_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint rows_u32 = static_cast<cl_uint>(num_rows);
                    const cl_uint cols_u32 = static_cast<cl_uint>(num_cols);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                        "clSetKernelArg(rowwise_sum, input)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(rowwise_sum, output)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32),
                        "clSetKernelArg(rowwise_sum, rows)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32),
                        "clSetKernelArg(rowwise_sum, cols)");

                    const std::size_t global = static_cast<std::size_t>(num_rows);
                    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                       kernel,
                                       1,
                                       nullptr,
                                       &global,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr),
                        "clEnqueueNDRangeKernel(rowwise_sum)");
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(rowwise_sum)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        output_bytes,
                        "clEnqueueReadBuffer(rowwise_sum, output)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(input_bytes);
            opencl::DeviceMemory out_dev(output_bytes);

            input_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("rowwise_sum_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint rows_u32 = static_cast<cl_uint>(num_rows);
            const cl_uint cols_u32 = static_cast<cl_uint>(num_cols);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(rowwise_sum, input)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(rowwise_sum, output)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32),
                "clSetKernelArg(rowwise_sum, rows)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32),
                "clSetKernelArg(rowwise_sum, cols)");

            const std::size_t global = static_cast<std::size_t>(num_rows);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(rowwise_sum)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(rowwise_sum)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL rowwise_sum fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->rowwise_sum());
    return t;
}

// Linear algebra
OpenCLTensorBackend OpenCLTensorBackend::matmul(const OpenCLTensorBackend& other) const
{
    if (shape().size() != 2 || other.shape().size() != 2)
    {
        warn_opencl_cpu_fallback_once("matmul", "OpenCL path requires rank-2 tensors");
    }
    else if (cols() != other.rows())
    {
        warn_opencl_cpu_fallback_once("matmul", "OpenCL path requires lhs.cols() == rhs.rows()");
    }
    else if (can_use_opencl("matmul"))
    {
        try
        {
            auto& ctx = opencl::OpenCLContext::instance();
            const Index m = rows();
            const Index k = cols();
            const Index n = other.cols();

            const std::size_t a_bytes = m * k * sizeof(float);
            const std::size_t b_bytes = k * n * sizeof(float);
            const std::size_t c_bytes = m * n * sizeof(float);
            EigenTensorBackend out(m, n);

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto a_buf = pool->acquire(a_bytes);
                auto b_buf = pool->acquire(b_bytes);
                auto c_buf = pool->acquire(c_bytes);
                if (a_buf && b_buf && c_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        a_bytes,
                        "clEnqueueWriteBuffer(matmul, a)");
                    copy_host_to_device(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        b_bytes,
                        "clEnqueueWriteBuffer(matmul, b)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("matmul_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_mem c_mem = c_buf->buffer;
                    const cl_uint m_u32 = static_cast<cl_uint>(m);
                    const cl_uint n_u32 = static_cast<cl_uint>(n);
                    const cl_uint k_u32 = static_cast<cl_uint>(k);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                        "clSetKernelArg(matmul, a)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                        "clSetKernelArg(matmul, b)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem),
                        "clSetKernelArg(matmul, c)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &m_u32),
                        "clSetKernelArg(matmul, m)");
                    check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(matmul, n)");
                    check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &k_u32),
                        "clSetKernelArg(matmul, k)");

                    const std::size_t global[2] = {m, n};
                    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                       kernel,
                                       2,
                                       nullptr,
                                       global,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr),
                        "clEnqueueNDRangeKernel(matmul)");
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(matmul)");

                    copy_device_to_host(ctx.get_queue(),
                        c_buf->buffer,
                        out.mutable_data_ptr(),
                        c_bytes,
                        "clEnqueueReadBuffer(matmul, c)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory a_dev(a_bytes);
            opencl::DeviceMemory b_dev(b_bytes);
            opencl::DeviceMemory out_dev(c_bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("matmul_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem c_mem = out_dev.get_device_buffer();
            const cl_uint m_u32 = static_cast<cl_uint>(m);
            const cl_uint n_u32 = static_cast<cl_uint>(n);
            const cl_uint k_u32 = static_cast<cl_uint>(k);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(matmul, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(matmul, b)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem), "clSetKernelArg(matmul, c)");
            check_cl_error(
                clSetKernelArg(kernel, 3, sizeof(cl_uint), &m_u32), "clSetKernelArg(matmul, m)");
            check_cl_error(
                clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32), "clSetKernelArg(matmul, n)");
            check_cl_error(
                clSetKernelArg(kernel, 5, sizeof(cl_uint), &k_u32), "clSetKernelArg(matmul, k)");

            const std::size_t global[2] = {m, n};
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(matmul)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(matmul)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL matmul fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->matmul(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed(const OpenCLTensorBackend& other) const
{
    warn_opencl_unimplemented_once("matmul_transposed");
    OpenCLTensorBackend t;
    t.m_backend =
        std::make_unique<EigenTensorBackend>(m_backend->matmul_transposed(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::transpose() const
{
    if (shape().size() != 2)
    {
        warn_opencl_cpu_fallback_once("transpose", "OpenCL path requires a rank-2 tensor");
    }
    else if (can_use_opencl("transpose"))
    {
        try
        {
            auto& ctx = opencl::OpenCLContext::instance();
            const Index in_rows = rows();
            const Index in_cols = cols();
            const std::size_t bytes = in_rows * in_cols * sizeof(float);

            EigenTensorBackend out(in_cols, in_rows);
            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto in_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (in_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        in_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(transpose, in)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("transpose_kernel");
                    const cl_mem in_mem = in_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint rows_u32 = static_cast<cl_uint>(in_rows);
                    const cl_uint cols_u32 = static_cast<cl_uint>(in_cols);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                        "clSetKernelArg(transpose, in)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(transpose, out)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32),
                        "clSetKernelArg(transpose, rows)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32),
                        "clSetKernelArg(transpose, cols)");

                    const std::size_t global[2] = {in_rows, in_cols};
                    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                       kernel,
                                       2,
                                       nullptr,
                                       global,
                                       nullptr,
                                       0,
                                       nullptr,
                                       nullptr),
                        "clEnqueueNDRangeKernel(transpose)");
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(transpose)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(transpose, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory in_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);

            in_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("transpose_kernel");
            const cl_mem in_mem = in_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint rows_u32 = static_cast<cl_uint>(in_rows);
            const cl_uint cols_u32 = static_cast<cl_uint>(in_cols);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(transpose, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(transpose, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32),
                "clSetKernelArg(transpose, rows)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32),
                "clSetKernelArg(transpose, cols)");

            const std::size_t global[2] = {in_rows, in_cols};
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(transpose)");
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(transpose)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL transpose fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->transpose());
    return t;
}

// Comparisons
OpenCLTensorBackend OpenCLTensorBackend::compare_lt(const OpenCLTensorBackend& other) const
{
    warn_opencl_unimplemented_once("compare_lt");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_lt(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_gt(const OpenCLTensorBackend& other) const
{
    warn_opencl_unimplemented_once("compare_gt");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_gt(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_le(const OpenCLTensorBackend& other) const
{
    warn_opencl_unimplemented_once("compare_le");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_le(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_ge(const OpenCLTensorBackend& other) const
{
    warn_opencl_unimplemented_once("compare_ge");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_ge(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_eq(const OpenCLTensorBackend& other) const
{
    warn_opencl_unimplemented_once("compare_eq");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_eq(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_lt_scalar(float value) const
{
    warn_opencl_unimplemented_once("compare_lt_scalar");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_lt_scalar(value));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_gt_scalar(float value) const
{
    warn_opencl_unimplemented_once("compare_gt_scalar");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_gt_scalar(value));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_le_scalar(float value) const
{
    warn_opencl_unimplemented_once("compare_le_scalar");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_le_scalar(value));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_ge_scalar(float value) const
{
    warn_opencl_unimplemented_once("compare_ge_scalar");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_ge_scalar(value));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_eq_scalar(float value) const
{
    warn_opencl_unimplemented_once("compare_eq_scalar");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_eq_scalar(value));
    return t;
}

// Gradient management
OpenCLTensorBackend& OpenCLTensorBackend::grad_ref()
{
    if (!m_grad_backend)
    {
        m_grad_backend = std::make_unique<OpenCLTensorBackend>(shape());
        m_grad_backend->m_backend = std::make_unique<EigenTensorBackend>(shape());
    }
    return *m_grad_backend;
}

const OpenCLTensorBackend& OpenCLTensorBackend::get_grad() const
{
    if (!m_grad_backend)
    {
        throw std::runtime_error("Gradient not allocated");
    }
    return *m_grad_backend;
}

void OpenCLTensorBackend::zero_grad()
{
    if (m_grad_backend)
    {
        *m_grad_backend->m_backend = EigenTensorBackend::zeros(rows(), cols());
    }
}

// Static GPU buffer pool management
namespace
{
std::unique_ptr<tensor::GPUBufferPool> g_buffer_pool;
std::mutex g_buffer_pool_mutex;
} // namespace

void OpenCLTensorBackend::init_buffer_pool(void* context, void* queue)
{
    std::lock_guard<std::mutex> lock(g_buffer_pool_mutex);
    if (!g_buffer_pool && context && queue)
    {
        g_buffer_pool = std::make_unique<tensor::GPUBufferPool>(
            static_cast<cl_context>(context), static_cast<cl_command_queue>(queue));
        NN_LOG_INFO("GPU buffer pool initialized");
    }
}

void OpenCLTensorBackend::shutdown_buffer_pool()
{
    std::lock_guard<std::mutex> lock(g_buffer_pool_mutex);
    if (g_buffer_pool)
    {
        g_buffer_pool.reset();
        NN_LOG_INFO("GPU buffer pool shut down");
    }
}

tensor::GPUBufferPool* OpenCLTensorBackend::get_buffer_pool()
{
    std::lock_guard<std::mutex> lock(g_buffer_pool_mutex);
    return g_buffer_pool.get();
}
} // namespace nn
