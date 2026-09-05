/**
 * @file src/core/tensor/opencl/OpenCLTensorBackendElementwise.cpp
 * @brief Out-of-place element-wise operations (add, subtract, multiply, divide, add_scalar,
 * multiply_scalar, divide_scalar).
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "OpenCLTensorBackendDetail.hpp"
#include "logging/Logger.hpp"
#include "tensor/opencl/DeviceMemory.hpp"
#include "tensor/opencl/KernelManager.hpp"
#include "tensor/opencl/OpenCLContext.hpp"
#include "tensor/opencl/OpenCLProfiling.hpp"
#include "tensor/opencl/OpenCLTensorBackend.hpp"

namespace nn
{

// Element-wise operations
OpenCLTensorBackend OpenCLTensorBackend::unary_elementwise(
    const char* kernel_name, const char* what) const
{
    // Device-resident fast path (see binary_elementwise() for the rationale).
    // The with-scalars entry point deliberately does NOT have this: no
    // launch_unary_scalar_resident call site existed for leaky_relu/clamp, and
    // adding one here would be a behaviour change smuggled into a refactor.
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_unary_resident(kernel_name, *this, fast_out, what)) return fast_out;
    }

    sync_gpu_if_needed();
    return run_unary_stages(kernel_name, what, {});
}

OpenCLTensorBackend OpenCLTensorBackend::exp() const
{
    return unary_elementwise("exp_kernel", "exp");
}

OpenCLTensorBackend OpenCLTensorBackend::sqrt() const
{
    return unary_elementwise("sqrt_kernel", "sqrt");
}

OpenCLTensorBackend OpenCLTensorBackend::square() const
{
    return unary_elementwise("square_kernel", "square");
}

// The (A, B, out, n) arg order every binary kernel in KernelManager.cpp
// expects, shared by both staging strategies below.
void OpenCLTensorBackend::enqueue_binary_kernel(const char* kernel_name,
    const char* what,
    cl_mem a_mem,
    cl_mem b_mem,
    cl_mem out_mem,
    std::size_t n)
{
    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
    const cl_uint n_u32 = static_cast<cl_uint>(n);

    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), what);
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), what);
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), what);
    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), what);

    const std::size_t local = 256;
    std::size_t global = round_up(n, local);
    check_cl_error(clEnqueueNDRangeKernel(
                       ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
        what);
    finish_queue_if_not_batching(ctx.get_queue(), what);
}

// Preferred staging: pooled buffers, so a hot loop stops re-allocating device
// memory every call. Empty optional = the pool is absent or exhausted.
std::optional<OpenCLTensorBackend> OpenCLTensorBackend::binary_stage_pooled(const char* kernel_name,
    const char* what,
    const OpenCLTensorBackend& other,
    std::size_t n) const
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr) return std::nullopt;

    const std::size_t bytes = n * sizeof(float);
    auto a_buf = pool->acquire(bytes);
    auto b_buf = pool->acquire(bytes);
    auto out_buf = pool->acquire(bytes);
    if (!a_buf || !b_buf || !out_buf) return std::nullopt;

    const auto& ctx = opencl::OpenCLContext::instance();
    copy_host_to_device(ctx.get_queue(), a_buf->buffer, host_data(), bytes, what);
    copy_host_to_device(ctx.get_queue(), b_buf->buffer, other.host_data(), bytes, what);
    enqueue_binary_kernel(kernel_name, what, a_buf->buffer, b_buf->buffer, out_buf->buffer, n);

    OpenCLHostStorage out(shape());
    // GPU-resident mode: keep result on GPU
    if (ensure_device_current("resident gate"))
    {
        OpenCLTensorBackend t;
        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
        t.m_has_gpu_memory = true;
        t.m_gpu_buffer = std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
        t.set_gpu_resident(true);
        t.m_needs_sync_to_host = true;
        return t;
    }

    copy_device_to_host(ctx.get_queue(), out_buf->buffer, out.mutable_data_ptr(), bytes, what);
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

// No pool (or it was exhausted): stage through one-shot device buffers, fully
// synchronous. No successor, so it always produces a result or throws.
OpenCLTensorBackend OpenCLTensorBackend::binary_stage_oneshot(const char* kernel_name,
    const char* what,
    const OpenCLTensorBackend& other,
    std::size_t n) const
{
    const std::size_t bytes = n * sizeof(float);
    opencl::DeviceMemory a_dev(bytes);
    opencl::DeviceMemory b_dev(bytes);
    opencl::DeviceMemory out_dev(bytes);
    a_dev.copy_to_device(host_data());
    b_dev.copy_to_device(other.host_data());

    enqueue_binary_kernel(kernel_name,
        what,
        a_dev.get_device_buffer(),
        b_dev.get_device_buffer(),
        out_dev.get_device_buffer(),
        n);

    OpenCLHostStorage out(shape());
    out_dev.copy_from_device(out.mutable_data_ptr());
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::binary_elementwise(
    const char* kernel_name, const char* what, const OpenCLTensorBackend& other) const
{
    // Device-resident fast path: operands' device copies are used directly
    // (uploaded once into their persistent buffers when stale); the result
    // stays on the GPU and is synced to host lazily.
    if (shape() == other.shape())
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_binary_resident(kernel_name, *this, other, fast_out, what)) return fast_out;
    }

    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();

    // Caller error, refused by type and by message before any CL call, so the
    // reader never has to disambiguate it from "no OpenCL device". Pinned by
    // OpenCLTensorBackendShapeMismatch.*.
    if (shape() != other.shape())
    {
        throw std::invalid_argument(
            std::string(what) + ": tensor shapes must match for an elementwise operation");
    }

    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl(what))
    {
        throw_opencl_only_failure(what, "OpenCL runtime unavailable");
    }

    try
    {
        const auto n = size();
        if (auto pooled = binary_stage_pooled(kernel_name, what, other, n)) return *pooled;
        return binary_stage_oneshot(kernel_name, what, other, n);
    }
    catch (const std::invalid_argument&)
    {
        throw; // a caller error must not be reclassified as a device failure
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure(what, e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::add(const OpenCLTensorBackend& other) const
{
    return binary_elementwise("add_kernel", "add", other);
}

OpenCLTensorBackend OpenCLTensorBackend::subtract(const OpenCLTensorBackend& other) const
{
    return binary_elementwise("subtract_kernel", "subtract", other);
}

OpenCLTensorBackend OpenCLTensorBackend::multiply(const OpenCLTensorBackend& other) const
{
    return binary_elementwise("multiply_kernel", "multiply", other);
}

OpenCLTensorBackend OpenCLTensorBackend::divide(const OpenCLTensorBackend& other) const
{
    return binary_elementwise("divide_kernel", "divide", other);
}

OpenCLTensorBackend OpenCLTensorBackend::add_scalar(float val) const
{
    // Device-resident fast path (see add() for the pattern rationale).
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_unary_scalar_resident("add_scalar_kernel", *this, val, fast_out, "add_scalar"))
            return fast_out;
    }

    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("add_scalar"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            OpenCLHostStorage out(shape());

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        input_buf->buffer,
                        host_data(),
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
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(add_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(add_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(host_data());

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
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(add_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("add_scalar", e.what());
        }
    }

    throw_opencl_only_failure("add_scalar", "OpenCL runtime unavailable");
}

OpenCLTensorBackend OpenCLTensorBackend::multiply_scalar(float val) const
{
    // Device-resident fast path (see add() for the pattern rationale).
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_unary_scalar_resident(
                "multiply_scalar_kernel", *this, val, fast_out, "multiply_scalar"))
            return fast_out;
    }

    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("multiply_scalar"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            OpenCLHostStorage out(shape());

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        input_buf->buffer,
                        host_data(),
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
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(multiply_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(multiply_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(host_data());

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
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(multiply_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("multiply_scalar", e.what());
        }
    }

    throw_opencl_only_failure("multiply_scalar", "OpenCL runtime unavailable");
}

OpenCLTensorBackend OpenCLTensorBackend::divide_scalar(float val) const
{
    // Device-resident fast path (see add() for the pattern rationale).
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_unary_scalar_resident(
                "divide_scalar_kernel", *this, val, fast_out, "divide_scalar"))
            return fast_out;
    }

    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("divide_scalar"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            OpenCLHostStorage out(shape());

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        input_buf->buffer,
                        host_data(),
                        bytes,
                        "clEnqueueWriteBuffer(divide_scalar, in)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("divide_scalar_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                        "clSetKernelArg(divide_scalar, in)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(divide_scalar, out)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &val),
                        "clSetKernelArg(divide_scalar, scalar)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(divide_scalar, size)");

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
                        "clEnqueueNDRangeKernel(divide_scalar)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(divide_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(divide_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(host_data());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("divide_scalar_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(divide_scalar, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(divide_scalar, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &val),
                "clSetKernelArg(divide_scalar, scalar)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(divide_scalar, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(divide_scalar)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(divide_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("divide_scalar", e.what());
        }
    }

    throw_opencl_only_failure("divide_scalar", "OpenCL runtime unavailable");
}

} // namespace nn
