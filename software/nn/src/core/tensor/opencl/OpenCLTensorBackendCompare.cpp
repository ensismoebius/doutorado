/**
 * @file src/core/tensor/opencl/OpenCLTensorBackendCompare.cpp
 * @brief Elementwise and scalar comparison operations (compare_lt/gt/le/ge/eq and their *_scalar
 * and *_broadcast variants).
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

// Comparisons
// compare_lt's own extension: row/column broadcasting (a dim of 1 on either
// side stands in for "repeat"), computed on the CPU. Pulled out of
// compare_lt() itself so that function is left with only the dispatch
// between this and the equal-shape device path, which compare_gt/le/ge/eq
// already share via compare_stage_pooled/oneshot.
OpenCLTensorBackend OpenCLTensorBackend::compare_lt_broadcast(
    const OpenCLTensorBackend& other) const
{
    const auto& ls = shape();
    const auto& rs = other.shape();
    if (ls.size() != 2 || rs.size() != 2)
    {
        throw std::invalid_argument("compare_lt: broadcasting only supported for 2D tensors");
    }
    const bool rows_ok = (ls[0] == rs[0] || ls[0] == 1 || rs[0] == 1);
    const bool cols_ok = (ls[1] == rs[1] || ls[1] == 1 || rs[1] == 1);
    if (!rows_ok || !cols_ok)
    {
        throw std::invalid_argument("compare_lt: shapes are not broadcast-compatible");
    }

    const Index out_rows = std::max(ls[0], rs[0]);
    const Index out_cols = std::max(ls[1], rs[1]);
    sync_gpu();
    other.sync_gpu();
    OpenCLTensorBackend t(out_rows, out_cols);
    for (Index r = 0; r < out_rows; ++r)
    {
        for (Index c = 0; c < out_cols; ++c)
        {
            const Index ar = (ls[0] == 1) ? 0 : r;
            const Index ac = (ls[1] == 1) ? 0 : c;
            const Index br = (rs[0] == 1) ? 0 : r;
            const Index bc = (rs[1] == 1) ? 0 : c;
            t.m_backend->at(r, c) =
                (m_backend->at(ar, ac) < other.m_backend->at(br, bc)) ? 1.0f : 0.0f;
        }
    }
    t.m_needs_sync_to_device = true;
    t.m_needs_sync_to_host = false;
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_lt(const OpenCLTensorBackend& other) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();

    // compare_lt is the one comparison that tolerates a shape mismatch, by
    // broadcasting -- everything else this file calls a "shape mismatch" is a
    // caller error there instead.
    if (shape() != other.shape())
    {
        return compare_lt_broadcast(other);
    }

    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl("compare_lt"))
    {
        throw_opencl_only_failure("compare_lt", "OpenCL runtime unavailable");
    }

    try
    {
        const auto n = size();
        if (auto pooled = compare_stage_pooled("compare_lt_kernel", "compare_lt", other, n))
            return *pooled;
        return compare_stage_oneshot("compare_lt_kernel", "compare_lt", other, n);
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("compare_lt", e.what());
    }
}

// Preferred staging: pooled buffers. Empty optional = pool absent/exhausted.
std::optional<OpenCLTensorBackend> OpenCLTensorBackend::compare_stage_pooled(
    const char* kernel_name,
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
    copy_device_to_host(ctx.get_queue(), out_buf->buffer, out.mutable_data_ptr(), bytes, what);
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

// No pool (or exhausted): one-shot device buffers, fully synchronous.
OpenCLTensorBackend OpenCLTensorBackend::compare_stage_oneshot(const char* kernel_name,
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

OpenCLTensorBackend OpenCLTensorBackend::compare_elementwise(
    const char* kernel_name, const char* what, const OpenCLTensorBackend& other) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();

    // Caller error, refused by type and message before any CL call -- see
    // binary_elementwise() above for why this must not be
    // throw_opencl_only_failure. Pinned by
    // OpenCLTensorBackendShapeMismatch.CompareFamilyRefusesMismatchedShapes.
    if (shape() != other.shape())
    {
        throw std::invalid_argument(
            std::string(what) + ": tensor shapes must match for an elementwise comparison");
    }

    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl(what))
    {
        throw_opencl_only_failure(what, "OpenCL runtime unavailable");
    }

    try
    {
        const auto n = size();
        if (auto pooled = compare_stage_pooled(kernel_name, what, other, n)) return *pooled;
        return compare_stage_oneshot(kernel_name, what, other, n);
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

OpenCLTensorBackend OpenCLTensorBackend::compare_gt(const OpenCLTensorBackend& other) const
{
    return compare_elementwise("compare_gt_kernel", "compare_gt", other);
}

OpenCLTensorBackend OpenCLTensorBackend::compare_le(const OpenCLTensorBackend& other) const
{
    return compare_elementwise("compare_le_kernel", "compare_le", other);
}

OpenCLTensorBackend OpenCLTensorBackend::compare_ge(const OpenCLTensorBackend& other) const
{
    return compare_elementwise("compare_ge_kernel", "compare_ge", other);
}

OpenCLTensorBackend OpenCLTensorBackend::compare_eq(const OpenCLTensorBackend& other) const
{
    return compare_elementwise("compare_eq_kernel", "compare_eq", other);
}

OpenCLTensorBackend OpenCLTensorBackend::compare_lt_scalar(float value) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("compare_lt_scalar"))
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
                        "clEnqueueWriteBuffer(compare_lt_scalar, in)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("compare_lt_scalar_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                        "clSetKernelArg(compare_lt_scalar, in)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(compare_lt_scalar, out)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &value),
                        "clSetKernelArg(compare_lt_scalar, value)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(compare_lt_scalar, size)");

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
                        "clEnqueueNDRangeKernel(compare_lt_scalar)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_lt_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_lt_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(host_data());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("compare_lt_scalar_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(compare_lt_scalar, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(compare_lt_scalar, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &value),
                "clSetKernelArg(compare_lt_scalar, value)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(compare_lt_scalar, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(compare_lt_scalar)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_lt_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("compare_lt_scalar", e.what());
        }
    }

    throw_opencl_only_failure("compare_lt_scalar", "OpenCL runtime unavailable");
}

OpenCLTensorBackend OpenCLTensorBackend::compare_gt_scalar(float value) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("compare_gt_scalar"))
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
                        "clEnqueueWriteBuffer(compare_gt_scalar, in)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("compare_gt_scalar_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                        "clSetKernelArg(compare_gt_scalar, in)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(compare_gt_scalar, out)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &value),
                        "clSetKernelArg(compare_gt_scalar, value)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(compare_gt_scalar, size)");

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
                        "clEnqueueNDRangeKernel(compare_gt_scalar)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_gt_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_gt_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(host_data());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("compare_gt_scalar_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(compare_gt_scalar, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(compare_gt_scalar, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &value),
                "clSetKernelArg(compare_gt_scalar, value)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(compare_gt_scalar, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(compare_gt_scalar)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_gt_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("compare_gt_scalar", e.what());
        }
    }

    throw_opencl_only_failure("compare_gt_scalar", "OpenCL runtime unavailable");
}

OpenCLTensorBackend OpenCLTensorBackend::compare_le_scalar(float value) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("compare_le_scalar"))
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
                        "clEnqueueWriteBuffer(compare_le_scalar, in)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("compare_le_scalar_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                        "clSetKernelArg(compare_le_scalar, in)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(compare_le_scalar, out)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &value),
                        "clSetKernelArg(compare_le_scalar, value)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(compare_le_scalar, size)");

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
                        "clEnqueueNDRangeKernel(compare_le_scalar)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_le_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_le_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(host_data());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("compare_le_scalar_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(compare_le_scalar, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(compare_le_scalar, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &value),
                "clSetKernelArg(compare_le_scalar, value)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(compare_le_scalar, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(compare_le_scalar)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_le_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("compare_le_scalar", e.what());
        }
    }

    throw_opencl_only_failure("compare_le_scalar", "OpenCL runtime unavailable");
}

OpenCLTensorBackend OpenCLTensorBackend::compare_ge_scalar(float value) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("compare_ge_scalar"))
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
                        "clEnqueueWriteBuffer(compare_ge_scalar, in)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("compare_ge_scalar_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                        "clSetKernelArg(compare_ge_scalar, in)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(compare_ge_scalar, out)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &value),
                        "clSetKernelArg(compare_ge_scalar, value)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(compare_ge_scalar, size)");

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
                        "clEnqueueNDRangeKernel(compare_ge_scalar)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_ge_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_ge_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(host_data());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("compare_ge_scalar_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(compare_ge_scalar, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(compare_ge_scalar, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &value),
                "clSetKernelArg(compare_ge_scalar, value)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(compare_ge_scalar, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(compare_ge_scalar)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_ge_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("compare_ge_scalar", e.what());
        }
    }

    throw_opencl_only_failure("compare_ge_scalar", "OpenCL runtime unavailable");
}

OpenCLTensorBackend OpenCLTensorBackend::compare_eq_scalar(float value) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("compare_eq_scalar"))
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
                        "clEnqueueWriteBuffer(compare_eq_scalar, in)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("compare_eq_scalar_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                        "clSetKernelArg(compare_eq_scalar, in)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(compare_eq_scalar, out)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &value),
                        "clSetKernelArg(compare_eq_scalar, value)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(compare_eq_scalar, size)");

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
                        "clEnqueueNDRangeKernel(compare_eq_scalar)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_eq_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_eq_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(host_data());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("compare_eq_scalar_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(compare_eq_scalar, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(compare_eq_scalar, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &value),
                "clSetKernelArg(compare_eq_scalar, value)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(compare_eq_scalar, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(compare_eq_scalar)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_eq_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("compare_eq_scalar", e.what());
        }
    }

    throw_opencl_only_failure("compare_eq_scalar", "OpenCL runtime unavailable");
}

} // namespace nn
