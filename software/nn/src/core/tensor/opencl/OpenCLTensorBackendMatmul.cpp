/**
 * @file src/core/tensor/opencl/OpenCLTensorBackendMatmul.cpp
 * @brief Matrix multiplication family: matmul, matmul_transposed, matmul_lhs_transposed, and the
 * fused bias-activated variants.
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

// Linear algebra
// The (a, b, c, m, n, k) arg order and plain (no local-size) 2D NDRange that
// matmul_kernel and matmul_rhs_transposed_kernel both use -- the two differ
// only in which of A/B they read transposed, which is baked into the kernel
// itself, not into how it is launched. Shared by both entry points below.
void OpenCLTensorBackend::enqueue_matmul_kernel(const char* kernel_name,
    const char* what,
    cl_mem a_mem,
    cl_mem b_mem,
    cl_mem c_mem,
    Index m,
    Index n,
    Index k)
{
    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
    const cl_uint m_u32 = static_cast<cl_uint>(m);
    const cl_uint n_u32 = static_cast<cl_uint>(n);
    const cl_uint k_u32 = static_cast<cl_uint>(k);

    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), what);
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), what);
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem), what);
    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &m_u32), what);
    check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32), what);
    check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &k_u32), what);

    const std::size_t global[2] = {m, n};
    check_cl_error(clEnqueueNDRangeKernel(
                       ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
        what);
}

// Stage 1: both operands already resident, multiplied with no transfer at
// all beyond whatever host writes are still pending on either one.
//
// `finish_after_launch` exists only because matmul() and matmul_transposed()
// disagreed on it before this decomposition: matmul's resident path called
// finish_queue_if_not_batching after the launch, matmul_transposed's did not.
// Whether that gap in matmul_transposed was deliberate (rely on the later
// lazy host-sync to wait) or a bug (the fast path racing ahead of the
// kernel), pooled/oneshot both finish either way -- only resident differs,
// and only for this one function. Preserved exactly rather than guessed at.
OpenCLTensorBackend OpenCLTensorBackend::matmul_stage_resident(const char* kernel_name,
    const char* what,
    const OpenCLTensorBackend& other,
    Index m,
    Index n,
    Index k,
    bool finish_after_launch) const
{
    const auto& ctx = opencl::OpenCLContext::instance();
    if (m_needs_sync_to_device)
    {
        copy_host_to_device(
            ctx.get_queue(), m_gpu_buffer->buffer, host_data(), m * k * sizeof(float), what);
        m_needs_sync_to_device = false;
    }
    if (other.m_needs_sync_to_device)
    {
        copy_host_to_device(ctx.get_queue(),
            other.m_gpu_buffer->buffer,
            other.host_data(),
            k * n * sizeof(float),
            what);
        other.m_needs_sync_to_device = false;
    }

    OpenCLTensorBackend t(m, n);
    t.set_gpu_resident(true);
    enqueue_matmul_kernel(kernel_name,
        what,
        m_gpu_buffer->buffer,
        other.m_gpu_buffer->buffer,
        t.m_gpu_buffer->buffer,
        m,
        n,
        k);
    if (finish_after_launch)
    {
        finish_queue_if_not_batching(ctx.get_queue(), what);
    }
    t.m_needs_sync_to_host = true;
    t.m_needs_sync_to_device = false;
    return t;
}

// Stage 2: staged through pooled buffers. Empty optional = pool
// absent/exhausted.
std::optional<OpenCLTensorBackend> OpenCLTensorBackend::matmul_stage_pooled(const char* kernel_name,
    const char* what,
    const OpenCLTensorBackend& other,
    Index m,
    Index n,
    Index k) const
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr) return std::nullopt;

    const std::size_t a_bytes = m * k * sizeof(float);
    const std::size_t b_bytes = k * n * sizeof(float);
    const std::size_t c_bytes = m * n * sizeof(float);
    auto a_buf = pool->acquire(a_bytes);
    auto b_buf = pool->acquire(b_bytes);
    auto c_buf = pool->acquire(c_bytes);
    if (!a_buf || !b_buf || !c_buf) return std::nullopt;

    const auto& ctx = opencl::OpenCLContext::instance();
    copy_host_to_device(ctx.get_queue(), a_buf->buffer, host_data(), a_bytes, what);
    copy_host_to_device(ctx.get_queue(), b_buf->buffer, other.host_data(), b_bytes, what);
    enqueue_matmul_kernel(kernel_name, what, a_buf->buffer, b_buf->buffer, c_buf->buffer, m, n, k);
    finish_queue_if_not_batching(ctx.get_queue(), what);

    OpenCLHostStorage out(m, n);
    copy_device_to_host(ctx.get_queue(), c_buf->buffer, out.mutable_data_ptr(), c_bytes, what);
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

// Stage 3: no pool, one-shot device buffers, fully synchronous. No
// successor, so it always produces a result or throws.
OpenCLTensorBackend OpenCLTensorBackend::matmul_stage_oneshot(const char* kernel_name,
    const char* what,
    const OpenCLTensorBackend& other,
    Index m,
    Index n,
    Index k) const
{
    const std::size_t a_bytes = m * k * sizeof(float);
    const std::size_t b_bytes = k * n * sizeof(float);
    const std::size_t c_bytes = m * n * sizeof(float);
    opencl::DeviceMemory a_dev(a_bytes);
    opencl::DeviceMemory b_dev(b_bytes);
    opencl::DeviceMemory out_dev(c_bytes);
    a_dev.copy_to_device(host_data());
    b_dev.copy_to_device(other.host_data());

    enqueue_matmul_kernel(kernel_name,
        what,
        a_dev.get_device_buffer(),
        b_dev.get_device_buffer(),
        out_dev.get_device_buffer(),
        m,
        n,
        k);
    finish_queue_if_not_batching(opencl::OpenCLContext::instance().get_queue(), what);

    OpenCLHostStorage out(m, n);
    out_dev.copy_from_device(out.mutable_data_ptr());
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

// The three stages in cheapest-transfer-first order, shared by matmul() and
// matmul_transposed(): identical launch shape (a, b, c, m, n, k; plain 2D
// NDRange), differing only in kernel_name, in how the caller derived m/n/k
// from its own transpose semantics, and in `finish_after_launch` (see
// matmul_stage_resident).
OpenCLTensorBackend OpenCLTensorBackend::matmul_dispatch(const char* kernel_name,
    const char* what,
    const OpenCLTensorBackend& other,
    Index m,
    Index n,
    Index k,
    bool finish_after_launch) const
{
    if (ensure_device_current("resident gate") && other.ensure_device_current("resident gate"))
    {
        return matmul_stage_resident(kernel_name, what, other, m, n, k, finish_after_launch);
    }
    if (auto pooled = matmul_stage_pooled(kernel_name, what, other, m, n, k)) return *pooled;
    return matmul_stage_oneshot(kernel_name, what, other, m, n, k);
}

OpenCLTensorBackend OpenCLTensorBackend::matmul(const OpenCLTensorBackend& other) const
{
    if (shape().size() != 2 || other.shape().size() != 2)
    {
        throw std::invalid_argument("matmul: both tensors must be rank-2");
    }
    if (cols() != other.rows())
    {
        throw std::invalid_argument("matmul: lhs.cols() must equal rhs.rows()");
    }

    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl("matmul"))
    {
        throw_opencl_only_failure(
            "matmul", "OpenCL runtime unavailable or matrix dimensions are invalid");
    }

    try
    {
        return matmul_dispatch("matmul_kernel",
            "matmul",
            other,
            rows(),
            other.cols(),
            cols(),
            /*finish_after_launch=*/true);
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("matmul", e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed(const OpenCLTensorBackend& other) const
{
    if (!other.m_gpu_resident)
    {
        other.sync_gpu_if_needed();
    }
    if (shape().size() != 2 || other.shape().size() != 2 || cols() != other.cols())
    {
        // Caller error, refused by type and message before any CL call, so the
        // reader never has to disambiguate it from "no OpenCL device". Pinned
        // by OpenCLTensorBackendShapeMismatch.MatmulFamily*.
        throw std::invalid_argument(
            "matmul_transposed: lhs.cols() must equal rhs.cols() (rhs is transposed)");
    }

    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl("matmul_transposed"))
    {
        throw_opencl_only_failure(
            "matmul_transposed", "OpenCL runtime unavailable or matrix dimensions are invalid");
    }

    try
    {
        return matmul_dispatch("matmul_rhs_transposed_kernel",
            "matmul_transposed",
            other,
            rows(),
            other.rows(),
            cols(),
            /*finish_after_launch=*/false);
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("matmul_transposed", e.what());
    }
}

// matmul_lhs_transposed()'s own staging strategies. Kept separate from
// matmul_dispatch above: this kernel launches tiled (local[2] = {16, 16},
// global rounded up to the tile size), not the plain 2D NDRange matmul() and
// matmul_transposed() share, and its output shape is (k, n) -- the lhs being
// transposed means the contraction dimension (k) becomes the result's row
// count. Also note, like matmul_transposed(), the resident stage here never
// called finish_queue_if_not_batching -- preserved exactly, not added.
void OpenCLTensorBackend::enqueue_matmul_lhs_transposed_kernel(
    cl_mem a_mem, cl_mem b_mem, cl_mem c_mem, Index m, Index n, Index k)
{
    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("matmul_lhs_transposed_kernel");
    const cl_uint m_u32 = static_cast<cl_uint>(m);
    const cl_uint n_u32 = static_cast<cl_uint>(n);
    const cl_uint k_u32 = static_cast<cl_uint>(k);

    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_lhs_transposed");
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_lhs_transposed");
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem), "matmul_lhs_transposed");
    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &m_u32), "matmul_lhs_transposed");
    check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32), "matmul_lhs_transposed");
    check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &k_u32), "matmul_lhs_transposed");

    const std::size_t local[2] = {16, 16};
    const std::size_t global[2] = {round_up(k, local[0]), round_up(n, local[1])};
    check_cl_error(clEnqueueNDRangeKernel(
                       ctx.get_queue(), kernel, 2, nullptr, global, local, 0, nullptr, nullptr),
        "matmul_lhs_transposed");
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_lhs_transposed_stage_resident(
    const OpenCLTensorBackend& other, Index m, Index n, Index k) const
{
    const auto& ctx = opencl::OpenCLContext::instance();
    if (m_needs_sync_to_device)
    {
        copy_host_to_device(ctx.get_queue(),
            m_gpu_buffer->buffer,
            host_data(),
            m * k * sizeof(float),
            "matmul_lhs_transposed");
        m_needs_sync_to_device = false;
    }
    if (other.m_needs_sync_to_device)
    {
        copy_host_to_device(ctx.get_queue(),
            other.m_gpu_buffer->buffer,
            other.host_data(),
            m * n * sizeof(float),
            "matmul_lhs_transposed");
        other.m_needs_sync_to_device = false;
    }

    OpenCLTensorBackend t(k, n);
    t.set_gpu_resident(true);
    enqueue_matmul_lhs_transposed_kernel(
        m_gpu_buffer->buffer, other.m_gpu_buffer->buffer, t.m_gpu_buffer->buffer, m, n, k);
    t.m_needs_sync_to_host = true;
    t.m_needs_sync_to_device = false;
    return t;
}

std::optional<OpenCLTensorBackend> OpenCLTensorBackend::matmul_lhs_transposed_stage_pooled(
    const OpenCLTensorBackend& other, Index m, Index n, Index k) const
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr) return std::nullopt;

    const std::size_t a_bytes = m * k * sizeof(float);
    const std::size_t b_bytes = m * n * sizeof(float);
    const std::size_t c_bytes = k * n * sizeof(float);
    auto a_buf = pool->acquire(a_bytes);
    auto b_buf = pool->acquire(b_bytes);
    auto c_buf = pool->acquire(c_bytes);
    if (!a_buf || !b_buf || !c_buf) return std::nullopt;

    const auto& ctx = opencl::OpenCLContext::instance();
    copy_host_to_device(
        ctx.get_queue(), a_buf->buffer, host_data(), a_bytes, "matmul_lhs_transposed");
    copy_host_to_device(
        ctx.get_queue(), b_buf->buffer, other.host_data(), b_bytes, "matmul_lhs_transposed");
    enqueue_matmul_lhs_transposed_kernel(a_buf->buffer, b_buf->buffer, c_buf->buffer, m, n, k);
    finish_queue_if_not_batching(ctx.get_queue(), "matmul_lhs_transposed");

    OpenCLHostStorage out(k, n);
    copy_device_to_host(
        ctx.get_queue(), c_buf->buffer, out.mutable_data_ptr(), c_bytes, "matmul_lhs_transposed");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_lhs_transposed_stage_oneshot(
    const OpenCLTensorBackend& other, Index m, Index n, Index k) const
{
    const std::size_t a_bytes = m * k * sizeof(float);
    const std::size_t b_bytes = m * n * sizeof(float);
    const std::size_t c_bytes = k * n * sizeof(float);
    opencl::DeviceMemory a_dev(a_bytes);
    opencl::DeviceMemory b_dev(b_bytes);
    opencl::DeviceMemory out_dev(c_bytes);
    a_dev.copy_to_device(host_data());
    b_dev.copy_to_device(other.host_data());

    enqueue_matmul_lhs_transposed_kernel(
        a_dev.get_device_buffer(), b_dev.get_device_buffer(), out_dev.get_device_buffer(), m, n, k);
    finish_queue_if_not_batching(
        opencl::OpenCLContext::instance().get_queue(), "matmul_lhs_transposed");

    OpenCLHostStorage out(k, n);
    out_dev.copy_from_device(out.mutable_data_ptr());
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_lhs_transposed(
    const OpenCLTensorBackend& other) const
{
    if (!m_gpu_resident)
    {
        sync_gpu_if_needed();
    }
    if (!other.m_gpu_resident)
    {
        other.sync_gpu_if_needed();
    }
    if (shape().size() != 2 || other.shape().size() != 2 || rows() != other.rows())
    {
        // Caller error -- see matmul_transposed() above for why this must not
        // be throw_opencl_only_failure.
        throw std::invalid_argument(
            "matmul_lhs_transposed: lhs.rows() must equal rhs.rows() (lhs is transposed)");
    }

    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl("matmul_lhs_transposed"))
    {
        throw_opencl_only_failure(
            "matmul_lhs_transposed", "OpenCL runtime unavailable or matrix dimensions are invalid");
    }

    try
    {
        const Index m = rows();
        const Index k = cols();
        const Index n = other.cols();

        if (ensure_device_current("resident gate") && other.ensure_device_current("resident gate"))
        {
            return matmul_lhs_transposed_stage_resident(other, m, n, k);
        }
        if (auto pooled = matmul_lhs_transposed_stage_pooled(other, m, n, k)) return *pooled;
        return matmul_lhs_transposed_stage_oneshot(other, m, n, k);
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("matmul_lhs_transposed", e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed_add_col_bias(
    const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias) const
{
    // Same (a, b, bias, c, m, n, k) launch as the four activated variants
    // below, with zero extra scalars -- the "no activation" case of that
    // shared family rather than a copy of its ~230-line body.
    return matmul_transposed_bias_activated(
        "matmul_rhs_transposed_bias_kernel", "matmul_transposed_add_col_bias", other, bias, {});
}

// Shared (a, b, bias, c, m, n, k, [extra scalars...]) arg-binding + launch for
// matmul_transposed_bias_activated()'s three staging strategies below -- each path differs
// only in where a_mem/b_mem/bias_mem/c_mem live (GPU-resident buffer, pool buffer, or a
// one-off DeviceMemory), not in how the kernel is invoked.
void OpenCLTensorBackend::enqueue_matmul_bias_activated_kernel(const char* kernel_name,
    const char* what,
    cl_mem a_mem,
    cl_mem b_mem,
    cl_mem bias_mem,
    cl_mem c_mem,
    Index m,
    Index n,
    Index k,
    std::initializer_list<float> extra_scalars)
{
    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
    const cl_uint m_u32 = static_cast<cl_uint>(m);
    const cl_uint n_u32 = static_cast<cl_uint>(n);
    const cl_uint k_u32 = static_cast<cl_uint>(k);
    check_cl_error(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), (std::string(what) + " arg0").c_str());
    check_cl_error(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), (std::string(what) + " arg1").c_str());
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem),
        (std::string(what) + " arg2").c_str());
    check_cl_error(
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), (std::string(what) + " arg3").c_str());
    check_cl_error(
        clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), (std::string(what) + " arg4").c_str());
    check_cl_error(
        clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), (std::string(what) + " arg5").c_str());
    check_cl_error(
        clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), (std::string(what) + " arg6").c_str());
    cl_uint extra_index = 7;
    for (const float scalar : extra_scalars)
    {
        check_cl_error(clSetKernelArg(kernel, extra_index, sizeof(float), &scalar), what);
        ++extra_index;
    }
    const std::size_t global[2] = {m, n};
    check_cl_error(clEnqueueNDRangeKernel(
                       ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
        (std::string("clEnqueueNDRangeKernel(") + what + ")").c_str());
}

std::optional<OpenCLTensorBackend> OpenCLTensorBackend::matmul_bias_activated_stage_resident(
    const char* kernel_name,
    const char* what,
    const OpenCLTensorBackend& other,
    const OpenCLTensorBackend& bias,
    std::initializer_list<float> extra_scalars,
    Index m,
    Index n,
    Index k,
    std::size_t a_bytes,
    std::size_t b_bytes,
    std::size_t bias_bytes) const
{
    if (!(ensure_device_current("resident gate") && other.ensure_device_current("resident gate") &&
            bias.ensure_device_current("resident gate")))
    {
        return std::nullopt;
    }

    const auto& ctx = opencl::OpenCLContext::instance();
    if (m_needs_sync_to_device)
    {
        copy_host_to_device(ctx.get_queue(),
            m_gpu_buffer->buffer,
            host_data(),
            a_bytes,
            (std::string("clEnqueueWriteBuffer(") + what + ", a)").c_str());
        m_needs_sync_to_device = false;
    }
    if (other.m_needs_sync_to_device)
    {
        copy_host_to_device(ctx.get_queue(),
            other.m_gpu_buffer->buffer,
            other.host_data(),
            b_bytes,
            (std::string("clEnqueueWriteBuffer(") + what + ", b)").c_str());
        other.m_needs_sync_to_device = false;
    }
    if (bias.m_needs_sync_to_device)
    {
        copy_host_to_device(ctx.get_queue(),
            bias.m_gpu_buffer->buffer,
            bias.host_data(),
            bias_bytes,
            (std::string("clEnqueueWriteBuffer(") + what + ", bias)").c_str());
        bias.m_needs_sync_to_device = false;
    }

    OpenCLTensorBackend t(m, n);
    t.set_gpu_resident(true);
    const cl_mem a_mem = m_gpu_buffer->buffer;
    const cl_mem b_mem = other.m_gpu_buffer->buffer;
    const cl_mem bias_mem = bias.m_gpu_buffer->buffer;
    const cl_mem c_mem = t.m_gpu_buffer->buffer;
    enqueue_matmul_bias_activated_kernel(
        kernel_name, what, a_mem, b_mem, bias_mem, c_mem, m, n, k, extra_scalars);
    t.m_needs_sync_to_host = true;
    t.m_needs_sync_to_device = false;
    return t;
}

std::optional<OpenCLTensorBackend> OpenCLTensorBackend::matmul_bias_activated_stage_pooled(
    const char* kernel_name,
    const char* what,
    const OpenCLTensorBackend& other,
    const OpenCLTensorBackend& bias,
    std::initializer_list<float> extra_scalars,
    Index m,
    Index n,
    Index k,
    std::size_t a_bytes,
    std::size_t b_bytes,
    std::size_t bias_bytes,
    std::size_t c_bytes,
    OpenCLHostStorage& out) const
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (!pool) return std::nullopt;

    auto a_buf = pool->acquire(a_bytes);
    auto b_buf = pool->acquire(b_bytes);
    auto bias_buf = pool->acquire(bias_bytes);
    auto c_buf = pool->acquire(c_bytes);
    if (!(a_buf && b_buf && bias_buf && c_buf)) return std::nullopt;

    const auto& ctx = opencl::OpenCLContext::instance();
    copy_host_to_device(ctx.get_queue(),
        a_buf->buffer,
        host_data(),
        a_bytes,
        (std::string("clEnqueueWriteBuffer(") + what + ", a)").c_str());
    copy_host_to_device(ctx.get_queue(),
        b_buf->buffer,
        other.host_data(),
        b_bytes,
        (std::string("clEnqueueWriteBuffer(") + what + ", b)").c_str());
    copy_host_to_device(ctx.get_queue(),
        bias_buf->buffer,
        bias.host_data(),
        bias_bytes,
        (std::string("clEnqueueWriteBuffer(") + what + ", bias)").c_str());

    const cl_mem a_mem = a_buf->buffer;
    const cl_mem b_mem = b_buf->buffer;
    const cl_mem bias_mem = bias_buf->buffer;
    const cl_mem c_mem = c_buf->buffer;
    enqueue_matmul_bias_activated_kernel(
        kernel_name, what, a_mem, b_mem, bias_mem, c_mem, m, n, k, extra_scalars);
    finish_queue_if_not_batching(ctx.get_queue(), (std::string("clFinish(") + what + ")").c_str());
    copy_device_to_host(ctx.get_queue(),
        c_buf->buffer,
        out.mutable_data_ptr(),
        c_bytes,
        (std::string("clEnqueueReadBuffer(") + what + ", c)").c_str());
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_bias_activated_stage_oneshot(
    const char* kernel_name,
    const char* what,
    const OpenCLTensorBackend& other,
    const OpenCLTensorBackend& bias,
    std::initializer_list<float> extra_scalars,
    Index m,
    Index n,
    Index k,
    std::size_t a_bytes,
    std::size_t b_bytes,
    std::size_t bias_bytes,
    std::size_t c_bytes,
    OpenCLHostStorage& out) const
{
    const auto& ctx = opencl::OpenCLContext::instance();
    opencl::DeviceMemory a_dev(a_bytes);
    opencl::DeviceMemory b_dev(b_bytes);
    opencl::DeviceMemory bias_dev(bias_bytes);
    opencl::DeviceMemory out_dev(c_bytes);
    a_dev.copy_to_device(host_data());
    b_dev.copy_to_device(other.host_data());
    bias_dev.copy_to_device(bias.host_data());

    const cl_mem a_mem = a_dev.get_device_buffer();
    const cl_mem b_mem = b_dev.get_device_buffer();
    const cl_mem bias_mem = bias_dev.get_device_buffer();
    const cl_mem c_mem = out_dev.get_device_buffer();
    enqueue_matmul_bias_activated_kernel(
        kernel_name, what, a_mem, b_mem, bias_mem, c_mem, m, n, k, extra_scalars);
    finish_queue_if_not_batching(ctx.get_queue(), (std::string("clFinish(") + what + ")").c_str());
    out_dev.copy_from_device(out.mutable_data_ptr());

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed_bias_activated(const char* kernel_name,
    const char* what,
    const OpenCLTensorBackend& other,
    const OpenCLTensorBackend& bias,
    std::initializer_list<float> extra_scalars) const
{
    if (!m_gpu_resident) sync_gpu_if_needed();
    if (!other.m_gpu_resident) other.sync_gpu_if_needed();
    if (!bias.m_gpu_resident) bias.sync_gpu_if_needed();

    if (shape().size() != 2 || other.shape().size() != 2 || bias.shape().size() != 2 ||
        cols() != other.cols() || bias.rows() != other.rows() || bias.cols() != 1)
    {
        // Caller error -- see matmul_transposed() above for why this must not
        // be throw_opencl_only_failure. Previously this threw the ambiguous
        // runtime_error, and always under the literal name
        // "matmul_transposed_add_col_bias_leaky_relu" regardless of which of
        // the 4 callers (relu/sigmoid/tanh/leaky_relu) hit it -- fixed to name
        // the operation that was actually called.
        throw std::invalid_argument(
            std::string(what) +
            ": lhs.cols() must equal rhs.cols() (rhs is transposed), and bias must be shape "
            "(rhs.rows(), 1)");
    }

    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl(what))
    {
        throw_opencl_only_failure(what, "OpenCL runtime unavailable");
    }

    try
    {
        const Index m = rows();
        const Index k = cols();
        const Index n = other.rows();

        const std::size_t a_bytes = m * k * sizeof(float);
        const std::size_t b_bytes = n * k * sizeof(float);
        const std::size_t bias_bytes = n * sizeof(float);
        const std::size_t c_bytes = m * n * sizeof(float);
        OpenCLHostStorage out(m, n);

        if (auto resident = matmul_bias_activated_stage_resident(kernel_name,
                what,
                other,
                bias,
                extra_scalars,
                m,
                n,
                k,
                a_bytes,
                b_bytes,
                bias_bytes))
        {
            return *resident;
        }

        if (auto pooled = matmul_bias_activated_stage_pooled(kernel_name,
                what,
                other,
                bias,
                extra_scalars,
                m,
                n,
                k,
                a_bytes,
                b_bytes,
                bias_bytes,
                c_bytes,
                out))
        {
            return *pooled;
        }

        return matmul_bias_activated_stage_oneshot(kernel_name,
            what,
            other,
            bias,
            extra_scalars,
            m,
            n,
            k,
            a_bytes,
            b_bytes,
            bias_bytes,
            c_bytes,
            out);
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure(what, e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed_add_col_bias_sigmoid(
    const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias) const
{
    return matmul_transposed_bias_activated(
        "matmul_rhs_transposed_bias_sigmoid_kernel", "matmul_bias_sigmoid", other, bias, {});
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed_add_col_bias_tanh(
    const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias) const
{
    return matmul_transposed_bias_activated(
        "matmul_rhs_transposed_bias_tanh_kernel", "matmul_bias_tanh", other, bias, {});
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed_add_col_bias_relu(
    const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias) const
{
    return matmul_transposed_bias_activated(
        "matmul_rhs_transposed_bias_relu_kernel", "matmul_bias_relu", other, bias, {});
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed_add_col_bias_leaky_relu(
    const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias, float alpha) const
{
    return matmul_transposed_bias_activated(
        "matmul_rhs_transposed_bias_leaky_relu_kernel", "matmul_bias_lrelu", other, bias, {alpha});
}

} // namespace nn
