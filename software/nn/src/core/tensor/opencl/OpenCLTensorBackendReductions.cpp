/**
 * @file src/core/tensor/opencl/OpenCLTensorBackendReductions.cpp
 * @brief Reduction operations: rowwise_sum, sum_rows/sum_cols, mean_squared_error, mean, norm, sum.
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

// Reduction
// The (in, out, rows, cols) arg order rowwise_sum_kernel expects, with a
// 1D NDRange over rows (one work-item per output row) -- shared by all three
// staging strategies below.
void OpenCLTensorBackend::enqueue_rowwise_sum_kernel(
    cl_mem in_mem, cl_mem out_mem, Index num_rows, Index num_cols)
{
    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("rowwise_sum_kernel");
    const cl_uint rows_u32 = static_cast<cl_uint>(num_rows);
    const cl_uint cols_u32 = static_cast<cl_uint>(num_cols);

    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "rowwise_sum");
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "rowwise_sum");
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32), "rowwise_sum");
    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32), "rowwise_sum");

    const std::size_t global = static_cast<std::size_t>(num_rows);
    check_cl_error(clEnqueueNDRangeKernel(
                       ctx.get_queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
        "rowwise_sum");
}

// Stage 1: already resident, summed with no transfer at all.
bool OpenCLTensorBackend::rowwise_sum_stage_resident(OpenCLTensorBackend& result) const
{
    if (!ensure_device_current("resident gate")) return false;

    if (m_needs_sync_to_device)
    {
        copy_host_to_device(opencl::OpenCLContext::instance().get_queue(),
            m_gpu_buffer->buffer,
            host_data(),
            rows() * cols() * sizeof(float),
            "rowwise_sum resident");
        m_needs_sync_to_device = false;
    }

    OpenCLTensorBackend t(rows(), 1);
    t.set_gpu_resident(true);
    enqueue_rowwise_sum_kernel(m_gpu_buffer->buffer, t.m_gpu_buffer->buffer, rows(), cols());
    t.m_needs_sync_to_host = true;
    t.m_needs_sync_to_device = false;
    result = std::move(t);
    return true;
}

// Stage 2: staged through pooled buffers. Empty optional = pool
// absent/exhausted.
std::optional<OpenCLTensorBackend> OpenCLTensorBackend::rowwise_sum_stage_pooled() const
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr) return std::nullopt;

    const std::size_t input_bytes = rows() * cols() * sizeof(float);
    const std::size_t output_bytes = rows() * sizeof(float);
    auto input_buf = pool->acquire(input_bytes);
    auto out_buf = pool->acquire(output_bytes);
    if (!input_buf || !out_buf) return std::nullopt;

    const auto& ctx = opencl::OpenCLContext::instance();
    copy_host_to_device(
        ctx.get_queue(), input_buf->buffer, host_data(), input_bytes, "rowwise_sum");
    enqueue_rowwise_sum_kernel(input_buf->buffer, out_buf->buffer, rows(), cols());
    finish_queue_if_not_batching(ctx.get_queue(), "rowwise_sum");

    OpenCLHostStorage out(rows(), 1);
    copy_device_to_host(
        ctx.get_queue(), out_buf->buffer, out.mutable_data_ptr(), output_bytes, "rowwise_sum");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

// Stage 3: no pool, one-shot device buffers, fully synchronous. No
// successor, so it always produces a result or throws.
OpenCLTensorBackend OpenCLTensorBackend::rowwise_sum_stage_oneshot() const
{
    const std::size_t input_bytes = rows() * cols() * sizeof(float);
    const std::size_t output_bytes = rows() * sizeof(float);
    opencl::DeviceMemory input_dev(input_bytes);
    opencl::DeviceMemory out_dev(output_bytes);
    input_dev.copy_to_device(host_data());

    enqueue_rowwise_sum_kernel(
        input_dev.get_device_buffer(), out_dev.get_device_buffer(), rows(), cols());
    finish_queue_if_not_batching(opencl::OpenCLContext::instance().get_queue(), "rowwise_sum");

    OpenCLHostStorage out(rows(), 1);
    out_dev.copy_from_device(out.mutable_data_ptr());
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::rowwise_sum() const
{
    // Caller error, refused by type before any CL call. Was previously
    // warn_opencl_cpu_fallback_once + an unconditional throw_opencl_only_failure
    // with a message naming two unrelated causes -- the same split fixed for
    // transpose()/matmul_transposed()/compare_gt() above. Pinned by
    // OpenCLTensorBackendShapeMismatch.RowwiseSumRefusesNonRank2.
    if (shape().size() != 2)
    {
        throw std::invalid_argument("rowwise_sum: tensor must be rank-2");
    }

    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl("rowwise_sum"))
    {
        throw_opencl_only_failure("rowwise_sum", "OpenCL runtime unavailable");
    }

    try
    {
        OpenCLTensorBackend resident;
        if (rowwise_sum_stage_resident(resident)) return resident;
        if (auto pooled = rowwise_sum_stage_pooled()) return *pooled;
        return rowwise_sum_stage_oneshot();
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("rowwise_sum", e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::sum_rows() const
{
    return rowwise_sum();
}

OpenCLTensorBackend OpenCLTensorBackend::sum_cols() const
{
    sync_gpu();
    if (shape().size() != 2)
    {
        throw std::invalid_argument("sum_cols is only valid for rank-2 tensors");
    }

    OpenCLTensorBackend out(1, cols());
    out.m_backend->fill(0.0F);
    for (Index c = 0; c < cols(); ++c)
    {
        float acc = 0.0F;
        for (Index r = 0; r < rows(); ++r)
        {
            acc += m_backend->at(r, c);
        }
        out.m_backend->at(0, c) = acc;
    }
    out.m_needs_sync_to_device = true;
    out.m_needs_sync_to_host = false;
    return out;
}

// Stage 1: both operands already resident, reduced with no input transfer.
std::optional<float> OpenCLTensorBackend::mse_stage_resident(const OpenCLTensorBackend& target,
    std::size_t n,
    std::size_t local,
    std::size_t global,
    std::size_t num_groups) const
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr || !ensure_device_current("resident gate") ||
        !target.ensure_device_current("resident gate"))
        return std::nullopt;

    const std::size_t partial_bytes = num_groups * sizeof(float);
    auto partial_buf = pool->acquire(partial_bytes);
    if (!partial_buf) return std::nullopt;

    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("mse_kernel");
    const cl_mem in_mem = m_gpu_buffer->buffer;
    const cl_mem tgt_mem = target.m_gpu_buffer->buffer;
    const cl_mem out_mem = partial_buf->buffer;
    const cl_uint n_u32 = static_cast<cl_uint>(n);

    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "mse");
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &tgt_mem), "mse");
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), "mse");
    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "mse");
    check_cl_error(clEnqueueNDRangeKernel(
                       ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
        "mse");

    std::vector<float> partials(num_groups);
    copy_device_to_host(
        ctx.get_queue(), partial_buf->buffer, partials.data(), partial_bytes, "mse");
    return reduce_partials(partials) / static_cast<float>(n);
}

// Stage 2: staged through pooled buffers, both uploads chained into the
// kernel launch on events; the partials readback is a plain blocking copy
// (unlike sum()'s pooled stage, this one never went through the async +
// clWaitForEvents path -- preserved exactly as it was).
std::optional<float> OpenCLTensorBackend::mse_stage_pooled(const OpenCLTensorBackend& target,
    std::size_t n,
    std::size_t local,
    std::size_t global,
    std::size_t num_groups) const
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr) return std::nullopt;

    const std::size_t bytes = n * sizeof(float);
    const std::size_t partial_bytes = num_groups * sizeof(float);
    auto input_buf = pool->acquire(bytes);
    auto target_buf = pool->acquire(bytes);
    auto partial_buf = pool->acquire(partial_bytes);
    if (!input_buf || !target_buf || !partial_buf) return std::nullopt;

    const auto& ctx = opencl::OpenCLContext::instance();
    cl_event h2d_evt[2] = {nullptr, nullptr};
    copy_host_to_device_async(
        ctx.get_queue(), input_buf->buffer, host_data(), bytes, "mse_input", &h2d_evt[0]);
    copy_host_to_device_async(
        ctx.get_queue(), target_buf->buffer, target.host_data(), bytes, "mse_target", &h2d_evt[1]);

    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("mse_kernel");
    const cl_mem in_mem = input_buf->buffer;
    const cl_mem tgt_mem = target_buf->buffer;
    const cl_mem out_mem = partial_buf->buffer;
    const cl_uint n_u32 = static_cast<cl_uint>(n);
    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "mse");
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &tgt_mem), "mse");
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), "mse");
    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "mse");
    check_cl_error(clEnqueueNDRangeKernel(
                       ctx.get_queue(), kernel, 1, nullptr, &global, &local, 2, h2d_evt, nullptr),
        "mse");

    if (h2d_evt[0]) clReleaseEvent(h2d_evt[0]);
    if (h2d_evt[1]) clReleaseEvent(h2d_evt[1]);

    std::vector<float> partials(num_groups);
    copy_device_to_host(
        ctx.get_queue(), partial_buf->buffer, partials.data(), partial_bytes, "mse_out");
    return reduce_partials(partials) / static_cast<float>(n);
}

// Stage 3: no pool, one-shot device buffers, fully synchronous. No
// successor, so it always produces a result or throws.
float OpenCLTensorBackend::mse_stage_oneshot(const OpenCLTensorBackend& target,
    std::size_t n,
    std::size_t local,
    std::size_t global,
    std::size_t num_groups) const
{
    sync_gpu();
    target.sync_gpu();

    const std::size_t bytes = n * sizeof(float);
    const std::size_t partial_bytes = num_groups * sizeof(float);
    opencl::DeviceMemory input_dev(bytes);
    opencl::DeviceMemory target_dev(bytes);
    opencl::DeviceMemory partial_dev(partial_bytes);
    input_dev.copy_to_device(host_data());
    target_dev.copy_to_device(target.host_data());

    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("mse_kernel");
    const cl_mem in_mem = input_dev.get_device_buffer();
    const cl_mem tgt_mem = target_dev.get_device_buffer();
    const cl_mem out_mem = partial_dev.get_device_buffer();
    const cl_uint n_u32 = static_cast<cl_uint>(n);
    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "mse");
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &tgt_mem), "mse");
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), "mse");
    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "mse");
    check_cl_error(clEnqueueNDRangeKernel(
                       ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
        "mse");
    finish_queue_if_not_batching(ctx.get_queue(), "mse");

    std::vector<float> partials(num_groups);
    partial_dev.copy_from_device(partials.data());
    return reduce_partials(partials) / static_cast<float>(n);
}

float OpenCLTensorBackend::mean_squared_error(const OpenCLTensorBackend& target) const
{
    if (shape() != target.shape())
    {
        throw std::invalid_argument("mean_squared_error requires equal shapes");
    }

    const auto n = size();
    if (n == 0) return 0.0F;

    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl("mse"))
    {
        throw_opencl_only_failure("mse", "OpenCL runtime unavailable");
    }

    try
    {
        const std::size_t local = 256;
        const std::size_t global = round_up(n, local);
        const std::size_t num_groups = global / local;

        if (auto resident = mse_stage_resident(target, n, local, global, num_groups))
            return *resident;
        if (auto pooled = mse_stage_pooled(target, n, local, global, num_groups)) return *pooled;
        return mse_stage_oneshot(target, n, local, global, num_groups);
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("mse", e.what());
    }
}

float OpenCLTensorBackend::mean() const
{
    const auto n = size();
    if (n == 0) return 0.0F;
    return sum() / static_cast<float>(n);
}

float OpenCLTensorBackend::norm() const
{
    sync_gpu();
    float sq = 0.0F;
    for (Index i = 0; i < size(); ++i)
    {
        const float v = m_backend->at(i);
        sq += v * v;
    }
    return std::sqrt(sq);
}

// sum_kernel writes one partial per work-group; the actual sum is the host
// reduction of those partials. Shared by all three staging strategies below,
// so the reduction itself cannot drift between them.
float OpenCLTensorBackend::reduce_partials(const std::vector<float>& partials)
{
    float result = 0.0F;
    for (const float partial : partials) result += partial;
    return result;
}

// Stage 1: already resident, reduced with no input transfer -- only the
// (small, num_groups-sized) partials come back to host, since the final
// reduction happens there regardless.
std::optional<float> OpenCLTensorBackend::sum_stage_resident(
    std::size_t n, std::size_t local, std::size_t global, std::size_t num_groups) const
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr || !ensure_device_current("resident gate")) return std::nullopt;

    const std::size_t partial_bytes = num_groups * sizeof(float);
    auto partial_buf = pool->acquire(partial_bytes);
    if (!partial_buf) return std::nullopt;

    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("sum_kernel");
    const cl_mem in_mem = m_gpu_buffer->buffer;
    const cl_mem out_mem = partial_buf->buffer;
    const cl_uint n_u32 = static_cast<cl_uint>(n);

    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "sum");
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "sum");
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "sum");
    check_cl_error(clEnqueueNDRangeKernel(
                       ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
        "sum");

    std::vector<float> partials(num_groups);
    copy_device_to_host(
        ctx.get_queue(), partial_buf->buffer, partials.data(), partial_bytes, "sum");
    return reduce_partials(partials);
}

// Stage 2: staged through pooled buffers, upload and kernel chained on
// events. The partials readback still blocks (via clWaitForEvents) rather
// than going through record_pending_gpu_op: sum() returns a float, and a
// float has no "resident, synced lazily" state to defer into.
std::optional<float> OpenCLTensorBackend::sum_stage_pooled(
    std::size_t n, std::size_t local, std::size_t global, std::size_t num_groups) const
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr) return std::nullopt;

    const std::size_t bytes = n * sizeof(float);
    const std::size_t partial_bytes = num_groups * sizeof(float);
    auto input_buf = pool->acquire(bytes);
    auto partial_buf = pool->acquire(partial_bytes);
    if (!input_buf || !partial_buf) return std::nullopt;

    const auto& ctx = opencl::OpenCLContext::instance();
    cl_event h2d_evt = nullptr;
    copy_host_to_device_async(
        ctx.get_queue(), input_buf->buffer, host_data(), bytes, "sum", &h2d_evt);

    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("sum_kernel");
    const cl_mem in_mem = input_buf->buffer;
    const cl_mem out_mem = partial_buf->buffer;
    const cl_uint n_u32 = static_cast<cl_uint>(n);
    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "sum");
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "sum");
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "sum");

    cl_event kernel_evt = nullptr;
    const bool has_wait = h2d_evt != nullptr;
    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                       kernel,
                       1,
                       nullptr,
                       &global,
                       &local,
                       has_wait ? 1U : 0U,
                       has_wait ? &h2d_evt : nullptr,
                       &kernel_evt),
        "sum");
    if (h2d_evt) clReleaseEvent(h2d_evt);

    std::vector<float> partials(num_groups);
    cl_event d2h_evt = nullptr;
    copy_device_to_host_async(
        ctx.get_queue(), partial_buf->buffer, partials.data(), partial_bytes, "sum", &d2h_evt);
    if (kernel_evt) clReleaseEvent(kernel_evt);
    if (d2h_evt) clWaitForEvents(1, &d2h_evt);

    return reduce_partials(partials);
}

// Stage 3: no pool, one-shot device buffers, fully synchronous. No
// successor, so it always produces a result or throws.
float OpenCLTensorBackend::sum_stage_oneshot(
    std::size_t n, std::size_t local, std::size_t global, std::size_t num_groups) const
{
    sync_gpu();

    const std::size_t bytes = n * sizeof(float);
    const std::size_t partial_bytes = num_groups * sizeof(float);
    opencl::DeviceMemory input_dev(bytes);
    opencl::DeviceMemory partial_dev(partial_bytes);
    input_dev.copy_to_device(host_data());

    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("sum_kernel");
    const cl_mem in_mem = input_dev.get_device_buffer();
    const cl_mem out_mem = partial_dev.get_device_buffer();
    const cl_uint n_u32 = static_cast<cl_uint>(n);
    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "sum");
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "sum");
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "sum");
    check_cl_error(clEnqueueNDRangeKernel(
                       ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
        "sum");
    finish_queue_if_not_batching(ctx.get_queue(), "sum");

    std::vector<float> partials(num_groups);
    partial_dev.copy_from_device(partials.data());
    return reduce_partials(partials);
}

float OpenCLTensorBackend::sum() const
{
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl("sum"))
    {
        throw_opencl_only_failure("sum", "OpenCL runtime unavailable");
    }

    try
    {
        const auto n = size();
        if (n == 0) return 0.0F;

        const std::size_t local = 256;
        const std::size_t global = round_up(n, local);
        const std::size_t num_groups = global / local;

        if (auto resident = sum_stage_resident(n, local, global, num_groups)) return *resident;
        if (auto pooled = sum_stage_pooled(n, local, global, num_groups)) return *pooled;
        return sum_stage_oneshot(n, local, global, num_groups);
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("sum", e.what());
    }
}

} // namespace nn
