/**
 * @file src/core/tensor/opencl/OpenCLTensorBackendInplace.cpp
 * @brief In-place binary/scalar/unary/broadcast operations (add_inplace, multiply_scalar_inplace,
 * add_row_broadcast_inplace, and friends).
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

// In-place operations
namespace
{

// clEnqueueNDRangeKernel wants (count, pointer) for the events to wait on, and
// a caller with two optional events has four cases. Collecting the non-null
// ones into an array turns those four near-identical branches into one call.
struct WaitList
{
    cl_event storage[2] = {nullptr, nullptr};
    cl_uint count = 0;

    void add(cl_event e)
    {
        if (e != nullptr) storage[count++] = e;
    }
    [[nodiscard]] const cl_event* events() const
    {
        return count > 0 ? storage : nullptr;
    }
};

} // namespace

// Stage 1 of binary_inplace: both operands already live on the device, so the
// kernel runs with no transfer at all. Returns false when that is not the case.
bool OpenCLTensorBackend::inplace_stage_resident(
    const char* kernel_name, const char* what, const OpenCLTensorBackend& other, std::size_t n)
{
    if (!ensure_device_current("resident gate") || !other.ensure_device_current("resident gate"))
    {
        return false;
    }

    const auto& ctx = opencl::OpenCLContext::instance();
    const std::size_t bytes = n * sizeof(float);

    // A resident buffer can still be behind its host mirror if the caller wrote
    // through at(); push those bytes up before reading them on the device.
    if (m_needs_sync_to_device)
    {
        copy_host_to_device(ctx.get_queue(),
            m_gpu_buffer->buffer,
            host_data(),
            bytes,
            (std::string(what) + " resident lhs").c_str());
        m_needs_sync_to_device = false;
    }
    if (other.m_needs_sync_to_device)
    {
        copy_host_to_device(ctx.get_queue(),
            other.m_gpu_buffer->buffer,
            other.host_data(),
            bytes,
            (std::string(what) + " resident rhs").c_str());
        other.m_needs_sync_to_device = false;
    }

    enqueue_inplace_binary_kernel(kernel_name,
        what,
        m_gpu_buffer->buffer,
        other.m_gpu_buffer->buffer,
        n,
        WaitList{}.events(),
        0,
        nullptr);
    finish_queue_if_not_batching(ctx.get_queue(), what);

    m_needs_sync_to_host = true;
    m_needs_sync_to_device = false;
    return true;
}

// Stage 2: neither operand is resident, but the pool can lend staging buffers.
// Upload, kernel and download are chained on events, so the host never blocks
// between them. Returns false when the pool is absent or exhausted.
bool OpenCLTensorBackend::inplace_stage_pooled(
    const char* kernel_name, const char* what, const OpenCLTensorBackend& other, std::size_t n)
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr) return false;

    const std::size_t bytes = n * sizeof(float);
    auto a_buf = pool->acquire(bytes);
    auto b_buf = pool->acquire(bytes);
    if (!a_buf || !b_buf) return false;

    const auto& ctx = opencl::OpenCLContext::instance();
    cl_event a_evt = nullptr;
    cl_event b_evt = nullptr;
    copy_host_to_device_async(ctx.get_queue(), a_buf->buffer, host_data(), bytes, what, &a_evt);
    copy_host_to_device_async(
        ctx.get_queue(), b_buf->buffer, other.host_data(), bytes, what, &b_evt);

    WaitList waits;
    waits.add(a_evt);
    waits.add(b_evt);

    cl_event kernel_evt = nullptr;
    enqueue_inplace_binary_kernel(kernel_name,
        what,
        a_buf->buffer,
        b_buf->buffer,
        n,
        waits.events(),
        waits.count,
        &kernel_evt);

    if (a_evt) clReleaseEvent(a_evt);
    if (b_evt) clReleaseEvent(b_evt);

    cl_event d2h_evt = nullptr;
    copy_device_to_host_async(
        ctx.get_queue(), a_buf->buffer, mutable_host_data(), bytes, what, &d2h_evt);
    mark_host_dirty();

    if (kernel_evt) clReleaseEvent(kernel_evt);
    record_pending_gpu_op(d2h_evt);
    return true;
}

// Stage 3: no residency, no pool. One-shot buffers, fully synchronous. Always
// succeeds or throws — there is no fourth strategy behind it.
void OpenCLTensorBackend::inplace_stage_oneshot(
    const char* kernel_name, const char* what, const OpenCLTensorBackend& other, std::size_t n)
{
    const auto& ctx = opencl::OpenCLContext::instance();
    const std::size_t bytes = n * sizeof(float);

    opencl::DeviceMemory a_dev(bytes);
    opencl::DeviceMemory b_dev(bytes);
    a_dev.copy_to_device(host_data());
    b_dev.copy_to_device(other.host_data());

    enqueue_inplace_binary_kernel(kernel_name,
        what,
        a_dev.get_device_buffer(),
        b_dev.get_device_buffer(),
        n,
        nullptr,
        0,
        nullptr);
    finish_queue_if_not_batching(ctx.get_queue(), what);

    a_dev.copy_from_device(mutable_host_data());
    mark_host_dirty();
}

// The (A, B, n) arg order every in-place binary kernel in KernelManager.cpp
// expects, in one place, so the three staging strategies cannot disagree.
void OpenCLTensorBackend::enqueue_inplace_binary_kernel(const char* kernel_name,
    const char* what,
    cl_mem a_mem,
    cl_mem b_mem,
    std::size_t n,
    const cl_event* wait_events,
    cl_uint wait_count,
    cl_event* out_event)
{
    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
    const cl_uint n_u32 = static_cast<cl_uint>(n);

    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), what);
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), what);
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), what);

    const std::size_t local = 256;
    std::size_t global = round_up(n, local);
    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                       kernel,
                       1,
                       nullptr,
                       &global,
                       &local,
                       wait_count,
                       wait_events,
                       out_event),
        what);
}

void OpenCLTensorBackend::binary_inplace(
    const char* kernel_name, const char* what, const OpenCLTensorBackend& other)
{
    // Device-resident fast path (see binary_elementwise() for the rationale).
    if (shape() == other.shape() && launch_inplace_binary_resident(kernel_name, *this, other, what))
        return;

    sync_gpu();
    other.sync_gpu();

    // Caller error, refused by type and message before any CL call. Pinned by
    // OpenCLTensorBackendShapeMismatch.InPlaceOpsRefuseMismatchedShapes.
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
        if (n == 0) return;

        // Three staging strategies, cheapest transfer first.
        if (inplace_stage_resident(kernel_name, what, other, n)) return;
        if (inplace_stage_pooled(kernel_name, what, other, n)) return;
        inplace_stage_oneshot(kernel_name, what, other, n);
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

void OpenCLTensorBackend::add_inplace(const OpenCLTensorBackend& other)
{
    binary_inplace("add_inplace_kernel", "add_inplace", other);
}

void OpenCLTensorBackend::subtract_inplace(const OpenCLTensorBackend& other)
{
    binary_inplace("subtract_inplace_kernel", "subtract_inplace", other);
}

void OpenCLTensorBackend::multiply_inplace(const OpenCLTensorBackend& other)
{
    binary_inplace("multiply_inplace_kernel", "multiply_inplace", other);
}

void OpenCLTensorBackend::divide_inplace(const OpenCLTensorBackend& other)
{
    binary_inplace("divide_inplace_kernel", "divide_inplace", other);
}

// Every in-place kernel here takes (data, scalars..., n): scalar_inplace's
// `val` is the only scalar, unary_inplace has none. Computing the size
// argument's index as 1 + scalars.size() is what lets those two otherwise
// identical bodies collapse into this one, on the same pattern as
// unary_elementwise/unary_scalars_elementwise above.
void OpenCLTensorBackend::enqueue_inplace_kernel(const char* kernel_name,
    const char* what,
    cl_mem data_mem,
    std::initializer_list<float> scalars,
    std::size_t n,
    const cl_event* wait_event,
    cl_event* out_event)
{
    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
    const cl_uint n_u32 = static_cast<cl_uint>(n);

    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), what);
    cl_uint arg_index = 1;
    for (const float scalar : scalars)
    {
        check_cl_error(clSetKernelArg(kernel, arg_index, sizeof(float), &scalar), what);
        ++arg_index;
    }
    check_cl_error(clSetKernelArg(kernel, arg_index, sizeof(cl_uint), &n_u32), what);

    const std::size_t local = 256;
    std::size_t global = round_up(n, local);
    const bool has_wait = wait_event != nullptr && *wait_event != nullptr;
    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                       kernel,
                       1,
                       nullptr,
                       &global,
                       &local,
                       has_wait ? 1U : 0U,
                       has_wait ? wait_event : nullptr,
                       out_event),
        what);
}

// Stage 1: already resident, mutated in place with no transfer at all.
bool OpenCLTensorBackend::inplace_elementwise_stage_resident(
    const char* kernel_name, const char* what, std::initializer_list<float> scalars, std::size_t n)
{
    if (!ensure_device_current("resident gate")) return false;

    enqueue_inplace_kernel(kernel_name, what, m_gpu_buffer->buffer, scalars, n, nullptr, nullptr);
    finish_queue_if_not_batching(opencl::OpenCLContext::instance().get_queue(), what);

    m_needs_sync_to_host = true;
    m_needs_sync_to_device = false;
    return true;
}

// Stage 2: staged through a pooled buffer, upload/kernel/download chained on
// events.
bool OpenCLTensorBackend::inplace_elementwise_stage_pooled(
    const char* kernel_name, const char* what, std::initializer_list<float> scalars, std::size_t n)
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr) return false;

    const std::size_t bytes = n * sizeof(float);
    auto data_buf = pool->acquire(bytes);
    if (!data_buf) return false;

    const auto& ctx = opencl::OpenCLContext::instance();
    cl_event h2d_evt = nullptr;
    copy_host_to_device_async(
        ctx.get_queue(), data_buf->buffer, host_data(), bytes, what, &h2d_evt);

    cl_event kernel_evt = nullptr;
    enqueue_inplace_kernel(kernel_name, what, data_buf->buffer, scalars, n, &h2d_evt, &kernel_evt);

    if (h2d_evt) clReleaseEvent(h2d_evt);

    cl_event d2h_evt = nullptr;
    copy_device_to_host_async(
        ctx.get_queue(), data_buf->buffer, mutable_host_data(), bytes, what, &d2h_evt);
    mark_host_dirty();

    if (kernel_evt) clReleaseEvent(kernel_evt);
    record_pending_gpu_op(d2h_evt);
    return true;
}

// Stage 3: no pool, one-shot device buffer, fully synchronous.
void OpenCLTensorBackend::inplace_elementwise_stage_oneshot(
    const char* kernel_name, const char* what, std::initializer_list<float> scalars, std::size_t n)
{
    const std::size_t bytes = n * sizeof(float);
    opencl::DeviceMemory data_dev(bytes);
    data_dev.copy_to_device(host_data());

    enqueue_inplace_kernel(
        kernel_name, what, data_dev.get_device_buffer(), scalars, n, nullptr, nullptr);
    finish_queue_if_not_batching(opencl::OpenCLContext::instance().get_queue(), what);

    data_dev.copy_from_device(mutable_host_data());
    mark_host_dirty();
}

// The three stages in cheapest-transfer-first order, shared by unary_inplace
// (no scalars) and scalar_inplace (one scalar). `n == 0` is a silent no-op
// here because it always was: a scalar op on an empty tensor changes nothing.
void OpenCLTensorBackend::run_inplace_elementwise_stages(
    const char* kernel_name, const char* what, std::initializer_list<float> scalars)
{
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl(what))
    {
        throw_opencl_only_failure(what, "OpenCL runtime unavailable");
    }

    try
    {
        const auto n = size();
        if (n == 0) return;

        if (inplace_elementwise_stage_resident(kernel_name, what, scalars, n)) return;
        if (inplace_elementwise_stage_pooled(kernel_name, what, scalars, n)) return;
        inplace_elementwise_stage_oneshot(kernel_name, what, scalars, n);
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure(what, e.what());
    }
}

void OpenCLTensorBackend::scalar_inplace(const char* kernel_name, const char* what, float val)
{
    // Device-resident fast path (see add() for the pattern rationale).
    if (launch_inplace_scalar_resident(kernel_name, *this, val, what)) return;

    sync_gpu();
    run_inplace_elementwise_stages(kernel_name, what, {val});
}

void OpenCLTensorBackend::add_scalar_inplace(float val)
{
    scalar_inplace("add_scalar_inplace_kernel", "add_scalar_inplace", val);
}

void OpenCLTensorBackend::multiply_scalar_inplace(float val)
{
    scalar_inplace("multiply_scalar_inplace_kernel", "multiply_scalar_inplace", val);
}

void OpenCLTensorBackend::divide_scalar_inplace(float val)
{
    scalar_inplace("divide_scalar_inplace_kernel", "divide_scalar_inplace", val);
}

void OpenCLTensorBackend::unary_inplace(const char* kernel_name, const char* what)
{
    // No resident fast-path call here and no leading sync_gpu(): unary_inplace
    // never had either, unlike scalar_inplace. Preserved exactly rather than
    // unified, since neither is understood well enough here to be sure adding
    // or removing either is behavior-neutral.
    run_inplace_elementwise_stages(kernel_name, what, {});
}

void OpenCLTensorBackend::sqrt_inplace()
{
    unary_inplace("sqrt_inplace_kernel", "sqrt_inplace");
}

void OpenCLTensorBackend::fill(float value)
{
    run_inplace_elementwise_stages("fill_kernel", "fill", {value});
}

void OpenCLTensorBackend::set_zero()
{
    fill(0.0F);
}

void OpenCLTensorBackend::set_ones()
{
    fill(1.0F);
}

// add_row_broadcast_inplace()'s (1,M) row and add_col_vector_to_rows_inplace()'s
// (M,1) col_vector hold the same M values in the same flat layout, and both
// launch the identical add_col_vector_to_rows_kernel(data, vec, rows, cols)
// with no shape-dependent branching at all -- confirmed by diffing the two
// bodies (they differed only in identifier names and which of {row.rows()==1,
// col_vector.cols()==1} each validates). Shared here; each public entry point
// still does its own shape check, in its own required orientation, before
// calling in.
//
// add_row_broadcast_inplace previously fell back to a CPU loop when OpenCL
// was unavailable -- the only such fallback among this pair, and a violation
// of the project's no-fallback policy once actually looked at side by side
// with add_col_vector_to_rows_inplace's sibling behaviour (which already
// raised). Removed: both now raise identically.
void OpenCLTensorBackend::enqueue_broadcast_vector_kernel(cl_mem data_mem,
    cl_mem vec_mem,
    Index num_rows,
    Index num_cols,
    cl_uint num_wait_events,
    const cl_event* wait_events,
    cl_event* out_event)
{
    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel =
        opencl::KernelManager::instance().get_kernel("add_col_vector_to_rows_kernel");
    const cl_uint rows_u32 = static_cast<cl_uint>(num_rows);
    const cl_uint cols_u32 = static_cast<cl_uint>(num_cols);

    check_cl_error(
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "add_col_vector_to_rows_kernel");
    check_cl_error(
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_mem), "add_col_vector_to_rows_kernel");
    check_cl_error(
        clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32), "add_col_vector_to_rows_kernel");
    check_cl_error(
        clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32), "add_col_vector_to_rows_kernel");

    const std::size_t local = 256;
    const std::size_t n = num_rows * num_cols;
    std::size_t global = round_up(n, local);
    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                       kernel,
                       1,
                       nullptr,
                       &global,
                       &local,
                       num_wait_events,
                       num_wait_events ? wait_events : nullptr,
                       out_event),
        "add_col_vector_to_rows_kernel");
}

// Stage 1: both operands already resident, mutated in place with no
// transfer at all.
bool OpenCLTensorBackend::broadcast_vector_stage_resident(
    const OpenCLTensorBackend& vec, const char* what)
{
    if (!ensure_device_current("resident gate") || !vec.ensure_device_current("resident gate"))
    {
        return false;
    }

    const auto& ctx = opencl::OpenCLContext::instance();
    const auto num_rows = rows();
    const auto num_cols = cols();
    const auto n = size();
    if (n == 0) return true;
    const std::size_t bytes = n * sizeof(float);
    const std::size_t vec_bytes = num_cols * sizeof(float);

    if (m_needs_sync_to_device)
    {
        copy_host_to_device(ctx.get_queue(), m_gpu_buffer->buffer, host_data(), bytes, what);
        m_needs_sync_to_device = false;
    }
    if (vec.m_needs_sync_to_device)
    {
        copy_host_to_device(
            ctx.get_queue(), vec.m_gpu_buffer->buffer, vec.host_data(), vec_bytes, what);
        vec.m_needs_sync_to_device = false;
    }

    enqueue_broadcast_vector_kernel(
        m_gpu_buffer->buffer, vec.m_gpu_buffer->buffer, num_rows, num_cols, 0, nullptr, nullptr);
    m_needs_sync_to_host = true;
    m_needs_sync_to_device = false;
    return true;
}

// Stage 2: staged through pooled buffers, upload/kernel/download chained on
// events rather than blocking between each step.
bool OpenCLTensorBackend::broadcast_vector_stage_pooled(
    const OpenCLTensorBackend& vec, const char* what)
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr) return false;

    const auto num_rows = rows();
    const auto num_cols = cols();
    const auto n = size();
    if (n == 0) return true;
    const std::size_t bytes = n * sizeof(float);
    const std::size_t vec_bytes = num_cols * sizeof(float);

    auto data_buf = pool->acquire(bytes);
    auto vec_buf = pool->acquire(vec_bytes);
    if (!data_buf || !vec_buf) return false;

    const auto& ctx = opencl::OpenCLContext::instance();
    cl_event data_evt = nullptr;
    cl_event vec_evt = nullptr;
    copy_host_to_device_async(
        ctx.get_queue(), data_buf->buffer, host_data(), bytes, what, &data_evt);
    copy_host_to_device_async(
        ctx.get_queue(), vec_buf->buffer, vec.host_data(), vec_bytes, what, &vec_evt);

    cl_event wait_events[2];
    cl_uint num_wait = 0;
    if (data_evt) wait_events[num_wait++] = data_evt;
    if (vec_evt) wait_events[num_wait++] = vec_evt;

    cl_event kernel_evt = nullptr;
    enqueue_broadcast_vector_kernel(
        data_buf->buffer, vec_buf->buffer, num_rows, num_cols, num_wait, wait_events, &kernel_evt);

    if (data_evt) clReleaseEvent(data_evt);
    if (vec_evt) clReleaseEvent(vec_evt);

    cl_event d2h_evt = nullptr;
    copy_device_to_host_async(
        ctx.get_queue(), data_buf->buffer, mutable_host_data(), bytes, what, &d2h_evt);
    mark_host_dirty();

    if (kernel_evt) clReleaseEvent(kernel_evt);
    record_pending_gpu_op(d2h_evt);
    return true;
}

// Stage 3: no pool, one-shot device buffers, fully synchronous. No
// successor, so it either finishes the mutation or throws.
void OpenCLTensorBackend::broadcast_vector_stage_oneshot(
    const OpenCLTensorBackend& vec, const char* what)
{
    const auto num_rows = rows();
    const auto num_cols = cols();
    const auto n = size();
    if (n == 0) return;
    const std::size_t bytes = n * sizeof(float);
    const std::size_t vec_bytes = num_cols * sizeof(float);

    opencl::DeviceMemory data_dev(bytes);
    opencl::DeviceMemory vec_dev(vec_bytes);
    data_dev.copy_to_device(host_data());
    vec_dev.copy_to_device(vec.host_data());

    enqueue_broadcast_vector_kernel(data_dev.get_device_buffer(),
        vec_dev.get_device_buffer(),
        num_rows,
        num_cols,
        0,
        nullptr,
        nullptr);
    finish_queue_if_not_batching(opencl::OpenCLContext::instance().get_queue(), what);

    data_dev.copy_from_device(mutable_host_data());
    mark_host_dirty();
}

// The three stages in cheapest-transfer-first order, shared by
// add_row_broadcast_inplace() and add_col_vector_to_rows_inplace(). Callers
// have already validated shape and raise std::invalid_argument themselves;
// this only ever raises for a genuine device-availability/execution failure.
void OpenCLTensorBackend::run_broadcast_vector_stages(
    const OpenCLTensorBackend& vec, const char* what)
{
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl(what))
    {
        throw_opencl_only_failure(what, "OpenCL runtime unavailable");
    }

    try
    {
        if (broadcast_vector_stage_resident(vec, what)) return;
        if (broadcast_vector_stage_pooled(vec, what)) return;
        broadcast_vector_stage_oneshot(vec, what);
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure(what, e.what());
    }
}

void OpenCLTensorBackend::add_row_broadcast_inplace(const OpenCLTensorBackend& row)
{
    if (shape().size() != 2 || row.shape().size() != 2 || row.rows() != 1 || row.cols() != cols())
    {
        throw std::invalid_argument("add_row_broadcast_inplace requires lhs=(N,M) and row=(1,M)");
    }
    run_broadcast_vector_stages(row, "add_row_broadcast_inplace");
}

OpenCLTensorBackend OpenCLTensorBackend::add_row_broadcast(const OpenCLTensorBackend& row) const
{
    OpenCLTensorBackend out(*this);
    out.add_row_broadcast_inplace(row);
    return out;
}

void OpenCLTensorBackend::square_inplace()
{
    unary_inplace("square_inplace_kernel", "square_inplace");
}

void OpenCLTensorBackend::add_col_vector_to_rows_inplace(const OpenCLTensorBackend& col_vector)
{
    // Preserved from before the merge: eager sync before validation, unlike
    // add_row_broadcast_inplace's sibling above, which has no such call.
    // Genuinely asymmetric, not harmonized away.
    if (!col_vector.m_gpu_resident)
    {
        col_vector.sync_gpu();
    }
    if (shape().size() != 2 || col_vector.shape().size() != 2 || cols() != col_vector.rows() ||
        col_vector.cols() != 1)
    {
        throw std::invalid_argument(
            "add_col_vector_to_rows_inplace requires lhs=(N,M) and col_vector=(M,1)");
    }
    run_broadcast_vector_stages(col_vector, "add_col_vector_to_rows_inplace");
}

} // namespace nn
