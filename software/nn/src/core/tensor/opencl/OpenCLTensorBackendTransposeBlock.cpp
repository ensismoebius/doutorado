/**
 * @file src/core/tensor/opencl/OpenCLTensorBackendTransposeBlock.cpp
 * @brief transpose() and block() device-side view operations.
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

// The (in, out, rows, cols) arg order transpose_kernel expects, and the 2D
// NDRange launch it uses -- shared by both staging strategies below.
void OpenCLTensorBackend::enqueue_transpose_kernel(
    cl_mem in_mem, cl_mem out_mem, Index in_rows, Index in_cols)
{
    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("transpose_kernel");
    const cl_uint rows_u32 = static_cast<cl_uint>(in_rows);
    const cl_uint cols_u32 = static_cast<cl_uint>(in_cols);

    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "transpose");
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "transpose");
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32), "transpose");
    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32), "transpose");

    const std::size_t global[2] = {in_rows, in_cols};
    check_cl_error(clEnqueueNDRangeKernel(
                       ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
        "transpose");
    finish_queue_if_not_batching(ctx.get_queue(), "transpose");
}

// Preferred staging: pooled buffers. Empty optional = pool absent/exhausted.
std::optional<OpenCLTensorBackend> OpenCLTensorBackend::transpose_stage_pooled(
    Index in_rows, Index in_cols, std::size_t bytes) const
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr) return std::nullopt;

    auto in_buf = pool->acquire(bytes);
    auto out_buf = pool->acquire(bytes);
    if (!in_buf || !out_buf) return std::nullopt;

    const auto& ctx = opencl::OpenCLContext::instance();
    copy_host_to_device(ctx.get_queue(), in_buf->buffer, host_data(), bytes, "transpose");
    enqueue_transpose_kernel(in_buf->buffer, out_buf->buffer, in_rows, in_cols);

    OpenCLHostStorage out(in_cols, in_rows);
    // GPU-resident mode: keep result on GPU
    if (ensure_device_current("resident gate"))
    {
        OpenCLTensorBackend t;
        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
        t.m_has_gpu_memory = true;
        t.m_gpu_buffer = std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
        t.set_gpu_resident(true);
        t.m_needs_sync_to_host = true;
        t.m_needs_sync_to_device = false;
        return t;
    }

    copy_device_to_host(
        ctx.get_queue(), out_buf->buffer, out.mutable_data_ptr(), bytes, "transpose");
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

// No pool (or exhausted): one-shot device buffers, fully synchronous.
OpenCLTensorBackend OpenCLTensorBackend::transpose_stage_oneshot(
    Index in_rows, Index in_cols, std::size_t bytes) const
{
    opencl::DeviceMemory in_dev(bytes);
    opencl::DeviceMemory out_dev(bytes);
    in_dev.copy_to_device(host_data());

    enqueue_transpose_kernel(
        in_dev.get_device_buffer(), out_dev.get_device_buffer(), in_rows, in_cols);

    OpenCLHostStorage out(in_cols, in_rows);
    out_dev.copy_from_device(out.mutable_data_ptr());
    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::transpose() const
{
    if (shape().size() != 2)
    {
        throw std::invalid_argument("transpose: tensor must be rank-2");
    }
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl("transpose"))
    {
        throw_opencl_only_failure(
            "transpose", "OpenCL runtime unavailable or tensor rank is invalid");
    }

    try
    {
        const Index in_rows = rows();
        const Index in_cols = cols();
        const std::size_t bytes = in_rows * in_cols * sizeof(float);

        if (auto pooled = transpose_stage_pooled(in_rows, in_cols, bytes)) return *pooled;
        return transpose_stage_oneshot(in_rows, in_cols, bytes);
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("transpose", e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::block(
    Index row, Index col, Index rows_n, Index cols_n) const
{
    if (shape().size() != 2)
    {
        throw std::invalid_argument("block is only valid for rank-2 tensors");
    }
    if (row + rows_n > rows() || col + cols_n > cols())
    {
        throw std::out_of_range("block exceeds tensor bounds");
    }

    OpenCLTensorBackend out(rows_n, cols_n);
    if (launch_strided_copy(
            *this, {row + col * rows(), 1, rows()}, out, {0, 1, rows_n}, rows_n, cols_n, "block"))
    {
        return out;
    }

    sync_gpu();
    for (Index i = 0; i < rows_n; ++i)
    {
        for (Index j = 0; j < cols_n; ++j)
        {
            out.m_backend->at(i, j) = m_backend->at(row + i, col + j);
        }
    }
    out.m_needs_sync_to_device = true;
    out.m_needs_sync_to_host = false;
    return out;
}

} // namespace nn
