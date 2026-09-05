/**
 * @file src/core/tensor/opencl/OpenCLTensorBackend.cpp
 * @brief OpenCL-only tensor backend implementation: construction, shape,
 *        element access, and device-side view/slice operations.
 *
 * Tensor metadata and host synchronization staging are managed locally,
 * while math operations execute through OpenCL kernels only. In-place ops,
 * elementwise ops, reductions, unary/LIF ops, matmul/transpose/block,
 * comparisons, and the runtime/buffer-pool machinery live in the sibling
 * OpenCLTensorBackend*.cpp files in this directory; OpenCLTensorBackendDetail.hpp
 * holds the host storage class and OpenCL helpers shared by all of them.
 */

#include "tensor/opencl/OpenCLTensorBackend.hpp"

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

namespace nn
{

OpenCLTensorBackend::OpenCLTensorBackend() : m_backend(std::make_unique<OpenCLHostStorage>()) {}

OpenCLTensorBackend::OpenCLTensorBackend(Index rows, Index cols)
    : m_backend(std::make_unique<OpenCLHostStorage>(rows, cols))
{
    try_allocate_gpu_buffer(rows * cols);
    m_needs_sync_to_device = true;
}

OpenCLTensorBackend::OpenCLTensorBackend(Index d1, Index d2, Index d3)
    : m_backend(std::make_unique<OpenCLHostStorage>(d1, d2, d3))
{
    try_allocate_gpu_buffer(d1 * d2 * d3);
    m_needs_sync_to_device = true;
}

OpenCLTensorBackend::OpenCLTensorBackend(Index d1, Index d2, Index d3, Index d4)
    : m_backend(std::make_unique<OpenCLHostStorage>(d1, d2, d3, d4))
{
    try_allocate_gpu_buffer(d1 * d2 * d3 * d4);
    m_needs_sync_to_device = true;
}

OpenCLTensorBackend::OpenCLTensorBackend(const std::vector<Index>& shape)
    : m_backend(std::make_unique<OpenCLHostStorage>(shape))
{
    const Index total =
        std::accumulate(shape.begin(), shape.end(), static_cast<Index>(1), std::multiplies<>{});
    try_allocate_gpu_buffer(total);
    m_needs_sync_to_device = true;
}

OpenCLTensorBackend::OpenCLTensorBackend(const OpenCLTensorBackend& other)
{
    other.sync_gpu_if_needed();
    if (other.m_backend)
    {
        m_backend = std::make_unique<OpenCLHostStorage>(*other.m_backend);
    }
    if (other.m_grad_backend)
    {
        m_grad_backend = std::make_unique<OpenCLTensorBackend>(*other.m_grad_backend);
    }

    m_gpu_resident = other.m_gpu_resident;
    m_pipeline_mode = other.m_pipeline_mode;
    m_needs_sync_to_host = false;
    m_needs_sync_to_device = true;

    if (other.m_has_gpu_memory && m_backend)
    {
        try_allocate_gpu_buffer(size());
        if (m_has_gpu_memory && m_gpu_buffer)
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = size() * sizeof(float);
            copy_host_to_device(ctx.get_queue(),
                m_gpu_buffer->buffer,
                host_data(),
                bytes,
                "OpenCLTensorBackend copy ctor");
            m_needs_sync_to_device = false;
        }
    }
}

OpenCLTensorBackend& OpenCLTensorBackend::operator=(const OpenCLTensorBackend& other)
{
    if (this != &other)
    {
        other.sync_gpu_if_needed();
        // m_backend is replaced below; an in-flight upload is still reading it.
        wait_for_upload();
        m_backend =
            other.m_backend ? std::make_unique<OpenCLHostStorage>(*other.m_backend) : nullptr;
        m_grad_backend = other.m_grad_backend
                             ? std::make_unique<OpenCLTensorBackend>(*other.m_grad_backend)
                             : nullptr;

        m_gpu_buffer.reset();
        m_has_gpu_memory = false;
        m_gpu_resident = other.m_gpu_resident;
        m_pipeline_mode = other.m_pipeline_mode;
        m_needs_sync_to_host = false;
        m_needs_sync_to_device = true;

        if (other.m_has_gpu_memory && m_backend)
        {
            try_allocate_gpu_buffer(size());
            if (m_has_gpu_memory && m_gpu_buffer)
            {
                const auto& ctx = opencl::OpenCLContext::instance();
                const std::size_t bytes = size() * sizeof(float);
                copy_host_to_device(ctx.get_queue(),
                    m_gpu_buffer->buffer,
                    host_data(),
                    bytes,
                    "OpenCLTensorBackend copy assign");
                m_needs_sync_to_device = false;
            }
        }
    }
    return *this;
}

// Move/destroy cannot be defaulted: m_upload_event is a raw owning handle, and a
// defaulted move would leave both objects releasing the same cl_event.
OpenCLTensorBackend::OpenCLTensorBackend(OpenCLTensorBackend&& other) noexcept
    : m_backend(std::move(other.m_backend)),
      m_grad_backend(std::move(other.m_grad_backend)),
      m_gpu_buffer(std::move(other.m_gpu_buffer)),
      m_has_gpu_memory(other.m_has_gpu_memory),
      m_gpu_resident(other.m_gpu_resident),
      m_pipeline_mode(other.m_pipeline_mode),
      m_needs_sync_to_host(other.m_needs_sync_to_host),
      m_needs_sync_to_device(other.m_needs_sync_to_device),
      m_pending_events_count(other.m_pending_events_count),
      m_upload_event(other.m_upload_event)
{
    for (std::size_t i = 0; i < other.m_pending_events_count; ++i)
    {
        m_pending_events[i] = other.m_pending_events[i];
    }
    other.m_pending_events_count = 0;
    other.m_upload_event = nullptr;
    other.m_has_gpu_memory = false;
}

OpenCLTensorBackend& OpenCLTensorBackend::operator=(OpenCLTensorBackend&& other) noexcept
{
    if (this != &other)
    {
        // Our host storage is about to be replaced; any DMA still reading it
        // must finish first.
        wait_for_upload();

        m_backend = std::move(other.m_backend);
        m_grad_backend = std::move(other.m_grad_backend);
        m_gpu_buffer = std::move(other.m_gpu_buffer);
        m_has_gpu_memory = other.m_has_gpu_memory;
        m_gpu_resident = other.m_gpu_resident;
        m_pipeline_mode = other.m_pipeline_mode;
        m_needs_sync_to_host = other.m_needs_sync_to_host;
        m_needs_sync_to_device = other.m_needs_sync_to_device;
        m_pending_events_count = other.m_pending_events_count;
        for (std::size_t i = 0; i < other.m_pending_events_count; ++i)
        {
            m_pending_events[i] = other.m_pending_events[i];
        }
        m_upload_event = other.m_upload_event;

        other.m_pending_events_count = 0;
        other.m_upload_event = nullptr;
        other.m_has_gpu_memory = false;
    }
    return *this;
}

OpenCLTensorBackend::~OpenCLTensorBackend()
{
    // Host storage is freed right after this; an in-flight async upload is still
    // reading from it.
    wait_for_upload();
}

void OpenCLTensorBackend::wait_for_upload() const
{
    if (m_upload_event == nullptr) return;

    cl_event evt = m_upload_event;
    m_upload_event = nullptr; // clear first: clWaitForEvents may throw-free paths
    clWaitForEvents(1, &evt);
    clReleaseEvent(evt);
}

OpenCLTensorBackend::RuntimeScope::~RuntimeScope()
{
    if (active)
    {
        OpenCLTensorBackend::shutdown_buffer_pool();
    }
}

OpenCLTensorBackend::RuntimeScope::RuntimeScope(RuntimeScope&& other) noexcept
    : device_name(std::move(other.device_name)), active(other.active)
{
    other.active = false;
}

auto OpenCLTensorBackend::RuntimeScope::operator=(RuntimeScope&& other) noexcept -> RuntimeScope&
{
    if (this != &other)
    {
        if (active)
        {
            OpenCLTensorBackend::shutdown_buffer_pool();
        }
        device_name = std::move(other.device_name);
        active = other.active;
        other.active = false;
    }
    return *this;
}

// Static Factories
OpenCLTensorBackend OpenCLTensorBackend::zeros(Index rows, Index cols)
{
    OpenCLTensorBackend t(rows, cols);
    t.m_backend->fill(0.0F);
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::ones(Index rows, Index cols)
{
    OpenCLTensorBackend t(rows, cols);
    t.m_backend->fill(1.0F);
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::random(Index rows, Index cols)
{
    OpenCLTensorBackend t(rows, cols);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    for (Index i = 0; i < t.size(); ++i)
    {
        t.m_backend->at(i) = dist(rng);
    }
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::random(Index rows, Index cols, std::mt19937& rng)
{
    OpenCLTensorBackend t(rows, cols);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    for (Index i = 0; i < t.size(); ++i)
    {
        t.m_backend->at(i) = dist(rng);
    }
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::random(Index d1, Index d2, Index d3)
{
    std::mt19937 rng(std::random_device{}());
    return random(d1, d2, d3, rng);
}

OpenCLTensorBackend OpenCLTensorBackend::random(Index d1, Index d2, Index d3, std::mt19937& rng)
{
    OpenCLTensorBackend t(d1, d2, d3);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    for (Index i = 0; i < t.size(); ++i) t.m_backend->at(i) = dist(rng);
    return t;
}

bool OpenCLTensorBackend::operator==(const OpenCLTensorBackend& other) const
{
    if (shape() != other.shape()) return false;
    for (Index i = 0; i < size(); ++i)
        if (m_backend->at(i) != other.m_backend->at(i)) return false;
    return true;
}

// Shape & Access
const std::vector<Index>& OpenCLTensorBackend::shape() const
{
    return m_backend->shape();
}

void OpenCLTensorBackend::reshape(const std::vector<Index>& new_shape)
{
    // The permutation below reads/writes host data: pull any pending device
    // result first, and mark the device copy stale afterwards.
    sync_gpu_if_needed();
    wait_for_upload();
    m_backend->reshape(new_shape);
    m_needs_sync_to_host = false;
    m_needs_sync_to_device = true;
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
    sync_gpu_if_needed();
    wait_for_upload(); // caller gets a writable pointer into m_backend
    m_needs_sync_to_device = true;
    return m_backend->at(i);
}

const float& OpenCLTensorBackend::at(Index i) const
{
    sync_gpu_if_needed();
    return m_backend->at(i);
}

float& OpenCLTensorBackend::at(Index row, Index col)
{
    sync_gpu_if_needed();
    wait_for_upload(); // caller gets a writable pointer into m_backend
    m_needs_sync_to_device = true;
    return m_backend->at(row, col);
}

const float& OpenCLTensorBackend::at(Index row, Index col) const
{
    sync_gpu_if_needed();
    return m_backend->at(row, col);
}

float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3)
{
    sync_gpu_if_needed();
    wait_for_upload(); // caller gets a writable pointer into m_backend
    m_needs_sync_to_device = true;
    return m_backend->at(d1, d2, d3);
}

const float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3) const
{
    sync_gpu_if_needed();
    return m_backend->at(d1, d2, d3);
}

float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3, Index d4)
{
    sync_gpu_if_needed();
    wait_for_upload(); // caller gets a writable pointer into m_backend
    m_needs_sync_to_device = true;
    return m_backend->at(d1, d2, d3, d4);
}

const float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3, Index d4) const
{
    sync_gpu_if_needed();
    return m_backend->at(d1, d2, d3, d4);
}

float& OpenCLTensorBackend::at(const std::vector<Index>& indices)
{
    sync_gpu_if_needed();
    wait_for_upload(); // caller gets a writable pointer into m_backend
    m_needs_sync_to_device = true;
    return m_backend->at(indices);
}

const float& OpenCLTensorBackend::at(const std::vector<Index>& indices) const
{
    sync_gpu_if_needed();
    return m_backend->at(indices);
}

float* OpenCLTensorBackend::mutable_data_ptr()
{
    sync_gpu();
    wait_for_upload(); // caller gets a writable pointer into m_backend
    m_needs_sync_to_device = true;
    return m_backend->mutable_data_ptr();
}

const float* OpenCLTensorBackend::data_ptr() const
{
    sync_gpu();
    return m_backend->data_ptr();
}

// Sync-on-read host access used by the host-staged fallback paths. See the
// declaration in OpenCLTensorBackend.hpp for why these exist.
const float* OpenCLTensorBackend::host_data() const
{
    sync_gpu_if_needed();
    return m_backend->data_ptr();
}

float* OpenCLTensorBackend::mutable_host_data()
{
    sync_gpu_if_needed();
    wait_for_upload(); // caller gets a writable pointer into m_backend
    m_needs_sync_to_device = true;
    return m_backend->mutable_data_ptr();
}

OpenCLTensorBackend OpenCLTensorBackend::row(Index i) const
{
    if (shape().size() != 2) throw std::invalid_argument("row requires rank-2 tensor");
    if (i >= rows()) throw std::out_of_range("row index out of range");

    OpenCLTensorBackend out(1, cols());
    // src is row i of a column-major (R,C): element (0,c) at i + c*R.
    if (launch_strided_copy(*this, {i, 0, rows()}, out, {0, 0, 1}, 1, cols(), "row"))
    {
        return out;
    }

    sync_gpu();
    for (Index c = 0; c < cols(); ++c)
    {
        out.m_backend->at(0, c) = m_backend->at(i, c);
    }
    out.m_needs_sync_to_device = true;
    out.m_needs_sync_to_host = false;
    return out;
}

OpenCLTensorBackend OpenCLTensorBackend::col(Index j) const
{
    if (shape().size() != 2) throw std::invalid_argument("col requires rank-2 tensor");
    if (j >= cols()) throw std::out_of_range("col index out of range");

    OpenCLTensorBackend out(rows(), 1);
    // src column j is contiguous at j*R; dst (R,1) is contiguous at 0.
    if (launch_strided_copy(*this, {j * rows(), 1, 0}, out, {0, 1, 0}, rows(), 1, "col"))
    {
        return out;
    }

    sync_gpu();
    for (Index r = 0; r < rows(); ++r)
    {
        out.m_backend->at(r, 0) = m_backend->at(r, j);
    }
    out.m_needs_sync_to_device = true;
    out.m_needs_sync_to_host = false;
    return out;
}

OpenCLTensorBackend OpenCLTensorBackend::leftCols(Index n) const
{
    if (shape().size() != 2) throw std::invalid_argument("leftCols requires rank-2 tensor");
    if (n > cols()) throw std::out_of_range("leftCols exceeds tensor width");

    OpenCLTensorBackend out(rows(), n);
    // Leading columns are contiguous in column-major order.
    if (launch_strided_copy(*this, {0, 1, rows()}, out, {0, 1, rows()}, rows(), n, "leftCols"))
    {
        return out;
    }

    sync_gpu();
    for (Index r = 0; r < rows(); ++r)
    {
        for (Index c = 0; c < n; ++c)
        {
            out.m_backend->at(r, c) = m_backend->at(r, c);
        }
    }
    out.m_needs_sync_to_device = true;
    out.m_needs_sync_to_host = false;
    return out;
}

OpenCLTensorBackend OpenCLTensorBackend::topRows(Index n) const
{
    if (shape().size() != 2) throw std::invalid_argument("topRows requires rank-2 tensor");
    if (n > rows()) throw std::out_of_range("topRows exceeds tensor height");

    OpenCLTensorBackend out(n, cols());
    if (launch_strided_copy(*this, {0, 1, rows()}, out, {0, 1, n}, n, cols(), "topRows"))
    {
        return out;
    }

    sync_gpu();
    for (Index r = 0; r < n; ++r)
    {
        for (Index c = 0; c < cols(); ++c)
        {
            out.m_backend->at(r, c) = m_backend->at(r, c);
        }
    }
    out.m_needs_sync_to_device = true;
    out.m_needs_sync_to_host = false;
    return out;
}

void OpenCLTensorBackend::setBlock(Index row, Index col, const OpenCLTensorBackend& block)
{
    if (shape().size() != 2 || block.shape().size() != 2)
    {
        throw std::invalid_argument("setBlock requires rank-2 tensors");
    }
    if (row + block.rows() > rows() || col + block.cols() > cols())
    {
        throw std::invalid_argument("setBlock: block exceeds tensor bounds");
    }

    // Partial write: launch_strided_copy uploads our current contents first, so
    // the elements outside the block are preserved.
    if (launch_strided_copy(block,
            {0, 1, block.rows()},
            *this,
            {row + col * rows(), 1, rows()},
            block.rows(),
            block.cols(),
            "setBlock"))
    {
        return;
    }

    sync_gpu();
    block.sync_gpu();
    for (Index r = 0; r < block.rows(); ++r)
    {
        for (Index c = 0; c < block.cols(); ++c)
        {
            m_backend->at(row + r, col + c) = block.m_backend->at(r, c);
        }
    }
    m_needs_sync_to_device = true;
    m_needs_sync_to_host = false;
}

OpenCLTensorBackend OpenCLTensorBackend::slice(std::span<const int> indices) const
{
    sync_gpu();
    if (shape().size() != 2) throw std::invalid_argument("slice requires rank-2 tensor");

    OpenCLTensorBackend out(indices.size(), cols());
    for (Index i = 0; i < indices.size(); ++i)
    {
        const auto src_r = static_cast<Index>(indices[i]);
        if (src_r >= rows()) throw std::out_of_range("slice index out of range");
        for (Index c = 0; c < cols(); ++c)
        {
            out.m_backend->at(i, c) = m_backend->at(src_r, c);
        }
    }
    out.m_needs_sync_to_device = true;
    out.m_needs_sync_to_host = false;
    return out;
}

OpenCLTensorBackend OpenCLTensorBackend::slice_batch(Index b) const
{
    const auto& s = shape();
    if (s.size() != 3) throw std::invalid_argument("slice_batch requires rank-3 tensor");
    if (b >= s[0]) throw std::out_of_range("slice_batch index out of range");

    OpenCLTensorBackend out(s[1], s[2]);
    // src(b,t,d) = b + (t + d*T)*B, b fixed -> base b, stride_t B, stride_d T*B.
    if (launch_strided_copy(
            *this, {b, s[0], s[1] * s[0]}, out, {0, 1, s[1]}, s[1], s[2], "slice_batch"))
    {
        return out;
    }

    sync_gpu();
    for (Index t = 0; t < s[1]; ++t)
    {
        for (Index d = 0; d < s[2]; ++d)
        {
            out.m_backend->at(t, d) = m_backend->at(b, t, d);
        }
    }
    out.m_needs_sync_to_device = true;
    out.m_needs_sync_to_host = false;
    return out;
}

void OpenCLTensorBackend::set_batch_slice(Index b, const OpenCLTensorBackend& val)
{
    const auto& s = shape();
    if (s.size() != 3) throw std::invalid_argument("set_batch_slice requires rank-3 tensor");
    if (b >= s[0]) throw std::out_of_range("set_batch_slice index out of range");
    if (val.rows() != s[1] || val.cols() != s[2])
        throw std::invalid_argument("set_batch_slice value shape mismatch");

    if (launch_strided_copy(
            val, {0, 1, s[1]}, *this, {b, s[0], s[1] * s[0]}, s[1], s[2], "set_batch_slice"))
    {
        return;
    }

    sync_gpu();
    val.sync_gpu();
    for (Index t = 0; t < s[1]; ++t)
    {
        for (Index d = 0; d < s[2]; ++d)
        {
            m_backend->at(b, t, d) = val.m_backend->at(t, d);
        }
    }
    m_needs_sync_to_device = true;
    m_needs_sync_to_host = false;
}

OpenCLTensorBackend OpenCLTensorBackend::slice_time(Index t) const
{
    const auto& s = shape();
    if (s.size() != 3) throw std::invalid_argument("slice_time requires rank-3 tensor");
    if (t >= s[1]) throw std::out_of_range("slice_time index out of range");

    OpenCLTensorBackend out(s[0], s[2]);
    // src(b,t,d) = b + (t + d*T)*B, t fixed -> base t*B, stride_b 1, stride_d T*B.
    if (launch_strided_copy(
            *this, {t * s[0], 1, s[1] * s[0]}, out, {0, 1, s[0]}, s[0], s[2], "slice_time"))
    {
        return out;
    }

    sync_gpu();
    for (Index b = 0; b < s[0]; ++b)
    {
        for (Index d = 0; d < s[2]; ++d)
        {
            out.m_backend->at(b, d) = m_backend->at(b, t, d);
        }
    }
    out.m_needs_sync_to_device = true;
    out.m_needs_sync_to_host = false;
    return out;
}

void OpenCLTensorBackend::set_time_slice(Index t, const OpenCLTensorBackend& val)
{
    const auto& s = shape();
    if (s.size() != 3) throw std::invalid_argument("set_time_slice requires rank-3 tensor");
    if (t >= s[1]) throw std::out_of_range("set_time_slice index out of range");
    if (val.rows() != s[0] || val.cols() != s[2])
        throw std::invalid_argument("set_time_slice value shape mismatch");

    if (launch_strided_copy(
            val, {0, 1, s[0]}, *this, {t * s[0], 1, s[1] * s[0]}, s[0], s[2], "set_time_slice"))
    {
        return;
    }

    sync_gpu();
    val.sync_gpu();
    for (Index b = 0; b < s[0]; ++b)
    {
        for (Index d = 0; d < s[2]; ++d)
        {
            m_backend->at(b, t, d) = val.m_backend->at(b, d);
        }
    }
    m_needs_sync_to_device = true;
    m_needs_sync_to_host = false;
}

} // namespace nn
