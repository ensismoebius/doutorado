/**
 * @file src/core/tensor/opencl/OpenCLTensorBackend.cpp
 * @brief OpenCL-only tensor backend implementation.
 *
 * Tensor metadata and host synchronization staging are managed locally,
 * while math operations execute through OpenCL kernels only.
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

#include "logging/Logger.hpp"
#include "tensor/opencl/DeviceMemory.hpp"
#include "tensor/opencl/KernelManager.hpp"
#include "tensor/opencl/OpenCLContext.hpp"
#include "tensor/opencl/OpenCLProfiling.hpp"

namespace nn
{

class OpenCLHostStorage
{
   public:
    OpenCLHostStorage() = default;

    explicit OpenCLHostStorage(Index rows, Index cols)
        : m_shape({rows, cols}), m_data(rows * cols, 0.0F)
    {
    }

    explicit OpenCLHostStorage(Index d1, Index d2, Index d3)
        : m_shape({d1, d2, d3}), m_data(d1 * d2 * d3, 0.0F)
    {
    }

    explicit OpenCLHostStorage(Index d1, Index d2, Index d3, Index d4)
        : m_shape({d1, d2, d3, d4}), m_data(d1 * d2 * d3 * d4, 0.0F)
    {
    }

    explicit OpenCLHostStorage(const std::vector<Index>& shape)
        : m_shape(shape), m_data(total_size(shape), 0.0F)
    {
    }

    const std::vector<Index>& shape() const
    {
        return m_shape;
    }

    void reshape(const std::vector<Index>& new_shape)
    {
        if (total_size(new_shape) != size())
        {
            throw std::invalid_argument("Reshape total size mismatch");
        }
        // Reshape must reinterpret the tensor in ROW-MAJOR logical order to
        // match the XTensor backend (backend-parity contract). This storage is
        // column-major, so a metadata-only shape swap would silently produce a
        // different logical tensor. Permute the buffer so element k of the
        // row-major traversal is preserved across the reshape. 1-D storage is
        // layout-free; the permutation only matters when either side is >= 2-D.
        if (m_shape.size() >= 2 || new_shape.size() >= 2)
        {
            std::vector<float> permuted(m_data.size());
            const Index n = size();
            for (Index k = 0; k < n; ++k)
            {
                permuted[row_major_to_storage(new_shape, k)] =
                    m_data[row_major_to_storage(m_shape, k)];
            }
            m_data = std::move(permuted);
        }
        m_shape = new_shape;
    }

    Index rows() const
    {
        return m_shape.empty() ? 0 : m_shape[0];
    }

    Index cols() const
    {
        return m_shape.size() < 2 ? 1 : m_shape[1];
    }

    Index size() const
    {
        return m_data.size();
    }

    float* mutable_data_ptr()
    {
        return m_data.data();
    }

    const float* data_ptr() const
    {
        return m_data.data();
    }

    void fill(float value)
    {
        std::fill(m_data.begin(), m_data.end(), value);
    }

    float& at(Index i)
    {
        if (i >= size())
        {
            throw std::out_of_range("Index out of range");
        }
        return m_data[i];
    }

    const float& at(Index i) const
    {
        if (i >= size())
        {
            throw std::out_of_range("Index out of range");
        }
        return m_data[i];
    }

    float& at(Index row, Index col)
    {
        return m_data[offset_2d(row, col)];
    }

    const float& at(Index row, Index col) const
    {
        return m_data[offset_2d(row, col)];
    }

    float& at(Index d1, Index d2, Index d3)
    {
        return m_data[offset_3d(d1, d2, d3)];
    }

    const float& at(Index d1, Index d2, Index d3) const
    {
        return m_data[offset_3d(d1, d2, d3)];
    }

    float& at(Index d1, Index d2, Index d3, Index d4)
    {
        return m_data[offset_4d(d1, d2, d3, d4)];
    }

    const float& at(Index d1, Index d2, Index d3, Index d4) const
    {
        return m_data[offset_4d(d1, d2, d3, d4)];
    }

    float& at(const std::vector<Index>& indices)
    {
        return m_data[offset_nd(indices)];
    }

    const float& at(const std::vector<Index>& indices) const
    {
        return m_data[offset_nd(indices)];
    }

   private:
    /// Row-major linear index k (XTensor logical order, last dim fastest) →
    /// this storage's column-major offset (first dim fastest).
    static Index row_major_to_storage(const std::vector<Index>& shape, Index k)
    {
        if (shape.size() <= 1)
        {
            return k;
        }
        std::vector<Index> idx(shape.size());
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d)
        {
            idx[static_cast<std::size_t>(d)] = k % shape[static_cast<std::size_t>(d)];
            k /= shape[static_cast<std::size_t>(d)];
        }
        Index off = 0;
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d)
        {
            off = idx[static_cast<std::size_t>(d)] + shape[static_cast<std::size_t>(d)] * off;
        }
        return off;
    }

    static Index total_size(const std::vector<Index>& shape)
    {
        return std::accumulate(shape.begin(),
            shape.end(),
            static_cast<Index>(1),
            [](Index acc, Index dim) { return acc * dim; });
    }

    Index offset_2d(Index row, Index col) const
    {
        if (m_shape.size() != 2)
        {
            throw std::invalid_argument("at(row, col) is only valid for 2D tensors");
        }
        if (row >= m_shape[0] || col >= m_shape[1])
        {
            throw std::out_of_range("Index out of range");
        }
        return row + (col * m_shape[0]);
    }

    Index offset_3d(Index d1, Index d2, Index d3) const
    {
        if (m_shape.size() != 3)
        {
            throw std::invalid_argument("at(d1, d2, d3) is only valid for 3D tensors");
        }
        if (d1 >= m_shape[0] || d2 >= m_shape[1] || d3 >= m_shape[2])
        {
            throw std::out_of_range("Index out of range");
        }
        const Index col_idx = d2 + d3 * m_shape[1];
        return d1 + (col_idx * m_shape[0]);
    }

    Index offset_4d(Index d1, Index d2, Index d3, Index d4) const
    {
        if (m_shape.size() != 4)
        {
            throw std::invalid_argument("at(d1, d2, d3, d4) is only valid for 4D tensors");
        }
        if (d1 >= m_shape[0] || d2 >= m_shape[1] || d3 >= m_shape[2] || d4 >= m_shape[3])
        {
            throw std::out_of_range("Index out of range");
        }

        const Index col_idx = d2 + (d3 + d4 * m_shape[2]) * m_shape[1];
        return d1 + (col_idx * m_shape[0]);
    }

    Index offset_nd(const std::vector<Index>& indices) const
    {
        if (indices.size() != m_shape.size())
        {
            throw std::invalid_argument("Indices dimension mismatch");
        }

        if (indices.size() == 1)
        {
            return indices[0];
        }
        if (indices.size() == 2)
        {
            return offset_2d(indices[0], indices[1]);
        }
        if (indices.size() == 3)
        {
            return offset_3d(indices[0], indices[1], indices[2]);
        }
        if (indices.size() == 4)
        {
            return offset_4d(indices[0], indices[1], indices[2], indices[3]);
        }

        Index flat_idx = 0;
        Index stride = 1;
        for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i)
        {
            if (indices[static_cast<std::size_t>(i)] >= m_shape[static_cast<std::size_t>(i)])
            {
                throw std::out_of_range("Index out of range");
            }
            flat_idx += indices[static_cast<std::size_t>(i)] * stride;
            stride *= m_shape[static_cast<std::size_t>(i)];
        }
        return flat_idx;
    }

    std::vector<Index> m_shape;
    std::vector<float> m_data;
};

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

void finish_queue_if_not_batching(cl_command_queue queue, const char* context)
{
    if (opencl::OpenCLContext::is_batching())
    {
        return;
    }
    check_cl_error(clFinish(queue), context);
}

class AsyncTransferManager
{
   public:
    static AsyncTransferManager& instance()
    {
        static AsyncTransferManager mgr;
        return mgr;
    }

    static cl_event copy_host_to_device_async(
        cl_command_queue queue, cl_mem device_buffer, const float* host_data, std::size_t bytes)
    {
        cl_event evt = nullptr;
        if (bytes > 0)
        {
            check_cl_error(
                clEnqueueWriteBuffer(
                    queue, device_buffer, CL_FALSE, 0, bytes, host_data, 0, nullptr, &evt),
                "copy_host_to_device_async");
        }
        return evt;
    }

    static cl_event copy_device_to_host_async(
        cl_command_queue queue, cl_mem device_buffer, float* host_data, std::size_t bytes)
    {
        cl_event evt = nullptr;
        if (bytes > 0)
        {
            check_cl_error(
                clEnqueueReadBuffer(
                    queue, device_buffer, CL_FALSE, 0, bytes, host_data, 0, nullptr, &evt),
                "copy_device_to_host_async");
        }
        return evt;
    }

    static void enqueue_kernel(cl_command_queue queue,
        cl_kernel kernel,
        std::size_t global_size,
        std::size_t local_size,
        cl_event* out_event = nullptr)
    {
        check_cl_error(
            clEnqueueNDRangeKernel(
                queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, out_event),
            "enqueue_kernel");
    }

    void wait_for_event(cl_command_queue queue, cl_event evt)
    {
        if (evt)
        {
            cl_event wait_evt = evt;
#if defined(CL_VERSION_1_2)
            check_cl_error(
                clEnqueueBarrierWithWaitList(queue, 1, &wait_evt, nullptr), "wait_for_event");
#else
            check_cl_error(clEnqueueWaitForEvents(queue, 1, &wait_evt), "wait_for_event");
#endif
            clReleaseEvent(evt);
        }
    }

    void wait_and_release_events(cl_command_queue queue, cl_event* events, cl_uint num_events)
    {
        if (num_events > 0 && events)
        {
#if defined(CL_VERSION_1_2)
            check_cl_error(clEnqueueBarrierWithWaitList(queue, num_events, events, nullptr),
                "wait_and_release_events");
#else
            check_cl_error(
                clEnqueueWaitForEvents(queue, num_events, events), "wait_and_release_events");
#endif
            for (cl_uint i = 0; i < num_events; ++i)
            {
                if (events[i]) clReleaseEvent(events[i]);
            }
        }
    }

    static void sync_queue(cl_command_queue queue)
    {
        finish_queue_if_not_batching(queue, "sync_queue");
    }

   private:
    AsyncTransferManager() = default;
};

static bool can_use_opencl()
{
    // OpenCL runtime is intentionally disabled under ASan to avoid false positives
    // from third-party drivers while preserving CPU fallback semantics.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
    const char* force_opencl = std::getenv("NN_FORCE_OPENCL_UNDER_ASAN");
    if (force_opencl == nullptr || force_opencl[0] == '\0' || force_opencl[0] == '0')
    {
        return false;
    }
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && ((__SANITIZE_ADDRESS__ + 0) == 1)
    const char* force_opencl = std::getenv("NN_FORCE_OPENCL_UNDER_ASAN");
    if (force_opencl == nullptr || force_opencl[0] == '\0' || force_opencl[0] == '0')
    {
        return false;
    }
#endif
    volatile bool runtime_available = opencl::OpenCLContext::instance().is_available();
    return runtime_available;
}

void warn_opencl_cpu_fallback_once(const std::string& operation, const std::string& reason)
{
    static std::mutex warned_mutex;
    static std::unordered_set<std::string> warned_messages;

    const std::string message = "OPENCL BACKEND CANNOT EXECUTE " + operation + ": " + reason;
    std::lock_guard<std::mutex> lock(warned_mutex);
    if (warned_messages.insert(message).second)
    {
        NN_LOG_WARN(message);
    }
}

static bool can_use_opencl(const char* operation)
{
#if defined(__SANITIZE_ADDRESS__) && ((__SANITIZE_ADDRESS__ + 0) == 1)
    const char* force_opencl = std::getenv("NN_FORCE_OPENCL_UNDER_ASAN");
    if (force_opencl == nullptr || force_opencl[0] == '\0' || force_opencl[0] == '0')
    {
        warn_opencl_cpu_fallback_once(
            operation, "AddressSanitizer build disables OpenCL execution");
        return false;
    }
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
    const char* force_opencl = std::getenv("NN_FORCE_OPENCL_UNDER_ASAN");
    if (force_opencl == nullptr || force_opencl[0] == '\0' || force_opencl[0] == '0')
    {
        warn_opencl_cpu_fallback_once(
            operation, "AddressSanitizer build disables OpenCL execution");
        return false;
    }
#endif
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl())
    {
        warn_opencl_cpu_fallback_once(operation, "OpenCL runtime or device is not available");
        return false;
    }
    return true;
#else
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl())
    {
        warn_opencl_cpu_fallback_once(operation, "OpenCL runtime or device is not available");
        return false;
    }
    return true;
#endif
}

[[noreturn]] void throw_opencl_only_failure(const std::string& operation, const std::string& reason)
{
    throw std::runtime_error("OpenCL-only backend failure in " + operation + ": " + reason);
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

struct PendingEvent
{
    cl_event event;
};

class EventTracker
{
   public:
    static EventTracker& instance()
    {
        static EventTracker tracker;
        return tracker;
    }

    void add_pending_event(cl_event evt)
    {
        if (evt)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_pending_events.push_back({evt});
        }
    }

    void flush_all(cl_command_queue queue)
    {
        std::vector<PendingEvent> events;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            events = std::move(m_pending_events);
            m_pending_events.clear();
        }
        if (!events.empty())
        {
            std::vector<cl_event> cl_events;
            cl_events.reserve(events.size());
            std::transform(events.begin(),
                events.end(),
                std::back_inserter(cl_events),
                [](const PendingEvent& pe) { return pe.event; });
#if defined(CL_VERSION_1_2)
            check_cl_error(
                clEnqueueBarrierWithWaitList(
                    queue, static_cast<cl_uint>(cl_events.size()), cl_events.data(), nullptr),
                "flush_all");
#else
            check_cl_error(clEnqueueWaitForEvents(
                               queue, static_cast<cl_uint>(cl_events.size()), cl_events.data()),
                "flush_all");
#endif
            for (auto evt : cl_events)
            {
                if (evt) clReleaseEvent(evt);
            }
        }
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto& pe : m_pending_events)
        {
            if (pe.event) clReleaseEvent(pe.event);
        }
        m_pending_events.clear();
    }

   private:
    std::mutex m_mutex;
    std::vector<PendingEvent> m_pending_events;
    EventTracker() = default;
};

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

void copy_host_to_device_async(cl_command_queue queue,
    cl_mem device_buffer,
    const float* host_data,
    std::size_t bytes,
    const char* context,
    cl_event* out_event = nullptr)
{
    if (bytes > 0)
    {
        check_cl_error(
            clEnqueueWriteBuffer(
                queue, device_buffer, CL_FALSE, 0, bytes, host_data, 0, nullptr, out_event),
            context);
    }
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

void copy_device_to_host_async(cl_command_queue queue,
    cl_mem device_buffer,
    float* host_data,
    std::size_t bytes,
    const char* context,
    cl_event* out_event = nullptr)
{
    if (bytes > 0)
    {
        check_cl_error(
            clEnqueueReadBuffer(
                queue, device_buffer, CL_FALSE, 0, bytes, host_data, 0, nullptr, out_event),
            context);
    }
}

auto read_gpu_busy_percent(std::string_view gpu_busy_percent_path) -> std::optional<int>
{
    std::ifstream input{std::string(gpu_busy_percent_path)};
    if (!input.good())
    {
        return std::nullopt;
    }

    int value = 0;
    input >> value;
    if (input.fail())
    {
        return std::nullopt;
    }

    return value;
}

auto to_opencl_tensor(const nn::OpenCLTensorBackend& source) -> nn::OpenCLTensorBackend
{
    nn::OpenCLTensorBackend result(source.shape());
    for (Index i = 0; i < source.size(); ++i)
    {
        result.at(i) = source.at(i);
    }
    return result;
}

auto compute_opencl_reconstruction_mse(
    const nn::OpenCLTensorBackend& prediction, const nn::OpenCLTensorBackend& target) -> float
{
    auto prediction_gpu = to_opencl_tensor(prediction);
    auto target_gpu = to_opencl_tensor(target);
    auto diff_gpu = prediction_gpu.add(target_gpu.multiply_scalar(-1.0F));
    auto squared_gpu = diff_gpu.multiply(diff_gpu);
    auto row_sums = squared_gpu.rowwise_sum();

    float total = 0.0F;
    for (Index i = 0; i < row_sums.size(); ++i)
    {
        total += row_sums.at(i);
    }

    return prediction.size() == 0 ? 0.0F : total / static_cast<float>(prediction.size());
}
} // namespace

// Constructors
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

void OpenCLTensorBackend::add_row_broadcast_inplace(const OpenCLTensorBackend& row)
{
    if (shape().size() != 2 || row.shape().size() != 2 || row.rows() != 1 || row.cols() != cols())
    {
        throw std::invalid_argument("add_row_broadcast_inplace requires lhs=(N,M) and row=(1,M)");
    }

    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("add_row_broadcast_inplace"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto num_rows = rows();
            const auto num_cols = cols();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);
            const std::size_t row_bytes = num_cols * sizeof(float);

            if (ensure_device_current("resident gate") &&
                row.ensure_device_current("resident gate"))
            {
                if (m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        m_gpu_buffer->buffer,
                        host_data(),
                        bytes,
                        "add_row_broadcast_inplace resident data");
                    m_needs_sync_to_device = false;
                }
                if (row.m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        row.m_gpu_buffer->buffer,
                        row.host_data(),
                        row_bytes,
                        "add_row_broadcast_inplace resident row");
                    row.m_needs_sync_to_device = false;
                }

                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("add_col_vector_to_rows_kernel");
                const cl_mem data_mem = m_gpu_buffer->buffer;
                const cl_mem row_mem = row.m_gpu_buffer->buffer;
                const cl_uint rows_u32 = static_cast<cl_uint>(num_rows);
                const cl_uint cols_u32 = static_cast<cl_uint>(num_cols);

                check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem),
                    "add_row_broadcast_inplace resident");
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &row_mem),
                    "add_row_broadcast_inplace resident");
                check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32),
                    "add_row_broadcast_inplace resident");
                check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32),
                    "add_row_broadcast_inplace resident");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "add_row_broadcast_inplace resident");
                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto data_buf = pool->acquire(bytes);
                auto row_buf = pool->acquire(row_bytes);
                if (data_buf && row_buf)
                {
                    cl_event data_evt = nullptr;
                    cl_event row_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        data_buf->buffer,
                        host_data(),
                        bytes,
                        "add_row_broadcast_inplace",
                        &data_evt);
                    copy_host_to_device_async(ctx.get_queue(),
                        row_buf->buffer,
                        row.host_data(),
                        row_bytes,
                        "add_row_broadcast_inplace",
                        &row_evt);

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel(
                        "add_col_vector_to_rows_kernel");
                    const cl_mem data_mem = data_buf->buffer;
                    const cl_mem row_mem = row_buf->buffer;
                    const cl_uint rows_u32 = static_cast<cl_uint>(num_rows);
                    const cl_uint cols_u32 = static_cast<cl_uint>(num_cols);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem),
                        "add_row_broadcast_inplace");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &row_mem),
                        "add_row_broadcast_inplace");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32),
                        "add_row_broadcast_inplace");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32),
                        "add_row_broadcast_inplace");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (data_evt && row_evt)
                    {
                        cl_event wait_events[2] = {data_evt, row_evt};
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           2,
                                           wait_events,
                                           &kernel_evt),
                            "add_row_broadcast_inplace");
                    }
                    else if (data_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &data_evt,
                                           &kernel_evt),
                            "add_row_broadcast_inplace");
                    }
                    else if (row_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &row_evt,
                                           &kernel_evt),
                            "add_row_broadcast_inplace");
                    }
                    else
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           0,
                                           nullptr,
                                           &kernel_evt),
                            "add_row_broadcast_inplace");
                    }

                    if (data_evt) clReleaseEvent(data_evt);
                    if (row_evt) clReleaseEvent(row_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        data_buf->buffer,
                        mutable_host_data(),
                        bytes,
                        "add_row_broadcast_inplace",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory data_dev(bytes);
            opencl::DeviceMemory row_dev(row_bytes);
            data_dev.copy_to_device(host_data());
            row_dev.copy_to_device(row.host_data());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("add_col_vector_to_rows_kernel");
            const cl_mem data_mem = data_dev.get_device_buffer();
            const cl_mem row_mem = row_dev.get_device_buffer();
            const cl_uint rows_u32 = static_cast<cl_uint>(num_rows);
            const cl_uint cols_u32 = static_cast<cl_uint>(num_cols);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "add_row_broadcast_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &row_mem), "add_row_broadcast_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32), "add_row_broadcast_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32), "add_row_broadcast_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "add_row_broadcast_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "add_row_broadcast_inplace");

            data_dev.copy_from_device(mutable_host_data());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            warn_opencl_cpu_fallback_once(
                "add_row_broadcast_inplace", std::string("OpenCL path failed: ") + e.what());
        }
    }

    for (Index i = 0; i < rows(); ++i)
    {
        for (Index j = 0; j < cols(); ++j)
        {
            m_backend->at(i, j) += row.m_backend->at(0, j);
        }
    }

    m_needs_sync_to_device = true;
    m_needs_sync_to_host = false;
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
    if (!col_vector.m_gpu_resident)
    {
        col_vector.sync_gpu();
    }
    if (shape().size() != 2 || col_vector.shape().size() != 2)
    {
        warn_opencl_cpu_fallback_once(
            "add_col_vector_to_rows_inplace", "OpenCL path requires rank-2 tensors");
    }
    else if (cols() != col_vector.rows() || col_vector.cols() != 1)
    {
        warn_opencl_cpu_fallback_once(
            "add_col_vector_to_rows_inplace", "OpenCL path requires col_vector to be (cols x 1)");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("add_col_vector_to_rows_inplace"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto num_rows = rows();
            const auto num_cols = cols();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);
            const std::size_t col_bytes = num_cols * sizeof(float);

            if (ensure_device_current("resident gate") &&
                col_vector.ensure_device_current("resident gate"))
            {
                if (m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        m_gpu_buffer->buffer,
                        host_data(),
                        bytes,
                        "add_col_vector_to_rows_inplace resident data");
                    m_needs_sync_to_device = false;
                }
                if (col_vector.m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        col_vector.m_gpu_buffer->buffer,
                        col_vector.host_data(),
                        col_bytes,
                        "add_col_vector_to_rows_inplace resident col");
                    col_vector.m_needs_sync_to_device = false;
                }

                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("add_col_vector_to_rows_kernel");
                const cl_mem data_mem = m_gpu_buffer->buffer;
                const cl_mem col_mem = col_vector.m_gpu_buffer->buffer;
                const cl_uint rows_u32 = static_cast<cl_uint>(num_rows);
                const cl_uint cols_u32 = static_cast<cl_uint>(num_cols);

                check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem),
                    "add_col_vector_to_rows_inplace resident");
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &col_mem),
                    "add_col_vector_to_rows_inplace resident");
                check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32),
                    "add_col_vector_to_rows_inplace resident");
                check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32),
                    "add_col_vector_to_rows_inplace resident");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "add_col_vector_to_rows_inplace resident");
                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto data_buf = pool->acquire(bytes);
                auto col_buf = pool->acquire(col_bytes);
                if (data_buf && col_buf)
                {
                    cl_event data_evt = nullptr;
                    cl_event col_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        data_buf->buffer,
                        host_data(),
                        bytes,
                        "add_col_vector_to_rows_inplace",
                        &data_evt);
                    copy_host_to_device_async(ctx.get_queue(),
                        col_buf->buffer,
                        col_vector.host_data(),
                        col_bytes,
                        "add_col_vector_to_rows_inplace",
                        &col_evt);

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel(
                        "add_col_vector_to_rows_kernel");
                    const cl_mem data_mem = data_buf->buffer;
                    const cl_mem col_mem = col_buf->buffer;
                    const cl_uint rows_u32 = static_cast<cl_uint>(num_rows);
                    const cl_uint cols_u32 = static_cast<cl_uint>(num_cols);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem),
                        "add_col_vector_to_rows_inplace");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &col_mem),
                        "add_col_vector_to_rows_inplace");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32),
                        "add_col_vector_to_rows_inplace");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32),
                        "add_col_vector_to_rows_inplace");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (data_evt && col_evt)
                    {
                        cl_event wait_events[2] = {data_evt, col_evt};
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           2,
                                           wait_events,
                                           &kernel_evt),
                            "add_col_vector_to_rows_inplace");
                    }
                    else if (data_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &data_evt,
                                           &kernel_evt),
                            "add_col_vector_to_rows_inplace");
                    }
                    else if (col_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &col_evt,
                                           &kernel_evt),
                            "add_col_vector_to_rows_inplace");
                    }
                    else
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           0,
                                           nullptr,
                                           &kernel_evt),
                            "add_col_vector_to_rows_inplace");
                    }

                    if (data_evt) clReleaseEvent(data_evt);
                    if (col_evt) clReleaseEvent(col_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        data_buf->buffer,
                        mutable_host_data(),
                        bytes,
                        "add_col_vector_to_rows_inplace",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory data_dev(bytes);
            opencl::DeviceMemory col_dev(col_bytes);
            data_dev.copy_to_device(host_data());
            col_dev.copy_to_device(col_vector.host_data());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("add_col_vector_to_rows_kernel");
            const cl_mem data_mem = data_dev.get_device_buffer();
            const cl_mem col_mem = col_dev.get_device_buffer();
            const cl_uint rows_u32 = static_cast<cl_uint>(num_rows);
            const cl_uint cols_u32 = static_cast<cl_uint>(num_cols);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem),
                "add_col_vector_to_rows_inplace");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &col_mem),
                "add_col_vector_to_rows_inplace");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32),
                "add_col_vector_to_rows_inplace");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32),
                "add_col_vector_to_rows_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "add_col_vector_to_rows_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "add_col_vector_to_rows_inplace");

            data_dev.copy_from_device(mutable_host_data());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("add_col_vector_to_rows_inplace", e.what());
        }
    }
    throw_opencl_only_failure(
        "add_col_vector_to_rows_inplace", "OpenCL runtime unavailable or tensor shape is invalid");
}

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

bool OpenCLTensorBackend::hasNaN() const
{
    sync_gpu();
    for (Index i = 0; i < size(); ++i)
    {
        if (std::isnan(m_backend->at(i))) return true;
    }
    return false;
}

OpenCLTensorBackend OpenCLTensorBackend::abs() const
{
    return unary_elementwise("abs_kernel", "abs");
}

OpenCLTensorBackend OpenCLTensorBackend::relu() const
{
    return unary_elementwise("relu_kernel", "relu");
}

// Every unary kernel in KernelManager.cpp takes (in, out, scalars..., n) —
// square has no scalars, leaky_relu one (alpha), clamp two (min, max) — so the
// index of the size argument is 2 + scalars.size(). Computing it in one place
// is what lets those three otherwise identical bodies collapse into one.
void OpenCLTensorBackend::enqueue_unary_kernel(const char* kernel_name,
    const char* what,
    cl_mem in_mem,
    cl_mem out_mem,
    std::initializer_list<float> scalars,
    std::size_t n,
    const cl_event* wait_event,
    cl_event* out_event)
{
    const auto& ctx = opencl::OpenCLContext::instance();
    cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
    const cl_uint n_u32 = static_cast<cl_uint>(n);

    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), what);
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), what);
    cl_uint arg_index = 2;
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

// Stage 1: input already on the device and the result can stay there, so this
// call moves no bytes across the bus. Empty optional = strategy unavailable.
std::optional<OpenCLTensorBackend> OpenCLTensorBackend::unary_stage_resident(
    const char* kernel_name,
    const char* what,
    std::initializer_list<float> scalars,
    std::size_t n) const
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr || !ensure_device_current("resident gate")) return std::nullopt;

    auto out_buf = pool->acquire(n * sizeof(float));
    if (!out_buf || !m_gpu_buffer) return std::nullopt;

    enqueue_unary_kernel(
        kernel_name, what, m_gpu_buffer->buffer, out_buf->buffer, scalars, n, nullptr, nullptr);
    finish_queue_if_not_batching(opencl::OpenCLContext::instance().get_queue(), what);

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(OpenCLHostStorage(shape()));
    t.m_has_gpu_memory = true;
    t.m_gpu_buffer = std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
    t.set_gpu_resident(true);
    t.m_needs_sync_to_host = true;
    t.m_needs_sync_to_device = false;
    return t;
}

// Stage 2: staged through pooled buffers, upload/kernel/download chained on
// events so the host does not block between them.
std::optional<OpenCLTensorBackend> OpenCLTensorBackend::unary_stage_pooled(const char* kernel_name,
    const char* what,
    std::initializer_list<float> scalars,
    std::size_t n,
    OpenCLHostStorage& out) const
{
    tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
    if (pool == nullptr) return std::nullopt;

    const std::size_t bytes = n * sizeof(float);
    auto input_buf = pool->acquire(bytes);
    auto out_buf = pool->acquire(bytes);
    if (!input_buf || !out_buf) return std::nullopt;

    const auto& ctx = opencl::OpenCLContext::instance();
    cl_event h2d_evt = nullptr;
    copy_host_to_device_async(
        ctx.get_queue(), input_buf->buffer, host_data(), bytes, what, &h2d_evt);

    cl_event kernel_evt = nullptr;
    enqueue_unary_kernel(
        kernel_name, what, input_buf->buffer, out_buf->buffer, scalars, n, &h2d_evt, &kernel_evt);

    if (h2d_evt) clReleaseEvent(h2d_evt);

    cl_event d2h_evt = nullptr;
    copy_device_to_host_async(
        ctx.get_queue(), out_buf->buffer, out.mutable_data_ptr(), bytes, what, &d2h_evt);

    if (kernel_evt) clReleaseEvent(kernel_evt);

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    t.record_pending_gpu_op(d2h_evt);
    return t;
}

// Stage 3: no pool, one-shot device buffers, fully synchronous. No successor,
// so it either produces a result or throws.
OpenCLTensorBackend OpenCLTensorBackend::unary_stage_oneshot(const char* kernel_name,
    const char* what,
    std::initializer_list<float> scalars,
    std::size_t n,
    OpenCLHostStorage& out) const
{
    const std::size_t bytes = n * sizeof(float);
    opencl::DeviceMemory input_dev(bytes);
    opencl::DeviceMemory out_dev(bytes);
    input_dev.copy_to_device(host_data());

    enqueue_unary_kernel(kernel_name,
        what,
        input_dev.get_device_buffer(),
        out_dev.get_device_buffer(),
        scalars,
        n,
        nullptr,
        nullptr);
    finish_queue_if_not_batching(opencl::OpenCLContext::instance().get_queue(), what);

    out_dev.copy_from_device(out.mutable_data_ptr());

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
    return t;
}

// The three stages in cheapest-transfer-first order, shared by the no-scalar
// and the with-scalars entry points.
OpenCLTensorBackend OpenCLTensorBackend::run_unary_stages(
    const char* kernel_name, const char* what, std::initializer_list<float> scalars) const
{
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl(what))
    {
        throw_opencl_only_failure(what, "OpenCL runtime unavailable");
    }

    try
    {
        const auto n = size();
        if (n == 0)
        {
            OpenCLTensorBackend empty(*this);
            return empty;
        }

        if (auto resident = unary_stage_resident(kernel_name, what, scalars, n)) return *resident;

        sync_gpu();
        OpenCLHostStorage out(shape());
        if (auto pooled = unary_stage_pooled(kernel_name, what, scalars, n, out)) return *pooled;
        return unary_stage_oneshot(kernel_name, what, scalars, n, out);
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure(what, e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::unary_scalars_elementwise(
    const char* kernel_name, const char* what, std::initializer_list<float> scalars) const
{
    return run_unary_stages(kernel_name, what, scalars);
}

OpenCLTensorBackend OpenCLTensorBackend::leaky_relu(float alpha) const
{
    return unary_scalars_elementwise("leaky_relu_kernel", "leaky_relu", {alpha});
}

// Every shape/nullness precondition lif_step_inplace has, pulled out so the
// kernel-launch body below is only the launch.
void OpenCLTensorBackend::validate_lif_step_shapes(const OpenCLTensorBackend& input,
    const OpenCLTensorBackend& output,
    const OpenCLTensorBackend* adapt_a,
    bool use_adaptation) const
{
    if (shape() != input.shape() || shape() != output.shape())
    {
        throw std::invalid_argument("lif_step_inplace requires v_mem/input/output with same shape");
    }
    if (use_adaptation)
    {
        if (adapt_a == nullptr)
        {
            throw std::invalid_argument(
                "lif_step_inplace requires adapt_a when use_adaptation=true");
        }
        adapt_a->sync_gpu_if_needed();
        if (shape() != adapt_a->shape())
        {
            throw std::invalid_argument("lif_step_inplace requires adapt_a with same shape");
        }
    }
}

// lif_step_kernel's 12-argument launch, staged through one-shot device
// buffers (this op has no resident/pooled variant -- BPTT calls it once per
// time step, so there is no hot loop to amortize a pool against).
void OpenCLTensorBackend::launch_lif_step_kernel(const OpenCLTensorBackend& input,
    OpenCLTensorBackend& output,
    OpenCLTensorBackend* adapt_a,
    float beta,
    float threshold,
    float reset_potential,
    bool reset_zero,
    float adapt_decay,
    float adapt_coupling,
    bool use_adaptation,
    std::size_t n)
{
    const auto& ctx = opencl::OpenCLContext::instance();
    const std::size_t bytes = n * sizeof(float);

    opencl::DeviceMemory v_mem_dev(bytes);
    opencl::DeviceMemory input_dev(bytes);
    opencl::DeviceMemory output_dev(bytes);
    std::unique_ptr<opencl::DeviceMemory> adapt_dev;

    v_mem_dev.copy_to_device(host_data());
    input_dev.copy_to_device(input.host_data());
    if (use_adaptation)
    {
        adapt_dev = std::make_unique<opencl::DeviceMemory>(bytes);
        adapt_dev->copy_to_device(adapt_a->host_data());
    }

    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("lif_step_kernel");
    const cl_mem v_mem = v_mem_dev.get_device_buffer();
    const cl_mem in_mem = input_dev.get_device_buffer();
    const cl_mem out_mem = output_dev.get_device_buffer();
    cl_mem adapt_mem = adapt_dev ? adapt_dev->get_device_buffer() : nullptr;

    const cl_int reset_zero_i32 = reset_zero ? 1 : 0;
    const cl_int use_adaptation_i32 = use_adaptation ? 1 : 0;
    const cl_uint n_u32 = static_cast<cl_uint>(n);

    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_mem), "lif_step_inplace");
    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &in_mem), "lif_step_inplace");
    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), "lif_step_inplace");
    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &adapt_mem), "lif_step_inplace");
    check_cl_error(clSetKernelArg(kernel, 4, sizeof(float), &beta), "lif_step_inplace");
    check_cl_error(clSetKernelArg(kernel, 5, sizeof(float), &threshold), "lif_step_inplace");
    check_cl_error(clSetKernelArg(kernel, 6, sizeof(float), &reset_potential), "lif_step_inplace");
    check_cl_error(clSetKernelArg(kernel, 7, sizeof(cl_int), &reset_zero_i32), "lif_step_inplace");
    check_cl_error(clSetKernelArg(kernel, 8, sizeof(float), &adapt_decay), "lif_step_inplace");
    check_cl_error(clSetKernelArg(kernel, 9, sizeof(float), &adapt_coupling), "lif_step_inplace");
    check_cl_error(
        clSetKernelArg(kernel, 10, sizeof(cl_int), &use_adaptation_i32), "lif_step_inplace");
    check_cl_error(clSetKernelArg(kernel, 11, sizeof(cl_uint), &n_u32), "lif_step_inplace");

    const std::size_t local = 256;
    std::size_t global = round_up(n, local);
    check_cl_error(clEnqueueNDRangeKernel(
                       ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
        "lif_step_inplace");
    finish_queue_if_not_batching(ctx.get_queue(), "lif_step_inplace");

    v_mem_dev.copy_from_device(mutable_host_data());
    mark_host_dirty();
    output_dev.copy_from_device(output.mutable_host_data());
    output.mark_host_dirty();
    if (use_adaptation)
    {
        adapt_dev->copy_from_device(adapt_a->mutable_host_data());
        adapt_a->mark_host_dirty();
    }
}

void OpenCLTensorBackend::lif_step_inplace(const OpenCLTensorBackend& input,
    OpenCLTensorBackend& output,
    OpenCLTensorBackend* adapt_a,
    float beta,
    float threshold,
    float reset_potential,
    bool reset_zero,
    float adapt_decay,
    float adapt_coupling,
    bool use_adaptation)
{
    sync_gpu_if_needed();
    input.sync_gpu_if_needed();
    output.sync_gpu_if_needed();
    validate_lif_step_shapes(input, output, adapt_a, use_adaptation);

    const auto n = size();
    if (n == 0) return;

    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl("lif_step_inplace"))
    {
        throw_opencl_only_failure("lif_step_inplace", "OpenCL runtime unavailable");
    }

    try
    {
        launch_lif_step_kernel(input,
            output,
            adapt_a,
            beta,
            threshold,
            reset_potential,
            reset_zero,
            adapt_decay,
            adapt_coupling,
            use_adaptation,
            n);
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("lif_step_inplace", e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::lif_grad(float threshold, float sharpness) const
{
    sync_gpu_if_needed();

    const auto n = size();
    if (n == 0)
    {
        OpenCLTensorBackend empty(*this);
        return empty;
    }

    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("lif_grad"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = n * sizeof(float);

            OpenCLHostStorage out(shape());

            opencl::DeviceMemory v_pre_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            v_pre_dev.copy_to_device(host_data());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("lif_grad_kernel");
            const cl_mem in_mem = v_pre_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "lif_grad");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "lif_grad");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &threshold), "lif_grad");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(float), &sharpness), "lif_grad");
            check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32), "lif_grad");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "lif_grad");
            finish_queue_if_not_batching(ctx.get_queue(), "lif_grad");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("lif_grad", e.what());
        }
    }

    throw_opencl_only_failure("lif_grad", "OpenCL runtime unavailable");
}

OpenCLTensorBackend OpenCLTensorBackend::clamp(float min_val, float max_val) const
{
    return unary_scalars_elementwise("clamp_kernel", "clamp", {min_val, max_val});
}

void OpenCLTensorBackend::clamp_inplace(float min_val, float max_val)
{
    // "clamp_inplace" everywhere now; the pre-refactor body used "clamp" for
    // can_use_opencl() and "clamp_inplace" for everything else, a harmless
    // (operation is diagnostics-only, see can_use_opencl()) but needless
    // inconsistency that a shared helper cannot preserve without carrying it
    // forward on purpose.
    run_inplace_elementwise_stages("clamp_inplace_kernel", "clamp_inplace", {min_val, max_val});
}

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
        const auto& ctx = opencl::OpenCLContext::instance();
        const Index m = rows();
        const Index k = cols();
        const Index n = other.rows();

        const std::size_t a_bytes = m * k * sizeof(float);
        const std::size_t b_bytes = n * k * sizeof(float);
        const std::size_t bias_bytes = n * sizeof(float);
        const std::size_t c_bytes = m * n * sizeof(float);
        OpenCLHostStorage out(m, n);

        // GPU-resident fast path
        if (ensure_device_current("resident gate") &&
            other.ensure_device_current("resident gate") &&
            bias.ensure_device_current("resident gate"))
        {
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
            cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
            const cl_mem a_mem = m_gpu_buffer->buffer;
            const cl_mem b_mem = other.m_gpu_buffer->buffer;
            const cl_mem bias_mem = bias.m_gpu_buffer->buffer;
            const cl_mem c_mem = t.m_gpu_buffer->buffer;
            const cl_uint m_u32 = static_cast<cl_uint>(m);
            const cl_uint n_u32 = static_cast<cl_uint>(n);
            const cl_uint k_u32 = static_cast<cl_uint>(k);
            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                (std::string(what) + " arg0").c_str());
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                (std::string(what) + " arg1").c_str());
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem),
                (std::string(what) + " arg2").c_str());
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem),
                (std::string(what) + " arg3").c_str());
            check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32),
                (std::string(what) + " arg4").c_str());
            check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32),
                (std::string(what) + " arg5").c_str());
            check_cl_error(clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32),
                (std::string(what) + " arg6").c_str());
            cl_uint extra_index = 7;
            for (const float scalar : extra_scalars)
            {
                check_cl_error(clSetKernelArg(kernel, extra_index, sizeof(float), &scalar), what);
                ++extra_index;
            }
            const std::size_t global[2] = {m, n};
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                (std::string("clEnqueueNDRangeKernel(") + what + ")").c_str());
            t.m_needs_sync_to_host = true;
            t.m_needs_sync_to_device = false;
            return t;
        }

        // Pool path
        tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
        if (pool)
        {
            auto a_buf = pool->acquire(a_bytes);
            auto b_buf = pool->acquire(b_bytes);
            auto bias_buf = pool->acquire(bias_bytes);
            auto c_buf = pool->acquire(c_bytes);
            if (a_buf && b_buf && bias_buf && c_buf)
            {
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

                cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
                const cl_mem a_mem = a_buf->buffer;
                const cl_mem b_mem = b_buf->buffer;
                const cl_mem bias_mem = bias_buf->buffer;
                const cl_mem c_mem = c_buf->buffer;
                const cl_uint m_u32 = static_cast<cl_uint>(m);
                const cl_uint n_u32 = static_cast<cl_uint>(n);
                const cl_uint k_u32 = static_cast<cl_uint>(k);
                check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                    (std::string(what) + " arg0").c_str());
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                    (std::string(what) + " arg1").c_str());
                check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem),
                    (std::string(what) + " arg2").c_str());
                check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem),
                    (std::string(what) + " arg3").c_str());
                check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32),
                    (std::string(what) + " arg4").c_str());
                check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32),
                    (std::string(what) + " arg5").c_str());
                check_cl_error(clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32),
                    (std::string(what) + " arg6").c_str());
                cl_uint extra_index = 7;
                for (const float scalar : extra_scalars)
                {
                    check_cl_error(
                        clSetKernelArg(kernel, extra_index, sizeof(float), &scalar), what);
                    ++extra_index;
                }
                const std::size_t global[2] = {m, n};
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                    (std::string("clEnqueueNDRangeKernel(") + what + ")").c_str());
                finish_queue_if_not_batching(
                    ctx.get_queue(), (std::string("clFinish(") + what + ")").c_str());
                copy_device_to_host(ctx.get_queue(),
                    c_buf->buffer,
                    out.mutable_data_ptr(),
                    c_bytes,
                    (std::string("clEnqueueReadBuffer(") + what + ", c)").c_str());
                OpenCLTensorBackend t;
                t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                return t;
            }
        }

        // DeviceMemory fallback
        opencl::DeviceMemory a_dev(a_bytes);
        opencl::DeviceMemory b_dev(b_bytes);
        opencl::DeviceMemory bias_dev(bias_bytes);
        opencl::DeviceMemory out_dev(c_bytes);
        a_dev.copy_to_device(host_data());
        b_dev.copy_to_device(other.host_data());
        bias_dev.copy_to_device(bias.host_data());

        cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
        const cl_mem a_mem = a_dev.get_device_buffer();
        const cl_mem b_mem = b_dev.get_device_buffer();
        const cl_mem bias_mem = bias_dev.get_device_buffer();
        const cl_mem c_mem = out_dev.get_device_buffer();
        const cl_uint m_u32 = static_cast<cl_uint>(m);
        const cl_uint n_u32 = static_cast<cl_uint>(n);
        const cl_uint k_u32 = static_cast<cl_uint>(k);
        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
            (std::string(what) + " arg0").c_str());
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
            (std::string(what) + " arg1").c_str());
        check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem),
            (std::string(what) + " arg2").c_str());
        check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem),
            (std::string(what) + " arg3").c_str());
        check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32),
            (std::string(what) + " arg4").c_str());
        check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32),
            (std::string(what) + " arg5").c_str());
        check_cl_error(clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32),
            (std::string(what) + " arg6").c_str());
        cl_uint extra_index = 7;
        for (const float scalar : extra_scalars)
        {
            check_cl_error(clSetKernelArg(kernel, extra_index, sizeof(float), &scalar), what);
            ++extra_index;
        }
        const std::size_t global[2] = {m, n};
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
            (std::string("clEnqueueNDRangeKernel(") + what + ")").c_str());
        finish_queue_if_not_batching(
            ctx.get_queue(), (std::string("clFinish(") + what + ")").c_str());
        out_dev.copy_from_device(out.mutable_data_ptr());

        OpenCLTensorBackend t;
        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
        return t;
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

// Gradient management
OpenCLTensorBackend& OpenCLTensorBackend::grad_ref()
{
    if (!m_grad_backend)
    {
        m_grad_backend = std::make_unique<OpenCLTensorBackend>(shape());
        m_grad_backend->m_backend = std::make_unique<OpenCLHostStorage>(shape());
    }
    return *m_grad_backend;
}

OpenCLTensorBackend OpenCLTensorBackend::get_grad() const
{
    if (!m_grad_backend) return OpenCLTensorBackend(shape());
    return *m_grad_backend;
}

void OpenCLTensorBackend::set_grad(const OpenCLTensorBackend& grad)
{
    if (shape() != grad.shape())
    {
        throw std::invalid_argument("set_grad requires gradient shape to match tensor shape");
    }
    if (!m_grad_backend)
    {
        m_grad_backend = std::make_unique<OpenCLTensorBackend>(grad);
    }
    else
    {
        *m_grad_backend = grad;
    }
}

void OpenCLTensorBackend::zero_grad()
{
    if (m_grad_backend)
    {
        m_grad_backend->m_backend->fill(0.0F);
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

std::string OpenCLTensorBackend::initialize_runtime_or_throw(bool opencl_profiling_enabled)
{
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
    throw std::runtime_error(
        "OpenCL backend initialization failed: AddressSanitizer-enabled build disables OpenCL "
        "execution");
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
    throw std::runtime_error(
        "OpenCL backend initialization failed: AddressSanitizer-enabled build disables OpenCL "
        "execution");
#endif

    // Deliberately does NOT call request_queue_profiling(false) when event
    // profiling is off. Queue profiling must stay on regardless: it paces
    // rusticl and masks a driver race that otherwise corrupts the heap
    // (see OpenCLContext::request_queue_profiling). Event-level profiling
    // needs a profiling queue, so enabling it here is safe; disabling it
    // would silently re-arm the hazard for every caller.
    if (opencl_profiling_enabled)
    {
        opencl::OpenCLContext::request_queue_profiling(true);
    }

    const auto& opencl_context = opencl::OpenCLContext::instance();
    if (!opencl_context.is_available())
    {
        throw std::runtime_error(
            "OpenCL backend initialization failed: no OpenCL device/context is available");
    }

    OpenCLTensorBackend::init_buffer_pool(opencl_context.get_context(), opencl_context.get_queue());
    opencl::profiling::set_enabled(opencl_profiling_enabled);
    return opencl_context.get_device_name();
}

OpenCLTensorBackend::RuntimeScope OpenCLTensorBackend::start_runtime_scope_or_throw(
    bool opencl_profiling_enabled)
{
    RuntimeScope scope;
    scope.device_name = initialize_runtime_or_throw(opencl_profiling_enabled);
    scope.active = true;
    return scope;
}

void OpenCLTensorBackend::verify_runtime_activity_or_throw(const OpenCLTensorBackend& prediction,
    const OpenCLTensorBackend& target,
    std::string_view gpu_busy_percent_path)
{
    const auto& opencl_context = opencl::OpenCLContext::instance();
    if (!opencl_context.is_available())
    {
        throw std::runtime_error(
            "OpenCL device requested but no OpenCL device/context is available");
    }

    const std::optional<int> busy_before = read_gpu_busy_percent(gpu_busy_percent_path);
    int peak_busy_percent = busy_before.value_or(0);
    float gpu_probe_loss = 0.0F;

    for (int iteration = 0; iteration < 8; ++iteration)
    {
        gpu_probe_loss = compute_opencl_reconstruction_mse(prediction, target);
        opencl_context.flush();
        if (const std::optional<int> busy_now = read_gpu_busy_percent(gpu_busy_percent_path))
        {
            peak_busy_percent = std::max(peak_busy_percent, *busy_now);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }

    std::string status = std::string("OpenCL verification probe complete") +
                         " | device=" + opencl_context.get_device_name() +
                         " | gpu_probe_mse=" + std::to_string(gpu_probe_loss);
    if (busy_before.has_value())
    {
        status += " | gpu_busy_before=" + std::to_string(*busy_before) +
                  "% | gpu_busy_peak=" + std::to_string(peak_busy_percent) + "%";
    }
    else
    {
        status += " | gpu_busy_percent unavailable at " + std::string(gpu_busy_percent_path);
    }
    NN_LOG_INFO(status);

    if (busy_before.has_value() && peak_busy_percent <= *busy_before)
    {
        throw std::runtime_error(
            "OpenCL device requested but gpu_busy_percent did not increase during the GPU probe");
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

// ── Device-resident fast path (see header) ─────────────────────────────────

bool OpenCLTensorBackend::adam_step_inplace(OpenCLTensorBackend& moment1,
    OpenCLTensorBackend& moment2,
    const OpenCLTensorBackend& grad,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float bias_correction1,
    float bias_correction2)
{
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl("adam_step_inplace")) return false;
    if (shape() != moment1.shape() || shape() != moment2.shape() || shape() != grad.shape())
        return false;
    try
    {
        if (!ensure_device_current("adam_step, param") ||
            !moment1.ensure_device_current("adam_step, m1") ||
            !moment2.ensure_device_current("adam_step, m2") ||
            !grad.ensure_device_current("adam_step, grad"))
            return false;

        const auto& ctx = opencl::OpenCLContext::instance();
        cl_kernel kernel = opencl::KernelManager::instance().get_kernel("adam_step_kernel");
        const cl_mem p_mem = m_gpu_buffer->buffer;
        const cl_mem m1_mem = moment1.m_gpu_buffer->buffer;
        const cl_mem m2_mem = moment2.m_gpu_buffer->buffer;
        const cl_mem g_mem = grad.m_gpu_buffer->buffer;
        const cl_uint n_u32 = static_cast<cl_uint>(size());

        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &p_mem), "adam_step");
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &m1_mem), "adam_step");
        check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &m2_mem), "adam_step");
        check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &g_mem), "adam_step");
        check_cl_error(clSetKernelArg(kernel, 4, sizeof(float), &lr), "adam_step");
        check_cl_error(clSetKernelArg(kernel, 5, sizeof(float), &beta1), "adam_step");
        check_cl_error(clSetKernelArg(kernel, 6, sizeof(float), &beta2), "adam_step");
        check_cl_error(clSetKernelArg(kernel, 7, sizeof(float), &epsilon), "adam_step");
        check_cl_error(clSetKernelArg(kernel, 8, sizeof(float), &bias_correction1), "adam_step");
        check_cl_error(clSetKernelArg(kernel, 9, sizeof(float), &bias_correction2), "adam_step");
        check_cl_error(clSetKernelArg(kernel, 10, sizeof(cl_uint), &n_u32), "adam_step");

        const std::size_t local = 256;
        const std::size_t global = round_up(size(), local);
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
            "adam_step");
        finish_queue_if_not_batching(ctx.get_queue(), "adam_step");
        mark_device_result();
        moment1.mark_device_result();
        moment2.mark_device_result();
        return true;
    }
    catch (const std::exception& e)
    {
        NN_LOG_DEBUG(std::string("adam_step_inplace fast path failed, falling back: ") + e.what());
        return false;
    }
}

bool OpenCLTensorBackend::ensure_device_current(const char* what) const
{
    if (!m_backend || size() == 0) return false;
    if (!m_has_gpu_memory || !m_gpu_buffer)
    {
        // Lazily allocating the persistent buffer is logically non-mutating
        // (a cache of the same value), hence the const_cast — same rationale
        // as the mutable sync flags.
        auto* self = const_cast<OpenCLTensorBackend*>(this);
        self->try_allocate_gpu_buffer(size());
        if (!m_has_gpu_memory || !m_gpu_buffer) return false;
    }
    if (m_needs_sync_to_device)
    {
        const auto& ctx = opencl::OpenCLContext::instance();
        const std::size_t bytes = size() * sizeof(float);

        // A previous async upload may still be reading m_backend.
        wait_for_upload();

        if (opencl::OpenCLContext::is_batching())
        {
            // Non-blocking: a blocking write flushes the queue and would undo
            // the batching the caller set up. The event keeps m_backend pinned
            // until the DMA completes (see wait_for_upload).
            copy_host_to_device_async(ctx.get_queue(),
                m_gpu_buffer->buffer,
                m_backend->data_ptr(),
                bytes,
                what,
                &m_upload_event);
        }
        else
        {
            copy_host_to_device(
                ctx.get_queue(), m_gpu_buffer->buffer, m_backend->data_ptr(), bytes, what);
        }
        m_needs_sync_to_device = false;
    }
    // The device copy is now current: opt this tensor into the resident
    // lifecycle so a later device-side write (which sets
    // m_needs_sync_to_host) is lazily pulled back by sync_gpu_if_needed().
    // Without this, kernels writing into a tensor that entered non-resident
    // would leave host reads (at(), data_ptr()) permanently stale.
    const_cast<OpenCLTensorBackend*>(this)->m_gpu_resident = true;
    return true;
}

// Whether view/slice ops run as device kernels instead of host element loops.
//
// Which one wins depends entirely on the per-enqueue cost. With queue profiling
// enabled — the safe default on rusticl, see OpenCLContext — every enqueue costs
// ~95 us, and the device path *adds* enqueues (678k -> 817k on an Guayaquil run,
// ~+13 s) because it replaces a cheap host memcpy of a small slice with a kernel
// launch. With profiling off the device path wins, but that configuration
// corrupts memory on this driver.
//
// So: off by default, matching the default queue mode. NN_OPENCL_DEVICE_VIEW_OPS=1
// enables it — worth doing on any stack where enqueues are cheap.
bool device_view_ops_enabled()
{
    static const bool enabled = []
    {
        const char* env = std::getenv("NN_OPENCL_DEVICE_VIEW_OPS");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

bool OpenCLTensorBackend::launch_strided_copy(const OpenCLTensorBackend& src,
    const StridedRegion& src_region,
    OpenCLTensorBackend& dst,
    const StridedRegion& dst_region,
    Index ni,
    Index nj,
    const char* what)
{
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl(what)) return false;
    if (!device_view_ops_enabled()) return false;
    if (ni == 0 || nj == 0) return true; // nothing to copy

    try
    {
        if (!src.ensure_device_current(what)) return false;

        // The destination is written, not read, but it must still hold a valid
        // device buffer — and for a partial write (setBlock and friends) the
        // untouched elements have to be the current ones, so the existing
        // contents are uploaded first.
        if (!dst.ensure_device_current(what)) return false;

        const auto& ctx = opencl::OpenCLContext::instance();
        cl_kernel kernel = opencl::KernelManager::instance().get_kernel("strided_copy_2d_kernel");

        const cl_mem src_mem = src.m_gpu_buffer->buffer;
        const cl_mem dst_mem = dst.m_gpu_buffer->buffer;
        const cl_uint args[] = {
            static_cast<cl_uint>(src_region.base),
            static_cast<cl_uint>(src_region.stride_i),
            static_cast<cl_uint>(src_region.stride_j),
            static_cast<cl_uint>(dst_region.base),
            static_cast<cl_uint>(dst_region.stride_i),
            static_cast<cl_uint>(dst_region.stride_j),
            static_cast<cl_uint>(ni),
            static_cast<cl_uint>(nj),
        };

        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &src_mem), what);
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst_mem), what);
        for (cl_uint i = 0; i < 8; ++i)
        {
            check_cl_error(clSetKernelArg(kernel, 2 + i, sizeof(cl_uint), &args[i]), what);
        }

        const std::size_t global[2] = {ni, nj};
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
            what);
        finish_queue_if_not_batching(ctx.get_queue(), what);
        dst.mark_device_result();
        return true;
    }
    catch (const std::exception& e)
    {
        NN_LOG_DEBUG(
            std::string("strided copy fast path failed, falling back: ") + what + ": " + e.what());
        return false;
    }
}

bool OpenCLTensorBackend::launch_binary_resident(const char* kernel_name,
    const OpenCLTensorBackend& a,
    const OpenCLTensorBackend& b,
    OpenCLTensorBackend& out,
    const char* what)
{
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl(what)) return false;
    try
    {
        if (!a.ensure_device_current(what) || !b.ensure_device_current(what)) return false;
        if (!out.m_gpu_buffer)
        {
            out.try_allocate_gpu_buffer(out.size());
            if (!out.m_gpu_buffer) return false;
        }

        const auto& ctx = opencl::OpenCLContext::instance();
        cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
        const cl_mem a_mem = a.m_gpu_buffer->buffer;
        const cl_mem b_mem = b.m_gpu_buffer->buffer;
        const cl_mem out_mem = out.m_gpu_buffer->buffer;
        const cl_uint n_u32 = static_cast<cl_uint>(a.size());

        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), what);
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), what);
        check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), what);
        check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), what);

        const std::size_t local = 256;
        const std::size_t global = round_up(a.size(), local);
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
            what);
        finish_queue_if_not_batching(ctx.get_queue(), what);
        out.mark_device_result();
        return true;
    }
    catch (const std::exception& e)
    {
        NN_LOG_DEBUG(
            std::string("resident fast path failed, falling back: ") + what + ": " + e.what());
        return false;
    }
}

bool OpenCLTensorBackend::launch_unary_resident(const char* kernel_name,
    const OpenCLTensorBackend& a,
    OpenCLTensorBackend& out,
    const char* what)
{
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl(what)) return false;
    try
    {
        if (!a.ensure_device_current(what)) return false;
        if (!out.m_gpu_buffer)
        {
            out.try_allocate_gpu_buffer(out.size());
            if (!out.m_gpu_buffer) return false;
        }

        const auto& ctx = opencl::OpenCLContext::instance();
        cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
        const cl_mem a_mem = a.m_gpu_buffer->buffer;
        const cl_mem out_mem = out.m_gpu_buffer->buffer;
        const cl_uint n_u32 = static_cast<cl_uint>(a.size());

        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), what);
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), what);
        check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), what);

        const std::size_t local = 256;
        const std::size_t global = round_up(a.size(), local);
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
            what);
        finish_queue_if_not_batching(ctx.get_queue(), what);
        out.mark_device_result();
        return true;
    }
    catch (const std::exception& e)
    {
        NN_LOG_DEBUG(
            std::string("resident fast path failed, falling back: ") + what + ": " + e.what());
        return false;
    }
}

bool OpenCLTensorBackend::launch_unary_scalar_resident(const char* kernel_name,
    const OpenCLTensorBackend& a,
    float scalar,
    OpenCLTensorBackend& out,
    const char* what)
{
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl(what)) return false;
    try
    {
        if (!a.ensure_device_current(what)) return false;
        if (!out.m_gpu_buffer)
        {
            out.try_allocate_gpu_buffer(out.size());
            if (!out.m_gpu_buffer) return false;
        }

        const auto& ctx = opencl::OpenCLContext::instance();
        cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
        const cl_mem a_mem = a.m_gpu_buffer->buffer;
        const cl_mem out_mem = out.m_gpu_buffer->buffer;
        const cl_uint n_u32 = static_cast<cl_uint>(a.size());

        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), what);
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), what);
        check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &scalar), what);
        check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), what);

        const std::size_t local = 256;
        const std::size_t global = round_up(a.size(), local);
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
            what);
        finish_queue_if_not_batching(ctx.get_queue(), what);
        out.mark_device_result();
        return true;
    }
    catch (const std::exception& e)
    {
        NN_LOG_DEBUG(
            std::string("resident fast path failed, falling back: ") + what + ": " + e.what());
        return false;
    }
}

bool OpenCLTensorBackend::launch_inplace_binary_resident(
    const char* kernel_name, OpenCLTensorBackend& a, const OpenCLTensorBackend& b, const char* what)
{
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl(what)) return false;
    try
    {
        if (!a.ensure_device_current(what) || !b.ensure_device_current(what)) return false;

        const auto& ctx = opencl::OpenCLContext::instance();
        cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
        const cl_mem a_mem = a.m_gpu_buffer->buffer;
        const cl_mem b_mem = b.m_gpu_buffer->buffer;
        const cl_uint n_u32 = static_cast<cl_uint>(a.size());

        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), what);
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), what);
        check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), what);

        const std::size_t local = 256;
        const std::size_t global = round_up(a.size(), local);
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
            what);
        finish_queue_if_not_batching(ctx.get_queue(), what);
        a.mark_device_result();
        return true;
    }
    catch (const std::exception& e)
    {
        NN_LOG_DEBUG(
            std::string("resident fast path failed, falling back: ") + what + ": " + e.what());
        return false;
    }
}

bool OpenCLTensorBackend::launch_inplace_scalar_resident(
    const char* kernel_name, OpenCLTensorBackend& a, float scalar, const char* what)
{
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl(what)) return false;
    try
    {
        if (!a.ensure_device_current(what)) return false;

        const auto& ctx = opencl::OpenCLContext::instance();
        cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kernel_name);
        const cl_mem a_mem = a.m_gpu_buffer->buffer;
        const cl_uint n_u32 = static_cast<cl_uint>(a.size());

        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), what);
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(float), &scalar), what);
        check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), what);

        const std::size_t local = 256;
        const std::size_t global = round_up(a.size(), local);
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
            what);
        finish_queue_if_not_batching(ctx.get_queue(), what);
        a.mark_device_result();
        return true;
    }
    catch (const std::exception& e)
    {
        NN_LOG_DEBUG(
            std::string("resident fast path failed, falling back: ") + what + ": " + e.what());
        return false;
    }
}

void OpenCLTensorBackend::sync_gpu() const
{
    if (m_pending_events_count > 0)
    {
        const auto& ctx = opencl::OpenCLContext::instance();
        AsyncTransferManager::instance().wait_and_release_events(
            ctx.get_queue(), m_pending_events, static_cast<cl_uint>(m_pending_events_count));
        m_pending_events_count = 0;
    }

    if (m_gpu_resident && m_has_gpu_memory && m_needs_sync_to_host && m_gpu_buffer && m_backend)
    {
        const auto& ctx = opencl::OpenCLContext::instance();
        const std::size_t bytes = size() * sizeof(float);
        // The readback overwrites m_backend, which an in-flight upload may still
        // be reading from.
        wait_for_upload();
        copy_device_to_host(ctx.get_queue(),
            m_gpu_buffer->buffer,
            m_backend->mutable_data_ptr(),
            bytes,
            "sync_gpu");
        m_needs_sync_to_host = false;
    }
}

void OpenCLTensorBackend::sync_gpu_if_needed() const
{
    if (m_gpu_resident && m_has_gpu_memory && m_needs_sync_to_host)
    {
        sync_gpu();
    }
}

void OpenCLTensorBackend::try_allocate_gpu_buffer(Index size)
{
    if (size == 0) return;
    // cppcheck-suppress knownConditionTrueFalse
    if (!can_use_opencl("constructor")) return;

    try
    {
        const auto& ctx = opencl::OpenCLContext::instance();
        const std::size_t bytes = static_cast<std::size_t>(size) * sizeof(float);

        cl_int err = 0;
        cl_mem buffer = clCreateBuffer(ctx.get_context(), CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err == CL_SUCCESS && buffer)
        {
            m_gpu_buffer = std::make_unique<tensor::GPUBuffer>(buffer, bytes);
            m_has_gpu_memory = true;
            NN_LOG_DEBUG("OpenCLTensorBackend: persistent GPU buffer allocated");
        }
    }
    catch (const std::exception& e)
    {
        NN_LOG_DEBUG(std::string("GPU buffer allocation failed: ") + e.what());
    }
}
} // namespace nn
