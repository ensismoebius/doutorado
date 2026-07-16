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
                m_backend->data_ptr(),
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
                    m_backend->data_ptr(),
                    bytes,
                    "OpenCLTensorBackend copy assign");
                m_needs_sync_to_device = false;
            }
        }
    }
    return *this;
}

OpenCLTensorBackend::OpenCLTensorBackend(OpenCLTensorBackend&& other) noexcept = default;

OpenCLTensorBackend& OpenCLTensorBackend::operator=(OpenCLTensorBackend&& other) noexcept = default;

OpenCLTensorBackend::~OpenCLTensorBackend() = default;

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
    m_needs_sync_to_device = true;
    return m_backend->mutable_data_ptr();
}

const float* OpenCLTensorBackend::data_ptr() const
{
    sync_gpu();
    return m_backend->data_ptr();
}

OpenCLTensorBackend OpenCLTensorBackend::row(Index i) const
{
    sync_gpu();
    if (shape().size() != 2) throw std::invalid_argument("row requires rank-2 tensor");
    if (i >= rows()) throw std::out_of_range("row index out of range");

    OpenCLTensorBackend out(1, cols());
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
    sync_gpu();
    if (shape().size() != 2) throw std::invalid_argument("col requires rank-2 tensor");
    if (j >= cols()) throw std::out_of_range("col index out of range");

    OpenCLTensorBackend out(rows(), 1);
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
    sync_gpu();
    if (shape().size() != 2) throw std::invalid_argument("leftCols requires rank-2 tensor");
    if (n > cols()) throw std::out_of_range("leftCols exceeds tensor width");

    OpenCLTensorBackend out(rows(), n);
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
    sync_gpu();
    if (shape().size() != 2) throw std::invalid_argument("topRows requires rank-2 tensor");
    if (n > rows()) throw std::out_of_range("topRows exceeds tensor height");

    OpenCLTensorBackend out(n, cols());
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
    sync_gpu();
    block.sync_gpu();
    if (shape().size() != 2 || block.shape().size() != 2)
    {
        throw std::invalid_argument("setBlock requires rank-2 tensors");
    }
    if (row + block.rows() > rows() || col + block.cols() > cols())
    {
        throw std::invalid_argument("setBlock: block exceeds tensor bounds");
    }

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
    sync_gpu();
    const auto& s = shape();
    if (s.size() != 3) throw std::invalid_argument("slice_batch requires rank-3 tensor");
    if (b >= s[0]) throw std::out_of_range("slice_batch index out of range");

    OpenCLTensorBackend out(s[1], s[2]);
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
    sync_gpu();
    val.sync_gpu();
    const auto& s = shape();
    if (s.size() != 3) throw std::invalid_argument("set_batch_slice requires rank-3 tensor");
    if (b >= s[0]) throw std::out_of_range("set_batch_slice index out of range");
    if (val.rows() != s[1] || val.cols() != s[2])
        throw std::invalid_argument("set_batch_slice value shape mismatch");

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
    sync_gpu();
    const auto& s = shape();
    if (s.size() != 3) throw std::invalid_argument("slice_time requires rank-3 tensor");
    if (t >= s[1]) throw std::out_of_range("slice_time index out of range");

    OpenCLTensorBackend out(s[0], s[2]);
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
    sync_gpu();
    val.sync_gpu();
    const auto& s = shape();
    if (s.size() != 3) throw std::invalid_argument("set_time_slice requires rank-3 tensor");
    if (t >= s[1]) throw std::out_of_range("set_time_slice index out of range");
    if (val.rows() != s[0] || val.cols() != s[2])
        throw std::invalid_argument("set_time_slice value shape mismatch");

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
void OpenCLTensorBackend::add_inplace(const OpenCLTensorBackend& other)
{
    // Device-resident fast path (see add() for the pattern rationale).
    if (shape() == other.shape() &&
        launch_inplace_binary_resident("add_inplace_kernel", *this, other, "add_inplace"))
        return;

    sync_gpu();
    other.sync_gpu();
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("add_inplace", "OpenCL path requires matching tensor shapes");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("add_inplace"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);

            if (ensure_device_current("resident gate") &&
                other.ensure_device_current("resident gate"))
            {
                if (m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        m_gpu_buffer->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "add_inplace resident lhs");
                    m_needs_sync_to_device = false;
                }
                if (other.m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        other.m_gpu_buffer->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "add_inplace resident rhs");
                    other.m_needs_sync_to_device = false;
                }

                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("add_inplace_kernel");
                const cl_mem a_mem = m_gpu_buffer->buffer;
                const cl_mem b_mem = other.m_gpu_buffer->buffer;
                const cl_uint n_u32 = static_cast<cl_uint>(n);

                check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "add_inplace");
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "add_inplace");
                check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "add_inplace");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "add_inplace");
                finish_queue_if_not_batching(ctx.get_queue(), "add_inplace");

                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                if (a_buf && b_buf)
                {
                    cl_event a_evt = nullptr;
                    cl_event b_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "add_inplace",
                        &a_evt);
                    copy_host_to_device_async(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "add_inplace",
                        &b_evt);

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("add_inplace_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(
                        clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "add_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "add_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "add_inplace");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (a_evt && b_evt)
                    {
                        cl_event wait_events[2] = {a_evt, b_evt};
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           2,
                                           wait_events,
                                           &kernel_evt),
                            "add_inplace");
                    }
                    else if (a_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &a_evt,
                                           &kernel_evt),
                            "add_inplace");
                    }
                    else if (b_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &b_evt,
                                           &kernel_evt),
                            "add_inplace");
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
                            "add_inplace");
                    }

                    if (a_evt) clReleaseEvent(a_evt);
                    if (b_evt) clReleaseEvent(b_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->mutable_data_ptr(),
                        bytes,
                        "add_inplace",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("add_inplace_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "add_inplace");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "add_inplace");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "add_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "add_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "add_inplace");

            a_dev.copy_from_device(m_backend->mutable_data_ptr());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("add_inplace", e.what());
        }
    }
    throw_opencl_only_failure("add_inplace", "OpenCL runtime unavailable or tensor shape mismatch");
}

void OpenCLTensorBackend::subtract_inplace(const OpenCLTensorBackend& other)
{
    // Device-resident fast path (see add() for the pattern rationale).
    if (shape() == other.shape() &&
        launch_inplace_binary_resident("subtract_inplace_kernel", *this, other, "subtract_inplace"))
        return;

    sync_gpu();
    other.sync_gpu();
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once(
            "subtract_inplace", "OpenCL path requires matching tensor shapes");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("subtract_inplace"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);

            if (ensure_device_current("resident gate") &&
                other.ensure_device_current("resident gate"))
            {
                if (m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        m_gpu_buffer->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "subtract_inplace resident lhs");
                    m_needs_sync_to_device = false;
                }
                if (other.m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        other.m_gpu_buffer->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "subtract_inplace resident rhs");
                    other.m_needs_sync_to_device = false;
                }

                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("subtract_inplace_kernel");
                const cl_mem a_mem = m_gpu_buffer->buffer;
                const cl_mem b_mem = other.m_gpu_buffer->buffer;
                const cl_uint n_u32 = static_cast<cl_uint>(n);

                check_cl_error(
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "subtract_inplace");
                check_cl_error(
                    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "subtract_inplace");
                check_cl_error(
                    clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "subtract_inplace");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "subtract_inplace");
                finish_queue_if_not_batching(ctx.get_queue(), "subtract_inplace");

                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                if (a_buf && b_buf)
                {
                    cl_event a_evt = nullptr;
                    cl_event b_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "subtract_inplace",
                        &a_evt);
                    copy_host_to_device_async(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "subtract_inplace",
                        &b_evt);

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("subtract_inplace_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(
                        clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "subtract_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "subtract_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "subtract_inplace");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (a_evt && b_evt)
                    {
                        cl_event wait_events[2] = {a_evt, b_evt};
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           2,
                                           wait_events,
                                           &kernel_evt),
                            "subtract_inplace");
                    }
                    else if (a_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &a_evt,
                                           &kernel_evt),
                            "subtract_inplace");
                    }
                    else if (b_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &b_evt,
                                           &kernel_evt),
                            "subtract_inplace");
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
                            "subtract_inplace");
                    }

                    if (a_evt) clReleaseEvent(a_evt);
                    if (b_evt) clReleaseEvent(b_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->mutable_data_ptr(),
                        bytes,
                        "subtract_inplace",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("subtract_inplace_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "subtract_inplace");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "subtract_inplace");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "subtract_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "subtract_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "subtract_inplace");

            a_dev.copy_from_device(m_backend->mutable_data_ptr());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("subtract_inplace", e.what());
        }
    }
    throw_opencl_only_failure(
        "subtract_inplace", "OpenCL runtime unavailable or tensor shape mismatch");
}

void OpenCLTensorBackend::multiply_inplace(const OpenCLTensorBackend& other)
{
    // Device-resident fast path (see add() for the pattern rationale).
    if (shape() == other.shape() &&
        launch_inplace_binary_resident("multiply_inplace_kernel", *this, other, "multiply_inplace"))
        return;

    sync_gpu();
    other.sync_gpu();
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once(
            "multiply_inplace", "OpenCL path requires matching tensor shapes");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("multiply_inplace"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);

            if (ensure_device_current("resident gate") &&
                other.ensure_device_current("resident gate"))
            {
                if (m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        m_gpu_buffer->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "multiply_inplace resident lhs");
                    m_needs_sync_to_device = false;
                }
                if (other.m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        other.m_gpu_buffer->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "multiply_inplace resident rhs");
                    other.m_needs_sync_to_device = false;
                }

                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("multiply_inplace_kernel");
                const cl_mem a_mem = m_gpu_buffer->buffer;
                const cl_mem b_mem = other.m_gpu_buffer->buffer;
                const cl_uint n_u32 = static_cast<cl_uint>(n);

                check_cl_error(
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "multiply_inplace");
                check_cl_error(
                    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "multiply_inplace");
                check_cl_error(
                    clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "multiply_inplace");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "multiply_inplace");
                finish_queue_if_not_batching(ctx.get_queue(), "multiply_inplace");

                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                if (a_buf && b_buf)
                {
                    cl_event a_evt = nullptr;
                    cl_event b_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "multiply_inplace",
                        &a_evt);
                    copy_host_to_device_async(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "multiply_inplace",
                        &b_evt);

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("multiply_inplace_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(
                        clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "multiply_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "multiply_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "multiply_inplace");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (a_evt && b_evt)
                    {
                        cl_event wait_events[2] = {a_evt, b_evt};
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           2,
                                           wait_events,
                                           &kernel_evt),
                            "multiply_inplace");
                    }
                    else if (a_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &a_evt,
                                           &kernel_evt),
                            "multiply_inplace");
                    }
                    else if (b_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &b_evt,
                                           &kernel_evt),
                            "multiply_inplace");
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
                            "multiply_inplace");
                    }

                    if (a_evt) clReleaseEvent(a_evt);
                    if (b_evt) clReleaseEvent(b_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->mutable_data_ptr(),
                        bytes,
                        "multiply_inplace",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("multiply_inplace_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "multiply_inplace");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "multiply_inplace");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "multiply_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "multiply_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "multiply_inplace");

            a_dev.copy_from_device(m_backend->mutable_data_ptr());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("multiply_inplace", e.what());
        }
    }
    throw_opencl_only_failure(
        "multiply_inplace", "OpenCL runtime unavailable or tensor shape mismatch");
}

void OpenCLTensorBackend::divide_inplace(const OpenCLTensorBackend& other)
{
    // Device-resident fast path (see add() for the pattern rationale).
    if (shape() == other.shape() &&
        launch_inplace_binary_resident("divide_inplace_kernel", *this, other, "divide_inplace"))
        return;

    sync_gpu();
    other.sync_gpu();
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once(
            "divide_inplace", "OpenCL path requires matching tensor shapes");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("divide_inplace"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);

            if (ensure_device_current("resident gate") &&
                other.ensure_device_current("resident gate"))
            {
                if (m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        m_gpu_buffer->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "divide_inplace resident lhs");
                    m_needs_sync_to_device = false;
                }
                if (other.m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        other.m_gpu_buffer->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "divide_inplace resident rhs");
                    other.m_needs_sync_to_device = false;
                }

                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("divide_inplace_kernel");
                const cl_mem a_mem = m_gpu_buffer->buffer;
                const cl_mem b_mem = other.m_gpu_buffer->buffer;
                const cl_uint n_u32 = static_cast<cl_uint>(n);

                check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "divide_inplace");
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "divide_inplace");
                check_cl_error(
                    clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "divide_inplace");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "divide_inplace");
                finish_queue_if_not_batching(ctx.get_queue(), "divide_inplace");

                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                if (a_buf && b_buf)
                {
                    cl_event a_evt = nullptr;
                    cl_event b_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "divide_inplace",
                        &a_evt);
                    copy_host_to_device_async(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "divide_inplace",
                        &b_evt);

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("divide_inplace_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(
                        clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "divide_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "divide_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "divide_inplace");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (a_evt && b_evt)
                    {
                        cl_event wait_events[2] = {a_evt, b_evt};
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           2,
                                           wait_events,
                                           &kernel_evt),
                            "divide_inplace");
                    }
                    else if (a_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &a_evt,
                                           &kernel_evt),
                            "divide_inplace");
                    }
                    else if (b_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &b_evt,
                                           &kernel_evt),
                            "divide_inplace");
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
                            "divide_inplace");
                    }

                    if (a_evt) clReleaseEvent(a_evt);
                    if (b_evt) clReleaseEvent(b_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->mutable_data_ptr(),
                        bytes,
                        "divide_inplace",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("divide_inplace_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "divide_inplace");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "divide_inplace");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "divide_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "divide_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "divide_inplace");

            a_dev.copy_from_device(m_backend->mutable_data_ptr());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("divide_inplace", e.what());
        }
    }
    throw_opencl_only_failure(
        "divide_inplace", "OpenCL runtime unavailable or tensor shape mismatch");
}

void OpenCLTensorBackend::add_scalar_inplace(float val)
{
    // Device-resident fast path (see add() for the pattern rationale).
    if (launch_inplace_scalar_resident(
            "add_scalar_inplace_kernel", *this, val, "add_scalar_inplace"))
        return;

    sync_gpu();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("add_scalar_inplace"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);

            if (ensure_device_current("resident gate"))
            {
                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("add_scalar_inplace_kernel");
                const cl_mem data_mem = m_gpu_buffer->buffer;
                const cl_uint n_u32 = static_cast<cl_uint>(n);

                check_cl_error(
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "add_scalar_inplace");
                check_cl_error(
                    clSetKernelArg(kernel, 1, sizeof(float), &val), "add_scalar_inplace");
                check_cl_error(
                    clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "add_scalar_inplace");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "add_scalar_inplace");
                finish_queue_if_not_batching(ctx.get_queue(), "add_scalar_inplace");

                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto data_buf = pool->acquire(bytes);
                if (data_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "add_scalar_inplace",
                        &h2d_evt);

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("add_scalar_inplace_kernel");
                    const cl_mem data_mem = data_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(
                        clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "add_scalar_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 1, sizeof(float), &val), "add_scalar_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "add_scalar_inplace");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "add_scalar_inplace");
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
                            "add_scalar_inplace");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->mutable_data_ptr(),
                        bytes,
                        "add_scalar_inplace",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory data_dev(bytes);
            data_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("add_scalar_inplace_kernel");
            const cl_mem data_mem = data_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "add_scalar_inplace");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(float), &val), "add_scalar_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "add_scalar_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "add_scalar_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "add_scalar_inplace");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("add_scalar_inplace", e.what());
        }
    }
    throw_opencl_only_failure("add_scalar_inplace", "OpenCL runtime unavailable");
}

void OpenCLTensorBackend::multiply_scalar_inplace(float val)
{
    // Device-resident fast path (see add() for the pattern rationale).
    if (launch_inplace_scalar_resident(
            "multiply_scalar_inplace_kernel", *this, val, "multiply_scalar_inplace"))
        return;

    sync_gpu();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("multiply_scalar_inplace"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);

            if (ensure_device_current("resident gate"))
            {
                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("multiply_scalar_inplace_kernel");
                const cl_mem data_mem = m_gpu_buffer->buffer;
                const cl_uint n_u32 = static_cast<cl_uint>(n);

                check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem),
                    "multiply_scalar_inplace");
                check_cl_error(
                    clSetKernelArg(kernel, 1, sizeof(float), &val), "multiply_scalar_inplace");
                check_cl_error(
                    clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "multiply_scalar_inplace");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "multiply_scalar_inplace");
                finish_queue_if_not_batching(ctx.get_queue(), "multiply_scalar_inplace");

                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto data_buf = pool->acquire(bytes);
                if (data_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "multiply_scalar_inplace",
                        &h2d_evt);

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel(
                        "multiply_scalar_inplace_kernel");
                    const cl_mem data_mem = data_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem),
                        "multiply_scalar_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 1, sizeof(float), &val), "multiply_scalar_inplace");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32),
                        "multiply_scalar_inplace");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "multiply_scalar_inplace");
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
                            "multiply_scalar_inplace");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->mutable_data_ptr(),
                        bytes,
                        "multiply_scalar_inplace",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory data_dev(bytes);
            data_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("multiply_scalar_inplace_kernel");
            const cl_mem data_mem = data_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "multiply_scalar_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(float), &val), "multiply_scalar_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "multiply_scalar_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "multiply_scalar_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "multiply_scalar_inplace");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("multiply_scalar_inplace", e.what());
        }
    }
    throw_opencl_only_failure("multiply_scalar_inplace", "OpenCL runtime unavailable");
}

void OpenCLTensorBackend::divide_scalar_inplace(float val)
{
    // Device-resident fast path (see add() for the pattern rationale).
    if (launch_inplace_scalar_resident(
            "divide_scalar_inplace_kernel", *this, val, "divide_scalar_inplace"))
        return;

    sync_gpu();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("divide_scalar_inplace"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);

            if (ensure_device_current("resident gate"))
            {
                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("divide_scalar_inplace_kernel");
                const cl_mem data_mem = m_gpu_buffer->buffer;
                const cl_uint n_u32 = static_cast<cl_uint>(n);

                check_cl_error(
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "divide_scalar_inplace");
                check_cl_error(
                    clSetKernelArg(kernel, 1, sizeof(float), &val), "divide_scalar_inplace");
                check_cl_error(
                    clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "divide_scalar_inplace");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "divide_scalar_inplace");
                finish_queue_if_not_batching(ctx.get_queue(), "divide_scalar_inplace");

                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto data_buf = pool->acquire(bytes);
                if (data_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "divide_scalar_inplace",
                        &h2d_evt);

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel(
                        "divide_scalar_inplace_kernel");
                    const cl_mem data_mem = data_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem),
                        "divide_scalar_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 1, sizeof(float), &val), "divide_scalar_inplace");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32),
                        "divide_scalar_inplace");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "divide_scalar_inplace");
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
                            "divide_scalar_inplace");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->mutable_data_ptr(),
                        bytes,
                        "divide_scalar_inplace",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory data_dev(bytes);
            data_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("divide_scalar_inplace_kernel");
            const cl_mem data_mem = data_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "divide_scalar_inplace");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(float), &val), "divide_scalar_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "divide_scalar_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "divide_scalar_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "divide_scalar_inplace");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("divide_scalar_inplace", e.what());
        }
    }
    throw_opencl_only_failure("divide_scalar_inplace", "OpenCL runtime unavailable");
}

void OpenCLTensorBackend::sqrt_inplace()
{
    sync_gpu();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("sqrt_inplace"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);

            if (ensure_device_current("resident gate"))
            {
                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("sqrt_inplace_kernel");
                const cl_mem data_mem = m_gpu_buffer->buffer;
                const cl_uint n_u32 = static_cast<cl_uint>(n);

                check_cl_error(
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "sqrt_inplace");
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_uint), &n_u32), "sqrt_inplace");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "sqrt_inplace");
                finish_queue_if_not_batching(ctx.get_queue(), "sqrt_inplace");

                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto data_buf = pool->acquire(bytes);
                if (data_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "sqrt_inplace",
                        &h2d_evt);

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("sqrt_inplace_kernel");
                    const cl_mem data_mem = data_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(
                        clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "sqrt_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 1, sizeof(cl_uint), &n_u32), "sqrt_inplace");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "sqrt_inplace");
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
                            "sqrt_inplace");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->mutable_data_ptr(),
                        bytes,
                        "sqrt_inplace",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory data_dev(bytes);
            data_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("sqrt_inplace_kernel");
            const cl_mem data_mem = data_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "sqrt_inplace");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_uint), &n_u32), "sqrt_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "sqrt_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "sqrt_inplace");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("sqrt_inplace", e.what());
        }
    }
    throw_opencl_only_failure("sqrt_inplace", "OpenCL runtime unavailable");
}

void OpenCLTensorBackend::fill(float value)
{
    sync_gpu();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("fill"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);

            if (ensure_device_current("resident gate"))
            {
                cl_kernel kernel = opencl::KernelManager::instance().get_kernel("fill_kernel");
                const cl_mem data_mem = m_gpu_buffer->buffer;
                const cl_uint n_u32 = static_cast<cl_uint>(n);

                check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "fill");
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(float), &value), "fill");
                check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "fill");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "fill");
                finish_queue_if_not_batching(ctx.get_queue(), "fill");

                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto data_buf = pool->acquire(bytes);
                if (data_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "fill",
                        &h2d_evt);

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("fill_kernel");
                    const cl_mem data_mem = data_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "fill");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(float), &value), "fill");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "fill");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "fill");
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
                            "fill");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->mutable_data_ptr(),
                        bytes,
                        "fill",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory data_dev(bytes);
            data_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("fill_kernel");
            const cl_mem data_mem = data_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "fill");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(float), &value), "fill");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "fill");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "fill");
            finish_queue_if_not_batching(ctx.get_queue(), "fill");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("fill", e.what());
        }
    }
    throw_opencl_only_failure("fill", "OpenCL runtime unavailable");
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
    sync_gpu();
    row.sync_gpu();
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
                        m_backend->data_ptr(),
                        bytes,
                        "add_row_broadcast_inplace resident data");
                    m_needs_sync_to_device = false;
                }
                if (row.m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        row.m_gpu_buffer->buffer,
                        row.m_backend->data_ptr(),
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
                        m_backend->data_ptr(),
                        bytes,
                        "add_row_broadcast_inplace",
                        &data_evt);
                    copy_host_to_device_async(ctx.get_queue(),
                        row_buf->buffer,
                        row.m_backend->data_ptr(),
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
                        m_backend->mutable_data_ptr(),
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
            data_dev.copy_to_device(m_backend->data_ptr());
            row_dev.copy_to_device(row.m_backend->data_ptr());

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

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
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
    sync_gpu();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("square_inplace"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);

            if (ensure_device_current("resident gate"))
            {
                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("square_inplace_kernel");
                const cl_mem data_mem = m_gpu_buffer->buffer;
                const cl_uint n_u32 = static_cast<cl_uint>(n);

                check_cl_error(
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "square_inplace");
                check_cl_error(
                    clSetKernelArg(kernel, 1, sizeof(cl_uint), &n_u32), "square_inplace");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "square_inplace");
                finish_queue_if_not_batching(ctx.get_queue(), "square_inplace");

                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto data_buf = pool->acquire(bytes);
                if (data_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "square_inplace",
                        &h2d_evt);

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("square_inplace_kernel");
                    const cl_mem data_mem = data_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(
                        clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "square_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 1, sizeof(cl_uint), &n_u32), "square_inplace");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "square_inplace");
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
                            "square_inplace");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->mutable_data_ptr(),
                        bytes,
                        "square_inplace",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory data_dev(bytes);
            data_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("square_inplace_kernel");
            const cl_mem data_mem = data_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "square_inplace");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_uint), &n_u32), "square_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "square_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "square_inplace");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("square_inplace", e.what());
        }
    }
    throw_opencl_only_failure("square_inplace", "OpenCL runtime unavailable");
}

void OpenCLTensorBackend::add_col_vector_to_rows_inplace(const OpenCLTensorBackend& col_vector)
{
    if (!m_gpu_resident)
    {
        sync_gpu();
    }
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
                        m_backend->data_ptr(),
                        bytes,
                        "add_col_vector_to_rows_inplace resident data");
                    m_needs_sync_to_device = false;
                }
                if (col_vector.m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        col_vector.m_gpu_buffer->buffer,
                        col_vector.m_backend->data_ptr(),
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
                        m_backend->data_ptr(),
                        bytes,
                        "add_col_vector_to_rows_inplace",
                        &data_evt);
                    copy_host_to_device_async(ctx.get_queue(),
                        col_buf->buffer,
                        col_vector.m_backend->data_ptr(),
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
                        m_backend->mutable_data_ptr(),
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
            data_dev.copy_to_device(m_backend->data_ptr());
            col_dev.copy_to_device(col_vector.m_backend->data_ptr());

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

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
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
OpenCLTensorBackend OpenCLTensorBackend::exp() const
{
    // Device-resident fast path (see add() for the pattern rationale).
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_unary_resident("exp_kernel", *this, fast_out, "exp")) return fast_out;
    }

    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
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

            const auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = n * sizeof(float);

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();

            if (pool && ensure_device_current("resident gate"))
            {
                auto out_buf = pool->acquire(bytes);
                if (out_buf && m_gpu_buffer)
                {
                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("exp_kernel");
                    const cl_mem in_mem = m_gpu_buffer->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "exp");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "exp");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "exp");

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
                        "exp");
                    finish_queue_if_not_batching(ctx.get_queue(), "exp");

                    OpenCLHostStorage out(shape());
                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.m_has_gpu_memory = true;
                    t.m_gpu_buffer = std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
                    t.set_gpu_resident(true);
                    t.m_needs_sync_to_host = true;
                    t.m_needs_sync_to_device = false;
                    return t;
                }
            }

            sync_gpu();
            OpenCLHostStorage out(shape());

            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "exp",
                        &h2d_evt);

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("exp_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "exp");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "exp");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "exp");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "exp");
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
                            "exp");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "exp",
                        &d2h_evt);

                    if (kernel_evt) clReleaseEvent(kernel_evt);

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.record_pending_gpu_op(d2h_evt);
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
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(exp)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("exp", e.what());
        }
    }

    throw_opencl_only_failure("exp", "OpenCL runtime unavailable");
}

OpenCLTensorBackend OpenCLTensorBackend::sqrt() const
{
    // Device-resident fast path (see add() for the pattern rationale).
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_unary_resident("sqrt_kernel", *this, fast_out, "sqrt")) return fast_out;
    }

    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("sqrt"))
    {
        try
        {
            const auto n = size();
            if (n == 0)
            {
                OpenCLTensorBackend empty(*this);
                return empty;
            }

            const auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = n * sizeof(float);

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();

            if (pool && ensure_device_current("resident gate"))
            {
                auto out_buf = pool->acquire(bytes);
                if (out_buf && m_gpu_buffer)
                {
                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("sqrt_kernel");
                    const cl_mem in_mem = m_gpu_buffer->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "sqrt");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "sqrt");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "sqrt");

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
                        "sqrt");
                    finish_queue_if_not_batching(ctx.get_queue(), "sqrt");

                    OpenCLHostStorage out(shape());
                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.m_has_gpu_memory = true;
                    t.m_gpu_buffer = std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
                    t.set_gpu_resident(true);
                    t.m_needs_sync_to_host = true;
                    t.m_needs_sync_to_device = false;
                    return t;
                }
            }

            sync_gpu();
            OpenCLHostStorage out(shape());

            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "sqrt",
                        &h2d_evt);

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("sqrt_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "sqrt");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "sqrt");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "sqrt");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "sqrt");
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
                            "sqrt");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "sqrt",
                        &d2h_evt);

                    if (kernel_evt) clReleaseEvent(kernel_evt);

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.record_pending_gpu_op(d2h_evt);
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("sqrt_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "clSetKernelArg(sqrt, in)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "clSetKernelArg(sqrt, out)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "clSetKernelArg(sqrt, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(sqrt)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(sqrt)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("sqrt", e.what());
        }
    }

    throw_opencl_only_failure("sqrt", "OpenCL runtime unavailable");
}

OpenCLTensorBackend OpenCLTensorBackend::square() const
{
    // Device-resident fast path (see add() for the pattern rationale).
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_unary_resident("square_kernel", *this, fast_out, "square")) return fast_out;
    }

    sync_gpu();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("square"))
    {
        try
        {
            const auto n = size();
            if (n == 0)
            {
                OpenCLTensorBackend empty(*this);
                return empty;
            }

            const auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = n * sizeof(float);
            OpenCLHostStorage out(shape());

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "square",
                        &h2d_evt);

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("square_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "square");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "square");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "square");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "square");
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
                            "square");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "square",
                        &d2h_evt);

                    if (kernel_evt) clReleaseEvent(kernel_evt);

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.record_pending_gpu_op(d2h_evt);
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("square_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "clSetKernelArg(square, in)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "clSetKernelArg(square, out)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "clSetKernelArg(square, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(square)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(square)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("square", e.what());
        }
    }

    throw_opencl_only_failure("square", "OpenCL runtime unavailable");
}

OpenCLTensorBackend OpenCLTensorBackend::add(const OpenCLTensorBackend& other) const
{
    // Device-resident fast path: operands' device copies are used directly
    // (uploaded once into their persistent buffers when stale); the result
    // stays on the GPU and is synced to host lazily.
    if (shape() == other.shape())
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_binary_resident("add_kernel", *this, other, fast_out, "add")) return fast_out;
    }

    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();
    if (shape() != other.shape())
    {
        throw std::invalid_argument("add: tensor shapes must match");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("add"))
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
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(add)");

                    // GPU-resident mode: keep result on GPU
                    if (ensure_device_current("resident gate"))
                    {
                        OpenCLTensorBackend t;
                        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                        t.m_has_gpu_memory = true;
                        if (out_buf)
                        {
                            t.m_gpu_buffer =
                                std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
                        }
                        t.set_gpu_resident(true);
                        t.m_needs_sync_to_host = true;
                        return t;
                    }

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(add, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
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
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(add)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("add", e.what());
        }
    }

    throw_opencl_only_failure("add", "OpenCL runtime unavailable or tensor shape mismatch");
}

OpenCLTensorBackend OpenCLTensorBackend::subtract(const OpenCLTensorBackend& other) const
{
    // Device-resident fast path: operands' device copies are used directly
    // (uploaded once into their persistent buffers when stale); the result
    // stays on the GPU and is synced to host lazily.
    if (shape() == other.shape())
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_binary_resident("subtract_kernel", *this, other, fast_out, "subtract"))
            return fast_out;
    }

    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("subtract", "OpenCL path requires matching tensor shapes");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("subtract"))
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
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (a_buf && b_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(subtract, a)");
                    copy_host_to_device(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(subtract, b)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("subtract_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                        "clSetKernelArg(subtract, a)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                        "clSetKernelArg(subtract, b)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(subtract, out)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(subtract, size)");

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
                        "clEnqueueNDRangeKernel(subtract)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(subtract)");

                    // GPU-resident mode: keep result on GPU
                    if (ensure_device_current("resident gate"))
                    {
                        OpenCLTensorBackend t;
                        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                        t.m_has_gpu_memory = true;
                        if (out_buf)
                        {
                            t.m_gpu_buffer =
                                std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
                        }
                        t.set_gpu_resident(true);
                        t.m_needs_sync_to_host = true;
                        return t;
                    }

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(subtract, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("subtract_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(subtract, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(subtract, b)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(subtract, out)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(subtract, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(subtract)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(subtract)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("subtract", e.what());
        }
    }

    throw_opencl_only_failure("subtract", "OpenCL runtime unavailable or tensor shape mismatch");
}

OpenCLTensorBackend OpenCLTensorBackend::multiply(const OpenCLTensorBackend& other) const
{
    // Device-resident fast path: operands' device copies are used directly
    // (uploaded once into their persistent buffers when stale); the result
    // stays on the GPU and is synced to host lazily.
    if (shape() == other.shape())
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_binary_resident("multiply_kernel", *this, other, fast_out, "multiply"))
            return fast_out;
    }

    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();
    if (shape() != other.shape())
    {
        throw std::invalid_argument("multiply: tensor shapes must match");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("multiply"))
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
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(multiply)");

                    // GPU-resident mode: keep result on GPU
                    if (ensure_device_current("resident gate"))
                    {
                        OpenCLTensorBackend t;
                        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                        t.m_has_gpu_memory = true;
                        if (out_buf)
                        {
                            t.m_gpu_buffer =
                                std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
                        }
                        t.set_gpu_resident(true);
                        t.m_needs_sync_to_host = true;
                        return t;
                    }

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(multiply, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
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
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(multiply)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("multiply", e.what());
        }
    }

    throw_opencl_only_failure("multiply", "OpenCL runtime unavailable or tensor shape mismatch");
}

OpenCLTensorBackend OpenCLTensorBackend::divide(const OpenCLTensorBackend& other) const
{
    // Device-resident fast path: operands' device copies are used directly
    // (uploaded once into their persistent buffers when stale); the result
    // stays on the GPU and is synced to host lazily.
    if (shape() == other.shape())
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_binary_resident("divide_kernel", *this, other, fast_out, "divide"))
            return fast_out;
    }

    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("divide", "OpenCL path requires matching tensor shapes");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("divide"))
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
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (a_buf && b_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(divide, a)");
                    copy_host_to_device(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(divide, b)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("divide_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                        "clSetKernelArg(divide, a)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                        "clSetKernelArg(divide, b)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(divide, out)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(divide, size)");

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
                        "clEnqueueNDRangeKernel(divide)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(divide)");

                    // GPU-resident mode: keep result on GPU
                    if (ensure_device_current("resident gate"))
                    {
                        OpenCLTensorBackend t;
                        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                        t.m_has_gpu_memory = true;
                        if (out_buf)
                        {
                            t.m_gpu_buffer =
                                std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
                        }
                        t.set_gpu_resident(true);
                        t.m_needs_sync_to_host = true;
                        return t;
                    }

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(divide, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("divide_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(divide, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(divide, b)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), "clSetKernelArg(divide, out)");
            check_cl_error(
                clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "clSetKernelArg(divide, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(divide)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(divide)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("divide", e.what());
        }
    }

    throw_opencl_only_failure("divide", "OpenCL runtime unavailable or tensor shape mismatch");
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
                        m_backend->data_ptr(),
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
            input_dev.copy_to_device(m_backend->data_ptr());

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
OpenCLTensorBackend OpenCLTensorBackend::rowwise_sum() const
{
    if (!m_gpu_resident)
    {
        sync_gpu_if_needed();
    }
    if (shape().size() != 2)
    {
        warn_opencl_cpu_fallback_once("rowwise_sum", "OpenCL path requires rank-2 tensors");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("rowwise_sum"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const Index num_rows = rows();
            const Index num_cols = cols();
            const std::size_t input_bytes = num_rows * num_cols * sizeof(float);
            const std::size_t output_bytes = num_rows * sizeof(float);

            OpenCLHostStorage out(num_rows, 1);

            if (ensure_device_current("resident gate"))
            {
                if (m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        m_gpu_buffer->buffer,
                        m_backend->data_ptr(),
                        input_bytes,
                        "clEnqueueWriteBuffer(rowwise_sum, input resident)");
                    m_needs_sync_to_device = false;
                }

                OpenCLTensorBackend t(num_rows, 1);
                t.set_gpu_resident(true);

                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("rowwise_sum_kernel");
                const cl_mem in_mem = m_gpu_buffer->buffer;
                const cl_mem out_mem = t.m_gpu_buffer->buffer;
                const cl_uint rows_u32 = static_cast<cl_uint>(num_rows);
                const cl_uint cols_u32 = static_cast<cl_uint>(num_cols);

                check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                    "clSetKernelArg(rowwise_sum, input resident)");
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                    "clSetKernelArg(rowwise_sum, output resident)");
                check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &rows_u32),
                    "clSetKernelArg(rowwise_sum, rows resident)");
                check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &cols_u32),
                    "clSetKernelArg(rowwise_sum, cols resident)");

                const std::size_t global = static_cast<std::size_t>(num_rows);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel(rowwise_sum resident)");
                t.m_needs_sync_to_host = true;
                t.m_needs_sync_to_device = false;
                return t;
            }

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
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(rowwise_sum)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        output_bytes,
                        "clEnqueueReadBuffer(rowwise_sum, output)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
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
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(rowwise_sum)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("rowwise_sum", e.what());
        }
    }

    throw_opencl_only_failure(
        "rowwise_sum", "OpenCL runtime unavailable or tensor rank is invalid");
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

float OpenCLTensorBackend::mean_squared_error(const OpenCLTensorBackend& target) const
{
    sync_gpu_if_needed();
    target.sync_gpu_if_needed();
    if (shape() != target.shape())
    {
        throw std::invalid_argument("mean_squared_error requires equal shapes");
    }

    const auto n = size();
    if (n == 0) return 0.0F;

    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("mse"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = n * sizeof(float);
            const std::size_t local = 256;
            const std::size_t global = round_up(n, local);
            const std::size_t num_groups = global / local;
            const std::size_t partial_bytes = num_groups * sizeof(float);

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();

            if (pool && ensure_device_current("resident gate") &&
                target.ensure_device_current("resident gate"))
            {
                auto partial_buf = pool->acquire(partial_bytes);
                if (partial_buf)
                {
                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("mse_kernel");
                    const cl_mem in_mem = m_gpu_buffer->buffer;
                    const cl_mem tgt_mem = target.m_gpu_buffer->buffer;
                    const cl_mem out_mem = partial_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "mse");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &tgt_mem), "mse");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), "mse");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "mse");

                    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                       kernel,
                                       1,
                                       nullptr,
                                       &global,
                                       &local,
                                       0,
                                       nullptr,
                                       nullptr),
                        "mse");

                    std::vector<float> partials(num_groups);
                    copy_device_to_host(ctx.get_queue(),
                        partial_buf->buffer,
                        partials.data(),
                        partial_bytes,
                        "mse");

                    float sq = 0.0F;
                    for (std::size_t i = 0; i < num_groups; ++i)
                    {
                        sq += partials[i];
                    }
                    return sq / static_cast<float>(n);
                }
            }

            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto target_buf = pool->acquire(bytes);
                auto partial_buf = pool->acquire(partial_bytes);
                if (input_buf && target_buf && partial_buf)
                {
                    cl_event h2d_evt[2] = {nullptr, nullptr};
                    copy_host_to_device_async(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "mse_input",
                        &h2d_evt[0]);
                    copy_host_to_device_async(ctx.get_queue(),
                        target_buf->buffer,
                        target.m_backend->data_ptr(),
                        bytes,
                        "mse_target",
                        &h2d_evt[1]);

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("mse_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem tgt_mem = target_buf->buffer;
                    const cl_mem out_mem = partial_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "mse");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &tgt_mem), "mse");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), "mse");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "mse");

                    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                       kernel,
                                       1,
                                       nullptr,
                                       &global,
                                       &local,
                                       2,
                                       h2d_evt,
                                       nullptr),
                        "mse");

                    if (h2d_evt[0]) clReleaseEvent(h2d_evt[0]);
                    if (h2d_evt[1]) clReleaseEvent(h2d_evt[1]);

                    std::vector<float> partials(num_groups);
                    copy_device_to_host(ctx.get_queue(),
                        partial_buf->buffer,
                        partials.data(),
                        partial_bytes,
                        "mse_out");

                    float sq = 0.0F;
                    for (std::size_t i = 0; i < num_groups; ++i)
                    {
                        sq += partials[i];
                    }
                    return sq / static_cast<float>(n);
                }
            }

            sync_gpu();
            target.sync_gpu();

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory target_dev(bytes);
            opencl::DeviceMemory partial_dev(partial_bytes);
            input_dev.copy_to_device(m_backend->data_ptr());
            target_dev.copy_to_device(target.m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("mse_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem tgt_mem = target_dev.get_device_buffer();
            const cl_mem out_mem = partial_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "clSetKernelArg(mse, in)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &tgt_mem), "clSetKernelArg(mse, target)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), "clSetKernelArg(mse, out)");
            check_cl_error(
                clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "clSetKernelArg(mse, size)");

            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(mse)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(mse)");

            std::vector<float> partials(num_groups);
            partial_dev.copy_from_device(partials.data());

            float sq = 0.0F;
            for (std::size_t i = 0; i < num_groups; ++i)
            {
                sq += partials[i];
            }
            return sq / static_cast<float>(n);
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("mse", e.what());
        }
    }

    throw_opencl_only_failure("mse", "OpenCL runtime unavailable");
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

float OpenCLTensorBackend::sum() const
{
    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("sum"))
    {
        try
        {
            const auto n = size();
            if (n == 0) return 0.0F;

            const auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = n * sizeof(float);
            const std::size_t local = 256;
            const std::size_t global = round_up(n, local);
            const std::size_t num_groups = global / local;
            const std::size_t partial_bytes = num_groups * sizeof(float);

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();

            if (pool && ensure_device_current("resident gate"))
            {
                auto partial_buf = pool->acquire(partial_bytes);
                if (partial_buf)
                {
                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("sum_kernel");
                    const cl_mem in_mem = m_gpu_buffer->buffer;
                    const cl_mem out_mem = partial_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "sum");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "sum");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "sum");

                    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                       kernel,
                                       1,
                                       nullptr,
                                       &global,
                                       &local,
                                       0,
                                       nullptr,
                                       nullptr),
                        "sum");

                    std::vector<float> partials(num_groups);
                    copy_device_to_host(ctx.get_queue(),
                        partial_buf->buffer,
                        partials.data(),
                        partial_bytes,
                        "sum");

                    float result = 0.0F;
                    for (std::size_t i = 0; i < num_groups; ++i)
                    {
                        result += partials[i];
                    }
                    return result;
                }
            }

            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto partial_buf = pool->acquire(partial_bytes);
                if (input_buf && partial_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "sum",
                        &h2d_evt);

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("sum_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = partial_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "sum");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "sum");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "sum");

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "sum");
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
                            "sum");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    std::vector<float> partials(num_groups);
                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        partial_buf->buffer,
                        partials.data(),
                        partial_bytes,
                        "sum",
                        &d2h_evt);

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    if (d2h_evt) clWaitForEvents(1, &d2h_evt);

                    float result = 0.0F;
                    for (std::size_t i = 0; i < num_groups; ++i)
                    {
                        result += partials[i];
                    }
                    return result;
                }
            }

            sync_gpu();

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory partial_dev(partial_bytes);
            input_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("sum_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = partial_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "clSetKernelArg(sum, in)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "clSetKernelArg(sum, out)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "clSetKernelArg(sum, size)");

            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(sum)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(sum)");

            std::vector<float> partials(num_groups);
            partial_dev.copy_from_device(partials.data());

            float result = 0.0F;
            for (std::size_t i = 0; i < num_groups; ++i)
            {
                result += partials[i];
            }
            return result;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("sum", e.what());
        }
    }

    throw_opencl_only_failure("sum", "OpenCL runtime unavailable");
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
    // Device-resident fast path (see add() for the pattern rationale).
    {
        OpenCLTensorBackend fast_out(shape());
        if (launch_unary_resident("abs_kernel", *this, fast_out, "abs")) return fast_out;
    }

    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("abs"))
    {
        try
        {
            const auto n = size();
            if (n == 0)
            {
                OpenCLTensorBackend empty(*this);
                return empty;
            }

            const auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = n * sizeof(float);

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();

            if (pool && ensure_device_current("resident gate"))
            {
                auto out_buf = pool->acquire(bytes);
                if (out_buf && m_gpu_buffer)
                {
                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("abs_kernel");
                    const cl_mem in_mem = m_gpu_buffer->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "abs");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "abs");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "abs");

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
                        "abs");
                    finish_queue_if_not_batching(ctx.get_queue(), "abs");

                    OpenCLHostStorage out(shape());
                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.m_has_gpu_memory = true;
                    t.m_gpu_buffer = std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
                    t.set_gpu_resident(true);
                    t.m_needs_sync_to_host = true;
                    t.m_needs_sync_to_device = false;
                    return t;
                }
            }

            sync_gpu();
            OpenCLHostStorage out(shape());

            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "abs",
                        &h2d_evt);

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("abs_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "abs");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "abs");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "abs");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "abs");
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
                            "abs");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "abs",
                        &d2h_evt);

                    if (kernel_evt) clReleaseEvent(kernel_evt);

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.record_pending_gpu_op(d2h_evt);
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("abs_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "clSetKernelArg(abs, in)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "clSetKernelArg(abs, out)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "clSetKernelArg(abs, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(abs)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(abs)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("abs", e.what());
        }
    }

    throw_opencl_only_failure("abs", "OpenCL runtime unavailable");
}

OpenCLTensorBackend OpenCLTensorBackend::relu() const
{
    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("relu"))
    {
        try
        {
            const auto n = size();
            if (n == 0)
            {
                OpenCLTensorBackend empty(*this);
                return empty;
            }

            const auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = n * sizeof(float);

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();

            if (pool && ensure_device_current("resident gate"))
            {
                auto out_buf = pool->acquire(bytes);
                if (out_buf && m_gpu_buffer)
                {
                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("relu_kernel");
                    const cl_mem in_mem = m_gpu_buffer->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "relu");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "relu");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "relu");

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
                        "relu");
                    finish_queue_if_not_batching(ctx.get_queue(), "relu");

                    OpenCLHostStorage out(shape());
                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.m_has_gpu_memory = true;
                    t.m_gpu_buffer = std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
                    t.set_gpu_resident(true);
                    t.m_needs_sync_to_host = true;
                    t.m_needs_sync_to_device = false;
                    return t;
                }
            }

            sync_gpu();
            OpenCLHostStorage out(shape());

            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "relu",
                        &h2d_evt);

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("relu_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "relu");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "relu");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "relu");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "relu");
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
                            "relu");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "relu",
                        &d2h_evt);

                    if (kernel_evt) clReleaseEvent(kernel_evt);

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.record_pending_gpu_op(d2h_evt);
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("relu_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "clSetKernelArg(relu, in)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "clSetKernelArg(relu, out)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_uint), &n_u32), "clSetKernelArg(relu, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(relu)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(relu)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("relu", e.what());
        }
    }

    throw_opencl_only_failure("relu", "OpenCL runtime unavailable");
}

OpenCLTensorBackend OpenCLTensorBackend::leaky_relu(float alpha) const
{
    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("leaky_relu"))
    {
        try
        {
            const auto n = size();
            if (n == 0)
            {
                OpenCLTensorBackend empty(*this);
                return empty;
            }

            const auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = n * sizeof(float);

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();

            if (pool && ensure_device_current("resident gate"))
            {
                auto out_buf = pool->acquire(bytes);
                if (out_buf && m_gpu_buffer)
                {
                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("leaky_relu_kernel");
                    const cl_mem in_mem = m_gpu_buffer->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(
                        clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "leaky_relu");
                    check_cl_error(
                        clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "leaky_relu");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &alpha), "leaky_relu");
                    check_cl_error(
                        clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "leaky_relu");

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
                        "leaky_relu");
                    finish_queue_if_not_batching(ctx.get_queue(), "leaky_relu");

                    OpenCLHostStorage out(shape());
                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.m_has_gpu_memory = true;
                    t.m_gpu_buffer = std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
                    t.set_gpu_resident(true);
                    t.m_needs_sync_to_host = true;
                    t.m_needs_sync_to_device = false;
                    return t;
                }
            }

            sync_gpu();
            OpenCLHostStorage out(shape());

            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "leaky_relu",
                        &h2d_evt);

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("leaky_relu_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(
                        clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "leaky_relu");
                    check_cl_error(
                        clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "leaky_relu");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &alpha), "leaky_relu");
                    check_cl_error(
                        clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "leaky_relu");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "leaky_relu");
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
                            "leaky_relu");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "leaky_relu",
                        &d2h_evt);

                    if (kernel_evt) clReleaseEvent(kernel_evt);

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.record_pending_gpu_op(d2h_evt);
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("leaky_relu_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem),
                "clSetKernelArg(leaky_relu, in)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(leaky_relu, out)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &alpha),
                "clSetKernelArg(leaky_relu, alpha)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(leaky_relu, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(leaky_relu)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(leaky_relu)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("leaky_relu", e.what());
        }
    }

    throw_opencl_only_failure("leaky_relu", "OpenCL runtime unavailable");
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

    const auto n = size();
    if (n == 0)
    {
        return;
    }

    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("lif_step_inplace"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = n * sizeof(float);

            opencl::DeviceMemory v_mem_dev(bytes);
            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory output_dev(bytes);
            std::unique_ptr<opencl::DeviceMemory> adapt_dev;

            v_mem_dev.copy_to_device(m_backend->data_ptr());
            input_dev.copy_to_device(input.m_backend->data_ptr());
            if (use_adaptation)
            {
                adapt_dev = std::make_unique<opencl::DeviceMemory>(bytes);
                adapt_dev->copy_to_device(adapt_a->m_backend->data_ptr());
            }

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("lif_step_kernel");
            const cl_mem v_mem = v_mem_dev.get_device_buffer();
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = output_dev.get_device_buffer();
            cl_mem adapt_mem = nullptr;
            if (adapt_dev)
            {
                adapt_mem = adapt_dev->get_device_buffer();
            }

            const cl_int reset_zero_i32 = reset_zero ? 1 : 0;
            const cl_int use_adaptation_i32 = use_adaptation ? 1 : 0;
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &v_mem), "lif_step_inplace");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &in_mem), "lif_step_inplace");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem), "lif_step_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 3, sizeof(cl_mem), &adapt_mem), "lif_step_inplace");
            check_cl_error(clSetKernelArg(kernel, 4, sizeof(float), &beta), "lif_step_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 5, sizeof(float), &threshold), "lif_step_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 6, sizeof(float), &reset_potential), "lif_step_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 7, sizeof(cl_int), &reset_zero_i32), "lif_step_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 8, sizeof(float), &adapt_decay), "lif_step_inplace");
            check_cl_error(
                clSetKernelArg(kernel, 9, sizeof(float), &adapt_coupling), "lif_step_inplace");
            check_cl_error(clSetKernelArg(kernel, 10, sizeof(cl_int), &use_adaptation_i32),
                "lif_step_inplace");
            check_cl_error(clSetKernelArg(kernel, 11, sizeof(cl_uint), &n_u32), "lif_step_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "lif_step_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "lif_step_inplace");

            v_mem_dev.copy_from_device(m_backend->mutable_data_ptr());
            mark_host_dirty();
            output_dev.copy_from_device(output.m_backend->mutable_data_ptr());
            output.mark_host_dirty();
            if (use_adaptation)
            {
                adapt_dev->copy_from_device(adapt_a->m_backend->mutable_data_ptr());
                adapt_a->mark_host_dirty();
            }
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("lif_step_inplace", e.what());
        }
    }

    throw_opencl_only_failure("lif_step_inplace", "OpenCL runtime unavailable");
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
            v_pre_dev.copy_to_device(m_backend->data_ptr());

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
    sync_gpu_if_needed();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("clamp"))
    {
        try
        {
            const auto n = size();
            if (n == 0)
            {
                OpenCLTensorBackend empty(*this);
                return empty;
            }

            const auto& ctx = opencl::OpenCLContext::instance();
            const std::size_t bytes = n * sizeof(float);

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();

            if (pool && ensure_device_current("resident gate"))
            {
                auto out_buf = pool->acquire(bytes);
                if (out_buf && m_gpu_buffer)
                {
                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("clamp_kernel");
                    const cl_mem in_mem = m_gpu_buffer->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "clamp");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "clamp");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &min_val), "clamp");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(float), &max_val), "clamp");
                    check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32), "clamp");

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
                        "clamp");
                    finish_queue_if_not_batching(ctx.get_queue(), "clamp");

                    OpenCLHostStorage out(shape());
                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.m_has_gpu_memory = true;
                    t.m_gpu_buffer = std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
                    t.set_gpu_resident(true);
                    t.m_needs_sync_to_host = true;
                    t.m_needs_sync_to_device = false;
                    return t;
                }
            }

            sync_gpu();
            OpenCLHostStorage out(shape());

            if (pool)
            {
                auto input_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (input_buf && out_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        input_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clamp",
                        &h2d_evt);

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel("clamp_kernel");
                    const cl_mem in_mem = input_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "clamp");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "clamp");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &min_val), "clamp");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(float), &max_val), "clamp");
                    check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32), "clamp");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "clamp");
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
                            "clamp");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clamp",
                        &d2h_evt);

                    if (kernel_evt) clReleaseEvent(kernel_evt);

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    t.record_pending_gpu_op(d2h_evt);
                    return t;
                }
            }

            opencl::DeviceMemory input_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            input_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("clamp_kernel");
            const cl_mem in_mem = input_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_mem), "clSetKernelArg(clamp, in)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &out_mem), "clSetKernelArg(clamp, out)");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(float), &min_val), "clSetKernelArg(clamp, min)");
            check_cl_error(
                clSetKernelArg(kernel, 3, sizeof(float), &max_val), "clSetKernelArg(clamp, max)");
            check_cl_error(
                clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32), "clSetKernelArg(clamp, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(clamp)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(clamp)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("clamp", e.what());
        }
    }

    throw_opencl_only_failure("clamp", "OpenCL runtime unavailable");
}

void OpenCLTensorBackend::clamp_inplace(float min_val, float max_val)
{
    sync_gpu();
    // cppcheck-suppress knownConditionTrueFalse
    if (can_use_opencl("clamp"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            if (n == 0) return;
            const std::size_t bytes = n * sizeof(float);

            if (ensure_device_current("resident gate"))
            {
                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("clamp_inplace_kernel");
                const cl_mem data_mem = m_gpu_buffer->buffer;
                const cl_uint n_u32 = static_cast<cl_uint>(n);

                check_cl_error(
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "clamp_inplace");
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(float), &min_val), "clamp_inplace");
                check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &max_val), "clamp_inplace");
                check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "clamp_inplace");

                const std::size_t local = 256;
                std::size_t global = round_up(n, local);
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                    "clamp_inplace");
                finish_queue_if_not_batching(ctx.get_queue(), "clamp_inplace");

                m_needs_sync_to_host = true;
                m_needs_sync_to_device = false;
                return;
            }

            tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
            if (pool)
            {
                auto data_buf = pool->acquire(bytes);
                if (data_buf)
                {
                    cl_event h2d_evt = nullptr;
                    copy_host_to_device_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clamp_inplace",
                        &h2d_evt);

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("clamp_inplace_kernel");
                    const cl_mem data_mem = data_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(
                        clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "clamp_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 1, sizeof(float), &min_val), "clamp_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 2, sizeof(float), &max_val), "clamp_inplace");
                    check_cl_error(
                        clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "clamp_inplace");

                    const std::size_t local = 256;
                    std::size_t global = round_up(n, local);

                    cl_event kernel_evt = nullptr;
                    if (h2d_evt)
                    {
                        check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                           kernel,
                                           1,
                                           nullptr,
                                           &global,
                                           &local,
                                           1,
                                           &h2d_evt,
                                           &kernel_evt),
                            "clamp_inplace");
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
                            "clamp_inplace");
                    }

                    if (h2d_evt) clReleaseEvent(h2d_evt);

                    cl_event d2h_evt = nullptr;
                    copy_device_to_host_async(ctx.get_queue(),
                        data_buf->buffer,
                        m_backend->mutable_data_ptr(),
                        bytes,
                        "clamp_inplace",
                        &d2h_evt);
                    mark_host_dirty();

                    if (kernel_evt) clReleaseEvent(kernel_evt);
                    record_pending_gpu_op(d2h_evt);
                    return;
                }
            }

            opencl::DeviceMemory data_dev(bytes);
            data_dev.copy_to_device(m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("clamp_inplace_kernel");
            const cl_mem data_mem = data_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &data_mem), "clamp_inplace");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(float), &min_val), "clamp_inplace");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(float), &max_val), "clamp_inplace");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32), "clamp_inplace");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clamp_inplace");
            finish_queue_if_not_batching(ctx.get_queue(), "clamp_inplace");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            mark_host_dirty();
            return;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("clamp_inplace", e.what());
        }
    }
    throw_opencl_only_failure("clamp_inplace", "OpenCL runtime unavailable");
}

// Linear algebra
OpenCLTensorBackend OpenCLTensorBackend::matmul(const OpenCLTensorBackend& other) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();
    if (shape().size() != 2 || other.shape().size() != 2)
    {
        throw std::invalid_argument("matmul: both tensors must be rank-2");
    }
    if (cols() != other.rows())
    {
        throw std::invalid_argument("matmul: lhs.cols() must equal rhs.rows()");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("matmul"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const Index m = rows();
            const Index k = cols();
            const Index n = other.cols();

            const std::size_t a_bytes = m * k * sizeof(float);
            const std::size_t b_bytes = k * n * sizeof(float);
            const std::size_t c_bytes = m * n * sizeof(float);
            OpenCLHostStorage out(m, n);

            if (ensure_device_current("resident gate") &&
                other.ensure_device_current("resident gate"))
            {
                if (m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        m_gpu_buffer->buffer,
                        m_backend->data_ptr(),
                        a_bytes,
                        "clEnqueueWriteBuffer(matmul, a resident)");
                    m_needs_sync_to_device = false;
                }
                if (other.m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        other.m_gpu_buffer->buffer,
                        other.m_backend->data_ptr(),
                        b_bytes,
                        "clEnqueueWriteBuffer(matmul, b resident)");
                    other.m_needs_sync_to_device = false;
                }

                OpenCLTensorBackend t(m, n);
                t.set_gpu_resident(true);

                cl_kernel kernel = opencl::KernelManager::instance().get_kernel("matmul_kernel");
                const cl_mem a_mem = m_gpu_buffer->buffer;
                const cl_mem b_mem = other.m_gpu_buffer->buffer;
                const cl_mem c_mem = t.m_gpu_buffer->buffer;
                const cl_uint m_u32 = static_cast<cl_uint>(m);
                const cl_uint n_u32 = static_cast<cl_uint>(n);
                const cl_uint k_u32 = static_cast<cl_uint>(k);

                check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                    "clSetKernelArg(matmul, a resident)");
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                    "clSetKernelArg(matmul, b resident)");
                check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem),
                    "clSetKernelArg(matmul, c resident)");
                check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &m_u32),
                    "clSetKernelArg(matmul, m resident)");
                check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32),
                    "clSetKernelArg(matmul, n resident)");
                check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &k_u32),
                    "clSetKernelArg(matmul, k resident)");

                const std::size_t global[2] = {m, n};
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel(matmul resident)");
                finish_queue_if_not_batching(ctx.get_queue(), "clFinish(matmul resident)");
                t.m_needs_sync_to_host = true;
                t.m_needs_sync_to_device = false;
                return t;
            }

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
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(matmul)");

                    copy_device_to_host(ctx.get_queue(),
                        c_buf->buffer,
                        out.mutable_data_ptr(),
                        c_bytes,
                        "clEnqueueReadBuffer(matmul, c)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
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
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(matmul)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("matmul", e.what());
        }
    }

    throw_opencl_only_failure(
        "matmul", "OpenCL runtime unavailable or matrix dimensions are invalid");
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed(const OpenCLTensorBackend& other) const
{
    if (!m_gpu_resident)
    {
        sync_gpu_if_needed();
    }
    if (!other.m_gpu_resident)
    {
        other.sync_gpu_if_needed();
    }
    if (shape().size() != 2 || other.shape().size() != 2 || cols() != other.cols())
    {
        throw_opencl_only_failure(
            "matmul_transposed", "OpenCL runtime unavailable or matrix dimensions are invalid");
    }
    else if (can_use_opencl("matmul_transposed"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const Index m = rows();
            const Index k = cols();
            const Index n = other.rows();

            const std::size_t a_bytes = m * k * sizeof(float);
            const std::size_t b_bytes = n * k * sizeof(float);
            const std::size_t c_bytes = m * n * sizeof(float);
            OpenCLHostStorage out(m, n);

            if (ensure_device_current("resident gate") &&
                other.ensure_device_current("resident gate"))
            {
                if (m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        m_gpu_buffer->buffer,
                        m_backend->data_ptr(),
                        a_bytes,
                        "clEnqueueWriteBuffer(matmul_transposed, a resident)");
                    m_needs_sync_to_device = false;
                }
                if (other.m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        other.m_gpu_buffer->buffer,
                        other.m_backend->data_ptr(),
                        b_bytes,
                        "clEnqueueWriteBuffer(matmul_transposed, b resident)");
                    other.m_needs_sync_to_device = false;
                }

                OpenCLTensorBackend t(m, n);
                t.set_gpu_resident(true);

                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("matmul_rhs_transposed_kernel");
                const cl_mem a_mem = m_gpu_buffer->buffer;
                const cl_mem b_mem = other.m_gpu_buffer->buffer;
                const cl_mem c_mem = t.m_gpu_buffer->buffer;
                const cl_uint m_u32 = static_cast<cl_uint>(m);
                const cl_uint n_u32 = static_cast<cl_uint>(n);
                const cl_uint k_u32 = static_cast<cl_uint>(k);

                check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                    "clSetKernelArg(matmul_transposed, a resident)");
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                    "clSetKernelArg(matmul_transposed, b resident)");
                check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem),
                    "clSetKernelArg(matmul_transposed, c resident)");
                check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &m_u32),
                    "clSetKernelArg(matmul_transposed, m resident)");
                check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32),
                    "clSetKernelArg(matmul_transposed, n resident)");
                check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &k_u32),
                    "clSetKernelArg(matmul_transposed, k resident)");

                const std::size_t global[2] = {m, n};
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel(matmul_transposed resident)");
                t.m_needs_sync_to_host = true;
                t.m_needs_sync_to_device = false;
                return t;
            }

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
                        "clEnqueueWriteBuffer(matmul_transposed, a)");
                    copy_host_to_device(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        b_bytes,
                        "clEnqueueWriteBuffer(matmul_transposed, b)");

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel(
                        "matmul_rhs_transposed_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_mem c_mem = c_buf->buffer;
                    const cl_uint m_u32 = static_cast<cl_uint>(m);
                    const cl_uint n_u32 = static_cast<cl_uint>(n);
                    const cl_uint k_u32 = static_cast<cl_uint>(k);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                        "clSetKernelArg(matmul_transposed, a)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                        "clSetKernelArg(matmul_transposed, b)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem),
                        "clSetKernelArg(matmul_transposed, c)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &m_u32),
                        "clSetKernelArg(matmul_transposed, m)");
                    check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(matmul_transposed, n)");
                    check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &k_u32),
                        "clSetKernelArg(matmul_transposed, k)");

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
                        "clEnqueueNDRangeKernel(matmul_transposed)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(matmul_transposed)");

                    copy_device_to_host(ctx.get_queue(),
                        c_buf->buffer,
                        out.mutable_data_ptr(),
                        c_bytes,
                        "clEnqueueReadBuffer(matmul_transposed, c)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory a_dev(a_bytes);
            opencl::DeviceMemory b_dev(b_bytes);
            opencl::DeviceMemory out_dev(c_bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("matmul_rhs_transposed_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem c_mem = out_dev.get_device_buffer();
            const cl_uint m_u32 = static_cast<cl_uint>(m);
            const cl_uint n_u32 = static_cast<cl_uint>(n);
            const cl_uint k_u32 = static_cast<cl_uint>(k);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                "clSetKernelArg(matmul_transposed, a)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                "clSetKernelArg(matmul_transposed, b)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem),
                "clSetKernelArg(matmul_transposed, c)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &m_u32),
                "clSetKernelArg(matmul_transposed, m)");
            check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(matmul_transposed, n)");
            check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &k_u32),
                "clSetKernelArg(matmul_transposed, k)");

            const std::size_t global[2] = {m, n};
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(matmul_transposed)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(matmul_transposed)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("matmul_transposed", e.what());
        }
    }

    throw_opencl_only_failure(
        "matmul_transposed", "OpenCL runtime unavailable or matrix dimensions are invalid");
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
        throw_opencl_only_failure(
            "matmul_lhs_transposed", "OpenCL runtime unavailable or matrix dimensions are invalid");
    }
    else if (can_use_opencl("matmul_lhs_transposed"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const Index m = rows();
            const Index k = cols();
            const Index n = other.cols();

            const std::size_t a_bytes = m * k * sizeof(float);
            const std::size_t b_bytes = m * n * sizeof(float);
            const std::size_t c_bytes = k * n * sizeof(float);
            OpenCLHostStorage out(k, n);

            if (ensure_device_current("resident gate") &&
                other.ensure_device_current("resident gate"))
            {
                if (m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        m_gpu_buffer->buffer,
                        m_backend->data_ptr(),
                        a_bytes,
                        "clEnqueueWriteBuffer(matmul_lhs_transposed, a resident)");
                    m_needs_sync_to_device = false;
                }
                if (other.m_needs_sync_to_device)
                {
                    copy_host_to_device(ctx.get_queue(),
                        other.m_gpu_buffer->buffer,
                        other.m_backend->data_ptr(),
                        b_bytes,
                        "clEnqueueWriteBuffer(matmul_lhs_transposed, b resident)");
                    other.m_needs_sync_to_device = false;
                }

                OpenCLTensorBackend t(k, n);
                t.set_gpu_resident(true);

                cl_kernel kernel =
                    opencl::KernelManager::instance().get_kernel("matmul_lhs_transposed_kernel");
                const cl_mem a_mem = m_gpu_buffer->buffer;
                const cl_mem b_mem = other.m_gpu_buffer->buffer;
                const cl_mem c_mem = t.m_gpu_buffer->buffer;
                const cl_uint m_u32 = static_cast<cl_uint>(m);
                const cl_uint n_u32 = static_cast<cl_uint>(n);
                const cl_uint k_u32 = static_cast<cl_uint>(k);

                check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                    "clSetKernelArg(matmul_lhs_transposed, a resident)");
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                    "clSetKernelArg(matmul_lhs_transposed, b resident)");
                check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem),
                    "clSetKernelArg(matmul_lhs_transposed, c resident)");
                check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &m_u32),
                    "clSetKernelArg(matmul_lhs_transposed, m resident)");
                check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32),
                    "clSetKernelArg(matmul_lhs_transposed, n resident)");
                check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &k_u32),
                    "clSetKernelArg(matmul_lhs_transposed, k resident)");

                const std::size_t local[2] = {16, 16};
                const std::size_t global[2] = {round_up(k, local[0]), round_up(n, local[1])};
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 2, nullptr, global, local, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel(matmul_lhs_transposed resident)");
                t.m_needs_sync_to_host = true;
                t.m_needs_sync_to_device = false;
                return t;
            }

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
                        "clEnqueueWriteBuffer(matmul_lhs_transposed, a)");
                    copy_host_to_device(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        b_bytes,
                        "clEnqueueWriteBuffer(matmul_lhs_transposed, b)");

                    cl_kernel kernel = opencl::KernelManager::instance().get_kernel(
                        "matmul_lhs_transposed_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_mem c_mem = c_buf->buffer;
                    const cl_uint m_u32 = static_cast<cl_uint>(m);
                    const cl_uint n_u32 = static_cast<cl_uint>(n);
                    const cl_uint k_u32 = static_cast<cl_uint>(k);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                        "clSetKernelArg(matmul_lhs_transposed, a)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                        "clSetKernelArg(matmul_lhs_transposed, b)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem),
                        "clSetKernelArg(matmul_lhs_transposed, c)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &m_u32),
                        "clSetKernelArg(matmul_lhs_transposed, m)");
                    check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(matmul_lhs_transposed, n)");
                    check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &k_u32),
                        "clSetKernelArg(matmul_lhs_transposed, k)");

                    const std::size_t local[2] = {16, 16};
                    const std::size_t global[2] = {round_up(k, local[0]), round_up(n, local[1])};
                    check_cl_error(clEnqueueNDRangeKernel(ctx.get_queue(),
                                       kernel,
                                       2,
                                       nullptr,
                                       global,
                                       local,
                                       0,
                                       nullptr,
                                       nullptr),
                        "clEnqueueNDRangeKernel(matmul_lhs_transposed)");
                    finish_queue_if_not_batching(
                        ctx.get_queue(), "clFinish(matmul_lhs_transposed)");

                    copy_device_to_host(ctx.get_queue(),
                        c_buf->buffer,
                        out.mutable_data_ptr(),
                        c_bytes,
                        "clEnqueueReadBuffer(matmul_lhs_transposed, c)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory a_dev(a_bytes);
            opencl::DeviceMemory b_dev(b_bytes);
            opencl::DeviceMemory out_dev(c_bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("matmul_lhs_transposed_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem c_mem = out_dev.get_device_buffer();
            const cl_uint m_u32 = static_cast<cl_uint>(m);
            const cl_uint n_u32 = static_cast<cl_uint>(n);
            const cl_uint k_u32 = static_cast<cl_uint>(k);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                "clSetKernelArg(matmul_lhs_transposed, a)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                "clSetKernelArg(matmul_lhs_transposed, b)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem),
                "clSetKernelArg(matmul_lhs_transposed, c)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &m_u32),
                "clSetKernelArg(matmul_lhs_transposed, m)");
            check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(matmul_lhs_transposed, n)");
            check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &k_u32),
                "clSetKernelArg(matmul_lhs_transposed, k)");

            const std::size_t local[2] = {16, 16};
            const std::size_t global[2] = {round_up(k, local[0]), round_up(n, local[1])};
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 2, nullptr, global, local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(matmul_lhs_transposed)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(matmul_lhs_transposed)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("matmul_lhs_transposed", e.what());
        }
    }

    throw_opencl_only_failure(
        "matmul_lhs_transposed", "OpenCL runtime unavailable or matrix dimensions are invalid");
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed_add_col_bias(
    const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias) const
{
    if (!m_gpu_resident)
    {
        sync_gpu_if_needed();
    }
    if (!other.m_gpu_resident)
    {
        other.sync_gpu_if_needed();
    }
    if (!bias.m_gpu_resident)
    {
        bias.sync_gpu_if_needed();
    }

    if (shape().size() != 2 || other.shape().size() != 2 || bias.shape().size() != 2)
    {
        throw_opencl_only_failure("matmul_transposed_add_col_bias",
            "OpenCL runtime unavailable or matrix dimensions are invalid");
    }
    if (cols() != other.cols() || bias.rows() != other.rows() || bias.cols() != 1)
    {
        throw_opencl_only_failure("matmul_transposed_add_col_bias",
            "OpenCL runtime unavailable or matrix dimensions are invalid");
    }
    if (!can_use_opencl("matmul_transposed_add_col_bias"))
    {
        throw_opencl_only_failure("matmul_transposed_add_col_bias",
            "OpenCL runtime unavailable or matrix dimensions are invalid");
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

        if (ensure_device_current("resident gate") &&
            other.ensure_device_current("resident gate") &&
            bias.ensure_device_current("resident gate"))
        {
            if (m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    m_gpu_buffer->buffer,
                    m_backend->data_ptr(),
                    a_bytes,
                    "clEnqueueWriteBuffer(matmul_transposed_add_col_bias, a resident)");
                m_needs_sync_to_device = false;
            }
            if (other.m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    other.m_gpu_buffer->buffer,
                    other.m_backend->data_ptr(),
                    b_bytes,
                    "clEnqueueWriteBuffer(matmul_transposed_add_col_bias, b resident)");
                other.m_needs_sync_to_device = false;
            }
            if (bias.m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    bias.m_gpu_buffer->buffer,
                    bias.m_backend->data_ptr(),
                    bias_bytes,
                    "clEnqueueWriteBuffer(matmul_transposed_add_col_bias, bias resident)");
                bias.m_needs_sync_to_device = false;
            }

            OpenCLTensorBackend t(m, n);
            t.set_gpu_resident(true);

            cl_kernel kernel =
                opencl::KernelManager::instance().get_kernel("matmul_rhs_transposed_bias_kernel");
            const cl_mem a_mem = m_gpu_buffer->buffer;
            const cl_mem b_mem = other.m_gpu_buffer->buffer;
            const cl_mem bias_mem = bias.m_gpu_buffer->buffer;
            const cl_mem c_mem = t.m_gpu_buffer->buffer;
            const cl_uint m_u32 = static_cast<cl_uint>(m);
            const cl_uint n_u32 = static_cast<cl_uint>(n);
            const cl_uint k_u32 = static_cast<cl_uint>(k);

            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                "clSetKernelArg(matmul_transposed_add_col_bias, a resident)");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                "clSetKernelArg(matmul_transposed_add_col_bias, b resident)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem),
                "clSetKernelArg(matmul_transposed_add_col_bias, bias resident)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem),
                "clSetKernelArg(matmul_transposed_add_col_bias, c resident)");
            check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32),
                "clSetKernelArg(matmul_transposed_add_col_bias, m resident)");
            check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(matmul_transposed_add_col_bias, n resident)");
            check_cl_error(clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32),
                "clSetKernelArg(matmul_transposed_add_col_bias, k resident)");

            const std::size_t global[2] = {m, n};
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(matmul_transposed_add_col_bias resident)");
            t.m_needs_sync_to_host = true;
            t.m_needs_sync_to_device = false;
            return t;
        }

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
                    m_backend->data_ptr(),
                    a_bytes,
                    "clEnqueueWriteBuffer(matmul_transposed_add_col_bias, a)");
                copy_host_to_device(ctx.get_queue(),
                    b_buf->buffer,
                    other.m_backend->data_ptr(),
                    b_bytes,
                    "clEnqueueWriteBuffer(matmul_transposed_add_col_bias, b)");
                copy_host_to_device(ctx.get_queue(),
                    bias_buf->buffer,
                    bias.m_backend->data_ptr(),
                    bias_bytes,
                    "clEnqueueWriteBuffer(matmul_transposed_add_col_bias, bias)");

                cl_kernel kernel = opencl::KernelManager::instance().get_kernel(
                    "matmul_rhs_transposed_bias_kernel");
                const cl_mem a_mem = a_buf->buffer;
                const cl_mem b_mem = b_buf->buffer;
                const cl_mem bias_mem = bias_buf->buffer;
                const cl_mem c_mem = c_buf->buffer;
                const cl_uint m_u32 = static_cast<cl_uint>(m);
                const cl_uint n_u32 = static_cast<cl_uint>(n);
                const cl_uint k_u32 = static_cast<cl_uint>(k);

                check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                    "clSetKernelArg(matmul_transposed_add_col_bias, a)");
                check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                    "clSetKernelArg(matmul_transposed_add_col_bias, b)");
                check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem),
                    "clSetKernelArg(matmul_transposed_add_col_bias, bias)");
                check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem),
                    "clSetKernelArg(matmul_transposed_add_col_bias, c)");
                check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32),
                    "clSetKernelArg(matmul_transposed_add_col_bias, m)");
                check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32),
                    "clSetKernelArg(matmul_transposed_add_col_bias, n)");
                check_cl_error(clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32),
                    "clSetKernelArg(matmul_transposed_add_col_bias, k)");

                const std::size_t global[2] = {m, n};
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel(matmul_transposed_add_col_bias)");
                finish_queue_if_not_batching(
                    ctx.get_queue(), "clFinish(matmul_transposed_add_col_bias)");

                copy_device_to_host(ctx.get_queue(),
                    c_buf->buffer,
                    out.mutable_data_ptr(),
                    c_bytes,
                    "clEnqueueReadBuffer(matmul_transposed_add_col_bias, c)");

                OpenCLTensorBackend t;
                t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                return t;
            }
        }

        opencl::DeviceMemory a_dev(a_bytes);
        opencl::DeviceMemory b_dev(b_bytes);
        opencl::DeviceMemory bias_dev(bias_bytes);
        opencl::DeviceMemory out_dev(c_bytes);
        a_dev.copy_to_device(m_backend->data_ptr());
        b_dev.copy_to_device(other.m_backend->data_ptr());
        bias_dev.copy_to_device(bias.m_backend->data_ptr());

        cl_kernel kernel =
            opencl::KernelManager::instance().get_kernel("matmul_rhs_transposed_bias_kernel");
        const cl_mem a_mem = a_dev.get_device_buffer();
        const cl_mem b_mem = b_dev.get_device_buffer();
        const cl_mem bias_mem = bias_dev.get_device_buffer();
        const cl_mem c_mem = out_dev.get_device_buffer();
        const cl_uint m_u32 = static_cast<cl_uint>(m);
        const cl_uint n_u32 = static_cast<cl_uint>(n);
        const cl_uint k_u32 = static_cast<cl_uint>(k);

        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
            "clSetKernelArg(matmul_transposed_add_col_bias, a)");
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
            "clSetKernelArg(matmul_transposed_add_col_bias, b)");
        check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem),
            "clSetKernelArg(matmul_transposed_add_col_bias, bias)");
        check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem),
            "clSetKernelArg(matmul_transposed_add_col_bias, c)");
        check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32),
            "clSetKernelArg(matmul_transposed_add_col_bias, m)");
        check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32),
            "clSetKernelArg(matmul_transposed_add_col_bias, n)");
        check_cl_error(clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32),
            "clSetKernelArg(matmul_transposed_add_col_bias, k)");

        const std::size_t global[2] = {m, n};
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel(matmul_transposed_add_col_bias)");
        finish_queue_if_not_batching(ctx.get_queue(), "clFinish(matmul_transposed_add_col_bias)");

        out_dev.copy_from_device(out.mutable_data_ptr());

        OpenCLTensorBackend t;
        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
        return t;
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("matmul_transposed_add_col_bias", e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed_add_col_bias_sigmoid(
    const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias) const
{
    if (!m_gpu_resident) sync_gpu_if_needed();
    if (!other.m_gpu_resident) other.sync_gpu_if_needed();
    if (!bias.m_gpu_resident) bias.sync_gpu_if_needed();

    if (shape().size() != 2 || other.shape().size() != 2 || bias.shape().size() != 2 ||
        cols() != other.cols() || bias.rows() != other.rows() || bias.cols() != 1 ||
        !can_use_opencl("matmul_transposed_add_col_bias_sigmoid"))
    {
        throw_opencl_only_failure("matmul_transposed_add_col_bias_sigmoid",
            "OpenCL runtime unavailable or matrix dimensions are invalid");
    }
    try
    {
        const auto& ctx = opencl::OpenCLContext::instance();
        const Index m = rows(), k = cols(), n = other.rows();
        const std::size_t a_bytes = m * k * sizeof(float);
        const std::size_t b_bytes = n * k * sizeof(float);
        const std::size_t bias_bytes = n * sizeof(float);
        const std::size_t c_bytes = m * n * sizeof(float);
        OpenCLHostStorage out(m, n);
        constexpr const char* kname = "matmul_rhs_transposed_bias_sigmoid_kernel";

        if (ensure_device_current("resident gate") &&
            other.ensure_device_current("resident gate") &&
            bias.ensure_device_current("resident gate"))
        {
            if (m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    m_gpu_buffer->buffer,
                    m_backend->data_ptr(),
                    a_bytes,
                    "matmul_bias_sigmoid a");
                m_needs_sync_to_device = false;
            }
            if (other.m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    other.m_gpu_buffer->buffer,
                    other.m_backend->data_ptr(),
                    b_bytes,
                    "matmul_bias_sigmoid b");
                other.m_needs_sync_to_device = false;
            }
            if (bias.m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    bias.m_gpu_buffer->buffer,
                    bias.m_backend->data_ptr(),
                    bias_bytes,
                    "matmul_bias_sigmoid bias");
                bias.m_needs_sync_to_device = false;
            }
            OpenCLTensorBackend t(m, n);
            t.set_gpu_resident(true);
            cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kname);
            const cl_mem a_mem = m_gpu_buffer->buffer, b_mem = other.m_gpu_buffer->buffer;
            const cl_mem bias_mem = bias.m_gpu_buffer->buffer, c_mem = t.m_gpu_buffer->buffer;
            const cl_uint m_u32 = static_cast<cl_uint>(m), n_u32 = static_cast<cl_uint>(n),
                          k_u32 = static_cast<cl_uint>(k);
            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_bias_sigmoid 0");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_bias_sigmoid 1");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "matmul_bias_sigmoid 2");
            check_cl_error(
                clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), "matmul_bias_sigmoid 3");
            check_cl_error(
                clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), "matmul_bias_sigmoid 4");
            check_cl_error(
                clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), "matmul_bias_sigmoid 5");
            check_cl_error(
                clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), "matmul_bias_sigmoid 6");
            const std::size_t global[2] = {m, n};
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                "matmul_bias_sigmoid enqueue");
            t.m_needs_sync_to_host = true;
            t.m_needs_sync_to_device = false;
            return t;
        }
        tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
        if (pool)
        {
            auto a_buf = pool->acquire(a_bytes), b_buf = pool->acquire(b_bytes);
            auto bias_buf = pool->acquire(bias_bytes), c_buf = pool->acquire(c_bytes);
            if (a_buf && b_buf && bias_buf && c_buf)
            {
                copy_host_to_device(ctx.get_queue(),
                    a_buf->buffer,
                    m_backend->data_ptr(),
                    a_bytes,
                    "matmul_bias_sigmoid a");
                copy_host_to_device(ctx.get_queue(),
                    b_buf->buffer,
                    other.m_backend->data_ptr(),
                    b_bytes,
                    "matmul_bias_sigmoid b");
                copy_host_to_device(ctx.get_queue(),
                    bias_buf->buffer,
                    bias.m_backend->data_ptr(),
                    bias_bytes,
                    "matmul_bias_sigmoid bias");
                cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kname);
                const cl_mem a_mem = a_buf->buffer, b_mem = b_buf->buffer;
                const cl_mem bias_mem = bias_buf->buffer, c_mem = c_buf->buffer;
                const cl_uint m_u32 = static_cast<cl_uint>(m), n_u32 = static_cast<cl_uint>(n),
                              k_u32 = static_cast<cl_uint>(k);
                check_cl_error(
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_bias_sigmoid 0");
                check_cl_error(
                    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_bias_sigmoid 1");
                check_cl_error(
                    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "matmul_bias_sigmoid 2");
                check_cl_error(
                    clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), "matmul_bias_sigmoid 3");
                check_cl_error(
                    clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), "matmul_bias_sigmoid 4");
                check_cl_error(
                    clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), "matmul_bias_sigmoid 5");
                check_cl_error(
                    clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), "matmul_bias_sigmoid 6");
                const std::size_t global[2] = {m, n};
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                    "matmul_bias_sigmoid enqueue");
                finish_queue_if_not_batching(ctx.get_queue(), "matmul_bias_sigmoid finish");
                copy_device_to_host(ctx.get_queue(),
                    c_buf->buffer,
                    out.mutable_data_ptr(),
                    c_bytes,
                    "matmul_bias_sigmoid read");
                OpenCLTensorBackend t;
                t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                return t;
            }
        }
        opencl::DeviceMemory a_dev(a_bytes), b_dev(b_bytes), bias_dev(bias_bytes), out_dev(c_bytes);
        a_dev.copy_to_device(m_backend->data_ptr());
        b_dev.copy_to_device(other.m_backend->data_ptr());
        bias_dev.copy_to_device(bias.m_backend->data_ptr());
        cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kname);
        const cl_mem a_mem = a_dev.get_device_buffer(), b_mem = b_dev.get_device_buffer();
        const cl_mem bias_mem = bias_dev.get_device_buffer(), c_mem = out_dev.get_device_buffer();
        const cl_uint m_u32 = static_cast<cl_uint>(m), n_u32 = static_cast<cl_uint>(n),
                      k_u32 = static_cast<cl_uint>(k);
        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_bias_sigmoid 0");
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_bias_sigmoid 1");
        check_cl_error(
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "matmul_bias_sigmoid 2");
        check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), "matmul_bias_sigmoid 3");
        check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), "matmul_bias_sigmoid 4");
        check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), "matmul_bias_sigmoid 5");
        check_cl_error(clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), "matmul_bias_sigmoid 6");
        const std::size_t global[2] = {m, n};
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
            "matmul_bias_sigmoid enqueue");
        finish_queue_if_not_batching(ctx.get_queue(), "matmul_bias_sigmoid finish");
        out_dev.copy_from_device(out.mutable_data_ptr());
        OpenCLTensorBackend t;
        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
        return t;
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("matmul_transposed_add_col_bias_sigmoid", e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed_add_col_bias_tanh(
    const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias) const
{
    if (!m_gpu_resident) sync_gpu_if_needed();
    if (!other.m_gpu_resident) other.sync_gpu_if_needed();
    if (!bias.m_gpu_resident) bias.sync_gpu_if_needed();

    if (shape().size() != 2 || other.shape().size() != 2 || bias.shape().size() != 2 ||
        cols() != other.cols() || bias.rows() != other.rows() || bias.cols() != 1 ||
        !can_use_opencl("matmul_transposed_add_col_bias_tanh"))
    {
        throw_opencl_only_failure("matmul_transposed_add_col_bias_tanh",
            "OpenCL runtime unavailable or matrix dimensions are invalid");
    }
    try
    {
        const auto& ctx = opencl::OpenCLContext::instance();
        const Index m = rows(), k = cols(), n = other.rows();
        const std::size_t a_bytes = m * k * sizeof(float);
        const std::size_t b_bytes = n * k * sizeof(float);
        const std::size_t bias_bytes = n * sizeof(float);
        const std::size_t c_bytes = m * n * sizeof(float);
        OpenCLHostStorage out(m, n);
        constexpr const char* kname = "matmul_rhs_transposed_bias_tanh_kernel";

        if (ensure_device_current("resident gate") &&
            other.ensure_device_current("resident gate") &&
            bias.ensure_device_current("resident gate"))
        {
            if (m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    m_gpu_buffer->buffer,
                    m_backend->data_ptr(),
                    a_bytes,
                    "matmul_bias_tanh a");
                m_needs_sync_to_device = false;
            }
            if (other.m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    other.m_gpu_buffer->buffer,
                    other.m_backend->data_ptr(),
                    b_bytes,
                    "matmul_bias_tanh b");
                other.m_needs_sync_to_device = false;
            }
            if (bias.m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    bias.m_gpu_buffer->buffer,
                    bias.m_backend->data_ptr(),
                    bias_bytes,
                    "matmul_bias_tanh bias");
                bias.m_needs_sync_to_device = false;
            }
            OpenCLTensorBackend t(m, n);
            t.set_gpu_resident(true);
            cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kname);
            const cl_mem a_mem = m_gpu_buffer->buffer, b_mem = other.m_gpu_buffer->buffer;
            const cl_mem bias_mem = bias.m_gpu_buffer->buffer, c_mem = t.m_gpu_buffer->buffer;
            const cl_uint m_u32 = static_cast<cl_uint>(m), n_u32 = static_cast<cl_uint>(n),
                          k_u32 = static_cast<cl_uint>(k);
            check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_bias_tanh 0");
            check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_bias_tanh 1");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "matmul_bias_tanh 2");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), "matmul_bias_tanh 3");
            check_cl_error(
                clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), "matmul_bias_tanh 4");
            check_cl_error(
                clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), "matmul_bias_tanh 5");
            check_cl_error(
                clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), "matmul_bias_tanh 6");
            const std::size_t global[2] = {m, n};
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                "matmul_bias_tanh enqueue");
            t.m_needs_sync_to_host = true;
            t.m_needs_sync_to_device = false;
            return t;
        }
        tensor::GPUBufferPool* pool = OpenCLTensorBackend::get_buffer_pool();
        if (pool)
        {
            auto a_buf = pool->acquire(a_bytes), b_buf = pool->acquire(b_bytes);
            auto bias_buf = pool->acquire(bias_bytes), c_buf = pool->acquire(c_bytes);
            if (a_buf && b_buf && bias_buf && c_buf)
            {
                copy_host_to_device(ctx.get_queue(),
                    a_buf->buffer,
                    m_backend->data_ptr(),
                    a_bytes,
                    "matmul_bias_tanh a");
                copy_host_to_device(ctx.get_queue(),
                    b_buf->buffer,
                    other.m_backend->data_ptr(),
                    b_bytes,
                    "matmul_bias_tanh b");
                copy_host_to_device(ctx.get_queue(),
                    bias_buf->buffer,
                    bias.m_backend->data_ptr(),
                    bias_bytes,
                    "matmul_bias_tanh bias");
                cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kname);
                const cl_mem a_mem = a_buf->buffer, b_mem = b_buf->buffer;
                const cl_mem bias_mem = bias_buf->buffer, c_mem = c_buf->buffer;
                const cl_uint m_u32 = static_cast<cl_uint>(m), n_u32 = static_cast<cl_uint>(n),
                              k_u32 = static_cast<cl_uint>(k);
                check_cl_error(
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_bias_tanh 0");
                check_cl_error(
                    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_bias_tanh 1");
                check_cl_error(
                    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "matmul_bias_tanh 2");
                check_cl_error(
                    clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), "matmul_bias_tanh 3");
                check_cl_error(
                    clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), "matmul_bias_tanh 4");
                check_cl_error(
                    clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), "matmul_bias_tanh 5");
                check_cl_error(
                    clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), "matmul_bias_tanh 6");
                const std::size_t global[2] = {m, n};
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                    "matmul_bias_tanh enqueue");
                finish_queue_if_not_batching(ctx.get_queue(), "matmul_bias_tanh finish");
                copy_device_to_host(ctx.get_queue(),
                    c_buf->buffer,
                    out.mutable_data_ptr(),
                    c_bytes,
                    "matmul_bias_tanh read");
                OpenCLTensorBackend t;
                t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                return t;
            }
        }
        opencl::DeviceMemory a_dev(a_bytes), b_dev(b_bytes), bias_dev(bias_bytes), out_dev(c_bytes);
        a_dev.copy_to_device(m_backend->data_ptr());
        b_dev.copy_to_device(other.m_backend->data_ptr());
        bias_dev.copy_to_device(bias.m_backend->data_ptr());
        cl_kernel kernel = opencl::KernelManager::instance().get_kernel(kname);
        const cl_mem a_mem = a_dev.get_device_buffer(), b_mem = b_dev.get_device_buffer();
        const cl_mem bias_mem = bias_dev.get_device_buffer(), c_mem = out_dev.get_device_buffer();
        const cl_uint m_u32 = static_cast<cl_uint>(m), n_u32 = static_cast<cl_uint>(n),
                      k_u32 = static_cast<cl_uint>(k);
        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_bias_tanh 0");
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_bias_tanh 1");
        check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "matmul_bias_tanh 2");
        check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), "matmul_bias_tanh 3");
        check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), "matmul_bias_tanh 4");
        check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), "matmul_bias_tanh 5");
        check_cl_error(clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), "matmul_bias_tanh 6");
        const std::size_t global[2] = {m, n};
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
            "matmul_bias_tanh enqueue");
        finish_queue_if_not_batching(ctx.get_queue(), "matmul_bias_tanh finish");
        out_dev.copy_from_device(out.mutable_data_ptr());
        OpenCLTensorBackend t;
        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
        return t;
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("matmul_transposed_add_col_bias_tanh", e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed_add_col_bias_relu(
    const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias) const
{
    if (!m_gpu_resident) sync_gpu_if_needed();
    if (!other.m_gpu_resident) other.sync_gpu_if_needed();
    if (!bias.m_gpu_resident) bias.sync_gpu_if_needed();

    if (shape().size() != 2 || other.shape().size() != 2 || bias.shape().size() != 2 ||
        cols() != other.cols() || bias.rows() != other.rows() || bias.cols() != 1 ||
        !can_use_opencl("matmul_transposed_add_col_bias_relu"))
    {
        throw_opencl_only_failure("matmul_transposed_add_col_bias_relu",
            "OpenCL runtime unavailable or matrix dimensions are invalid");
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
                    m_backend->data_ptr(),
                    a_bytes,
                    "clEnqueueWriteBuffer(matmul_bias_relu, a)");
                m_needs_sync_to_device = false;
            }
            if (other.m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    other.m_gpu_buffer->buffer,
                    other.m_backend->data_ptr(),
                    b_bytes,
                    "clEnqueueWriteBuffer(matmul_bias_relu, b)");
                other.m_needs_sync_to_device = false;
            }
            if (bias.m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    bias.m_gpu_buffer->buffer,
                    bias.m_backend->data_ptr(),
                    bias_bytes,
                    "clEnqueueWriteBuffer(matmul_bias_relu, bias)");
                bias.m_needs_sync_to_device = false;
            }

            OpenCLTensorBackend t(m, n);
            t.set_gpu_resident(true);
            cl_kernel kernel = opencl::KernelManager::instance().get_kernel(
                "matmul_rhs_transposed_bias_relu_kernel");
            const cl_mem a_mem = m_gpu_buffer->buffer;
            const cl_mem b_mem = other.m_gpu_buffer->buffer;
            const cl_mem bias_mem = bias.m_gpu_buffer->buffer;
            const cl_mem c_mem = t.m_gpu_buffer->buffer;
            const cl_uint m_u32 = static_cast<cl_uint>(m);
            const cl_uint n_u32 = static_cast<cl_uint>(n);
            const cl_uint k_u32 = static_cast<cl_uint>(k);
            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_bias_relu arg0");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_bias_relu arg1");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "matmul_bias_relu arg2");
            check_cl_error(
                clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), "matmul_bias_relu arg3");
            check_cl_error(
                clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), "matmul_bias_relu arg4");
            check_cl_error(
                clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), "matmul_bias_relu arg5");
            check_cl_error(
                clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), "matmul_bias_relu arg6");
            const std::size_t global[2] = {m, n};
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(matmul_bias_relu resident)");
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
                    m_backend->data_ptr(),
                    a_bytes,
                    "clEnqueueWriteBuffer(matmul_bias_relu, a)");
                copy_host_to_device(ctx.get_queue(),
                    b_buf->buffer,
                    other.m_backend->data_ptr(),
                    b_bytes,
                    "clEnqueueWriteBuffer(matmul_bias_relu, b)");
                copy_host_to_device(ctx.get_queue(),
                    bias_buf->buffer,
                    bias.m_backend->data_ptr(),
                    bias_bytes,
                    "clEnqueueWriteBuffer(matmul_bias_relu, bias)");

                cl_kernel kernel = opencl::KernelManager::instance().get_kernel(
                    "matmul_rhs_transposed_bias_relu_kernel");
                const cl_mem a_mem = a_buf->buffer;
                const cl_mem b_mem = b_buf->buffer;
                const cl_mem bias_mem = bias_buf->buffer;
                const cl_mem c_mem = c_buf->buffer;
                const cl_uint m_u32 = static_cast<cl_uint>(m);
                const cl_uint n_u32 = static_cast<cl_uint>(n);
                const cl_uint k_u32 = static_cast<cl_uint>(k);
                check_cl_error(
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_bias_relu arg0");
                check_cl_error(
                    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_bias_relu arg1");
                check_cl_error(
                    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "matmul_bias_relu arg2");
                check_cl_error(
                    clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), "matmul_bias_relu arg3");
                check_cl_error(
                    clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), "matmul_bias_relu arg4");
                check_cl_error(
                    clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), "matmul_bias_relu arg5");
                check_cl_error(
                    clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), "matmul_bias_relu arg6");
                const std::size_t global[2] = {m, n};
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel(matmul_bias_relu)");
                finish_queue_if_not_batching(ctx.get_queue(), "clFinish(matmul_bias_relu)");
                copy_device_to_host(ctx.get_queue(),
                    c_buf->buffer,
                    out.mutable_data_ptr(),
                    c_bytes,
                    "clEnqueueReadBuffer(matmul_bias_relu, c)");
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
        a_dev.copy_to_device(m_backend->data_ptr());
        b_dev.copy_to_device(other.m_backend->data_ptr());
        bias_dev.copy_to_device(bias.m_backend->data_ptr());

        cl_kernel kernel =
            opencl::KernelManager::instance().get_kernel("matmul_rhs_transposed_bias_relu_kernel");
        const cl_mem a_mem = a_dev.get_device_buffer();
        const cl_mem b_mem = b_dev.get_device_buffer();
        const cl_mem bias_mem = bias_dev.get_device_buffer();
        const cl_mem c_mem = out_dev.get_device_buffer();
        const cl_uint m_u32 = static_cast<cl_uint>(m);
        const cl_uint n_u32 = static_cast<cl_uint>(n);
        const cl_uint k_u32 = static_cast<cl_uint>(k);
        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_bias_relu arg0");
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_bias_relu arg1");
        check_cl_error(
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "matmul_bias_relu arg2");
        check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), "matmul_bias_relu arg3");
        check_cl_error(clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), "matmul_bias_relu arg4");
        check_cl_error(clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), "matmul_bias_relu arg5");
        check_cl_error(clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), "matmul_bias_relu arg6");
        const std::size_t global[2] = {m, n};
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel(matmul_bias_relu)");
        finish_queue_if_not_batching(ctx.get_queue(), "clFinish(matmul_bias_relu)");
        out_dev.copy_from_device(out.mutable_data_ptr());

        OpenCLTensorBackend t;
        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
        return t;
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("matmul_transposed_add_col_bias_relu", e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed_add_col_bias_leaky_relu(
    const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias, float alpha) const
{
    if (!m_gpu_resident) sync_gpu_if_needed();
    if (!other.m_gpu_resident) other.sync_gpu_if_needed();
    if (!bias.m_gpu_resident) bias.sync_gpu_if_needed();

    if (shape().size() != 2 || other.shape().size() != 2 || bias.shape().size() != 2 ||
        cols() != other.cols() || bias.rows() != other.rows() || bias.cols() != 1 ||
        !can_use_opencl("matmul_transposed_add_col_bias_leaky_relu"))
    {
        throw_opencl_only_failure("matmul_transposed_add_col_bias_leaky_relu",
            "OpenCL runtime unavailable or matrix dimensions are invalid");
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
                    m_backend->data_ptr(),
                    a_bytes,
                    "clEnqueueWriteBuffer(matmul_bias_lrelu, a)");
                m_needs_sync_to_device = false;
            }
            if (other.m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    other.m_gpu_buffer->buffer,
                    other.m_backend->data_ptr(),
                    b_bytes,
                    "clEnqueueWriteBuffer(matmul_bias_lrelu, b)");
                other.m_needs_sync_to_device = false;
            }
            if (bias.m_needs_sync_to_device)
            {
                copy_host_to_device(ctx.get_queue(),
                    bias.m_gpu_buffer->buffer,
                    bias.m_backend->data_ptr(),
                    bias_bytes,
                    "clEnqueueWriteBuffer(matmul_bias_lrelu, bias)");
                bias.m_needs_sync_to_device = false;
            }

            OpenCLTensorBackend t(m, n);
            t.set_gpu_resident(true);
            cl_kernel kernel = opencl::KernelManager::instance().get_kernel(
                "matmul_rhs_transposed_bias_leaky_relu_kernel");
            const cl_mem a_mem = m_gpu_buffer->buffer;
            const cl_mem b_mem = other.m_gpu_buffer->buffer;
            const cl_mem bias_mem = bias.m_gpu_buffer->buffer;
            const cl_mem c_mem = t.m_gpu_buffer->buffer;
            const cl_uint m_u32 = static_cast<cl_uint>(m);
            const cl_uint n_u32 = static_cast<cl_uint>(n);
            const cl_uint k_u32 = static_cast<cl_uint>(k);
            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_bias_lrelu arg0");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_bias_lrelu arg1");
            check_cl_error(
                clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "matmul_bias_lrelu arg2");
            check_cl_error(
                clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), "matmul_bias_lrelu arg3");
            check_cl_error(
                clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), "matmul_bias_lrelu arg4");
            check_cl_error(
                clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), "matmul_bias_lrelu arg5");
            check_cl_error(
                clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), "matmul_bias_lrelu arg6");
            check_cl_error(
                clSetKernelArg(kernel, 7, sizeof(float), &alpha), "matmul_bias_lrelu arg7");
            const std::size_t global[2] = {m, n};
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(matmul_bias_lrelu resident)");
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
                    m_backend->data_ptr(),
                    a_bytes,
                    "clEnqueueWriteBuffer(matmul_bias_lrelu, a)");
                copy_host_to_device(ctx.get_queue(),
                    b_buf->buffer,
                    other.m_backend->data_ptr(),
                    b_bytes,
                    "clEnqueueWriteBuffer(matmul_bias_lrelu, b)");
                copy_host_to_device(ctx.get_queue(),
                    bias_buf->buffer,
                    bias.m_backend->data_ptr(),
                    bias_bytes,
                    "clEnqueueWriteBuffer(matmul_bias_lrelu, bias)");

                cl_kernel kernel = opencl::KernelManager::instance().get_kernel(
                    "matmul_rhs_transposed_bias_leaky_relu_kernel");
                const cl_mem a_mem = a_buf->buffer;
                const cl_mem b_mem = b_buf->buffer;
                const cl_mem bias_mem = bias_buf->buffer;
                const cl_mem c_mem = c_buf->buffer;
                const cl_uint m_u32 = static_cast<cl_uint>(m);
                const cl_uint n_u32 = static_cast<cl_uint>(n);
                const cl_uint k_u32 = static_cast<cl_uint>(k);
                check_cl_error(
                    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_bias_lrelu arg0");
                check_cl_error(
                    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_bias_lrelu arg1");
                check_cl_error(
                    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "matmul_bias_lrelu arg2");
                check_cl_error(
                    clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), "matmul_bias_lrelu arg3");
                check_cl_error(
                    clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), "matmul_bias_lrelu arg4");
                check_cl_error(
                    clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), "matmul_bias_lrelu arg5");
                check_cl_error(
                    clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), "matmul_bias_lrelu arg6");
                check_cl_error(
                    clSetKernelArg(kernel, 7, sizeof(float), &alpha), "matmul_bias_lrelu arg7");
                const std::size_t global[2] = {m, n};
                check_cl_error(
                    clEnqueueNDRangeKernel(
                        ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
                    "clEnqueueNDRangeKernel(matmul_bias_lrelu)");
                finish_queue_if_not_batching(ctx.get_queue(), "clFinish(matmul_bias_lrelu)");
                copy_device_to_host(ctx.get_queue(),
                    c_buf->buffer,
                    out.mutable_data_ptr(),
                    c_bytes,
                    "clEnqueueReadBuffer(matmul_bias_lrelu, c)");
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
        a_dev.copy_to_device(m_backend->data_ptr());
        b_dev.copy_to_device(other.m_backend->data_ptr());
        bias_dev.copy_to_device(bias.m_backend->data_ptr());

        cl_kernel kernel = opencl::KernelManager::instance().get_kernel(
            "matmul_rhs_transposed_bias_leaky_relu_kernel");
        const cl_mem a_mem = a_dev.get_device_buffer();
        const cl_mem b_mem = b_dev.get_device_buffer();
        const cl_mem bias_mem = bias_dev.get_device_buffer();
        const cl_mem c_mem = out_dev.get_device_buffer();
        const cl_uint m_u32 = static_cast<cl_uint>(m);
        const cl_uint n_u32 = static_cast<cl_uint>(n);
        const cl_uint k_u32 = static_cast<cl_uint>(k);
        check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "matmul_bias_lrelu arg0");
        check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "matmul_bias_lrelu arg1");
        check_cl_error(
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_mem), "matmul_bias_lrelu arg2");
        check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_mem), &c_mem), "matmul_bias_lrelu arg3");
        check_cl_error(
            clSetKernelArg(kernel, 4, sizeof(cl_uint), &m_u32), "matmul_bias_lrelu arg4");
        check_cl_error(
            clSetKernelArg(kernel, 5, sizeof(cl_uint), &n_u32), "matmul_bias_lrelu arg5");
        check_cl_error(
            clSetKernelArg(kernel, 6, sizeof(cl_uint), &k_u32), "matmul_bias_lrelu arg6");
        check_cl_error(clSetKernelArg(kernel, 7, sizeof(float), &alpha), "matmul_bias_lrelu arg7");
        const std::size_t global[2] = {m, n};
        check_cl_error(
            clEnqueueNDRangeKernel(
                ctx.get_queue(), kernel, 2, nullptr, global, nullptr, 0, nullptr, nullptr),
            "clEnqueueNDRangeKernel(matmul_bias_lrelu)");
        finish_queue_if_not_batching(ctx.get_queue(), "clFinish(matmul_bias_lrelu)");
        out_dev.copy_from_device(out.mutable_data_ptr());

        OpenCLTensorBackend t;
        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
        return t;
    }
    catch (const std::exception& e)
    {
        throw_opencl_only_failure("matmul_transposed_add_col_bias_leaky_relu", e.what());
    }
}

OpenCLTensorBackend OpenCLTensorBackend::transpose() const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    if (shape().size() != 2)
    {
        throw std::invalid_argument("transpose: tensor must be rank-2");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("transpose"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const Index in_rows = rows();
            const Index in_cols = cols();
            const std::size_t bytes = in_rows * in_cols * sizeof(float);

            OpenCLHostStorage out(in_cols, in_rows);
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
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(transpose)");

                    // GPU-resident mode: keep result on GPU
                    if (ensure_device_current("resident gate"))
                    {
                        OpenCLTensorBackend t;
                        t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                        t.m_has_gpu_memory = true;
                        if (out_buf)
                        {
                            t.m_gpu_buffer =
                                std::make_unique<tensor::GPUBuffer>(std::move(*out_buf));
                        }
                        t.set_gpu_resident(true);
                        t.m_needs_sync_to_host = true;
                        t.m_needs_sync_to_device = false;
                        return t;
                    }

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(transpose, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
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
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(transpose)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("transpose", e.what());
        }
    }

    throw_opencl_only_failure("transpose", "OpenCL runtime unavailable or tensor rank is invalid");
}

OpenCLTensorBackend OpenCLTensorBackend::block(
    Index row, Index col, Index rows_n, Index cols_n) const
{
    sync_gpu();
    if (shape().size() != 2)
    {
        throw std::invalid_argument("block is only valid for rank-2 tensors");
    }
    if (row + rows_n > rows() || col + cols_n > cols())
    {
        throw std::out_of_range("block exceeds tensor bounds");
    }

    OpenCLTensorBackend out(rows_n, cols_n);
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
OpenCLTensorBackend OpenCLTensorBackend::compare_lt(const OpenCLTensorBackend& other) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();
    if (shape() != other.shape())
    {
        const auto& ls = shape();
        const auto& rs = other.shape();
        if (ls.size() != 2 || rs.size() != 2)
            throw std::invalid_argument("compare_lt: broadcasting only supported for 2D tensors");
        const bool rows_ok = (ls[0] == rs[0] || ls[0] == 1 || rs[0] == 1);
        const bool cols_ok = (ls[1] == rs[1] || ls[1] == 1 || rs[1] == 1);
        if (!rows_ok || !cols_ok)
            throw std::invalid_argument("compare_lt: shapes are not broadcast-compatible");
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
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("compare_lt"))
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
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (a_buf && b_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(compare_lt, a)");
                    copy_host_to_device(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(compare_lt, b)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("compare_lt_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                        "clSetKernelArg(compare_lt, a)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                        "clSetKernelArg(compare_lt, b)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(compare_lt, out)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(compare_lt, size)");

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
                        "clEnqueueNDRangeKernel(compare_lt)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_lt)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_lt, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("compare_lt_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(compare_lt, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(compare_lt, b)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(compare_lt, out)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(compare_lt, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(compare_lt)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_lt)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("compare_lt", e.what());
        }
    }

    throw_opencl_only_failure("compare_lt", "OpenCL runtime unavailable or tensor shape mismatch");
}

OpenCLTensorBackend OpenCLTensorBackend::compare_gt(const OpenCLTensorBackend& other) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("compare_gt", "OpenCL path requires matching tensor shapes");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("compare_gt"))
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
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (a_buf && b_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(compare_gt, a)");
                    copy_host_to_device(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(compare_gt, b)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("compare_gt_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                        "clSetKernelArg(compare_gt, a)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                        "clSetKernelArg(compare_gt, b)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(compare_gt, out)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(compare_gt, size)");

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
                        "clEnqueueNDRangeKernel(compare_gt)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_gt)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_gt, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("compare_gt_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(compare_gt, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(compare_gt, b)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(compare_gt, out)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(compare_gt, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(compare_gt)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_gt)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("compare_gt", e.what());
        }
    }

    throw_opencl_only_failure("compare_gt", "OpenCL runtime unavailable or tensor shape mismatch");
}

OpenCLTensorBackend OpenCLTensorBackend::compare_le(const OpenCLTensorBackend& other) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("compare_le", "OpenCL path requires matching tensor shapes");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("compare_le"))
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
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (a_buf && b_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(compare_le, a)");
                    copy_host_to_device(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(compare_le, b)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("compare_le_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                        "clSetKernelArg(compare_le, a)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                        "clSetKernelArg(compare_le, b)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(compare_le, out)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(compare_le, size)");

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
                        "clEnqueueNDRangeKernel(compare_le)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_le)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_le, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("compare_le_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(compare_le, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(compare_le, b)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(compare_le, out)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(compare_le, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(compare_le)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_le)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("compare_le", e.what());
        }
    }

    throw_opencl_only_failure("compare_le", "OpenCL runtime unavailable or tensor shape mismatch");
}

OpenCLTensorBackend OpenCLTensorBackend::compare_ge(const OpenCLTensorBackend& other) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("compare_ge", "OpenCL path requires matching tensor shapes");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("compare_ge"))
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
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (a_buf && b_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(compare_ge, a)");
                    copy_host_to_device(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(compare_ge, b)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("compare_ge_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                        "clSetKernelArg(compare_ge, a)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                        "clSetKernelArg(compare_ge, b)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(compare_ge, out)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(compare_ge, size)");

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
                        "clEnqueueNDRangeKernel(compare_ge)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_ge)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_ge, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("compare_ge_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(compare_ge, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(compare_ge, b)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(compare_ge, out)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(compare_ge, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(compare_ge)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_ge)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("compare_ge", e.what());
        }
    }

    throw_opencl_only_failure("compare_ge", "OpenCL runtime unavailable or tensor shape mismatch");
}

OpenCLTensorBackend OpenCLTensorBackend::compare_eq(const OpenCLTensorBackend& other) const
{
    // Lazy-sync guard: a GPU-resident operand may hold stale host data
    // (m_needs_sync_to_host). This op reads host pointers, so pull the
    // device result down first (no-op when already in sync).
    sync_gpu_if_needed();
    other.sync_gpu_if_needed();
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("compare_eq", "OpenCL path requires matching tensor shapes");
    }
    // cppcheck-suppress knownConditionTrueFalse
    else if (can_use_opencl("compare_eq"))
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
                auto a_buf = pool->acquire(bytes);
                auto b_buf = pool->acquire(bytes);
                auto out_buf = pool->acquire(bytes);
                if (a_buf && b_buf && out_buf)
                {
                    copy_host_to_device(ctx.get_queue(),
                        a_buf->buffer,
                        m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(compare_eq, a)");
                    copy_host_to_device(ctx.get_queue(),
                        b_buf->buffer,
                        other.m_backend->data_ptr(),
                        bytes,
                        "clEnqueueWriteBuffer(compare_eq, b)");

                    cl_kernel kernel =
                        opencl::KernelManager::instance().get_kernel("compare_eq_kernel");
                    const cl_mem a_mem = a_buf->buffer;
                    const cl_mem b_mem = b_buf->buffer;
                    const cl_mem out_mem = out_buf->buffer;
                    const cl_uint n_u32 = static_cast<cl_uint>(n);

                    check_cl_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem),
                        "clSetKernelArg(compare_eq, a)");
                    check_cl_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem),
                        "clSetKernelArg(compare_eq, b)");
                    check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                        "clSetKernelArg(compare_eq, out)");
                    check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                        "clSetKernelArg(compare_eq, size)");

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
                        "clEnqueueNDRangeKernel(compare_eq)");
                    finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_eq)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_eq, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
                    return t;
                }
            }

            opencl::DeviceMemory a_dev(bytes);
            opencl::DeviceMemory b_dev(bytes);
            opencl::DeviceMemory out_dev(bytes);
            a_dev.copy_to_device(m_backend->data_ptr());
            b_dev.copy_to_device(other.m_backend->data_ptr());

            cl_kernel kernel = opencl::KernelManager::instance().get_kernel("compare_eq_kernel");
            const cl_mem a_mem = a_dev.get_device_buffer();
            const cl_mem b_mem = b_dev.get_device_buffer();
            const cl_mem out_mem = out_dev.get_device_buffer();
            const cl_uint n_u32 = static_cast<cl_uint>(n);

            check_cl_error(
                clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem), "clSetKernelArg(compare_eq, a)");
            check_cl_error(
                clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem), "clSetKernelArg(compare_eq, b)");
            check_cl_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem),
                "clSetKernelArg(compare_eq, out)");
            check_cl_error(clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_u32),
                "clSetKernelArg(compare_eq, size)");

            const std::size_t local = 256;
            std::size_t global = round_up(n, local);
            check_cl_error(
                clEnqueueNDRangeKernel(
                    ctx.get_queue(), kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(compare_eq)");
            finish_queue_if_not_batching(ctx.get_queue(), "clFinish(compare_eq)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<OpenCLHostStorage>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            throw_opencl_only_failure("compare_eq", e.what());
        }
    }

    throw_opencl_only_failure("compare_eq", "OpenCL runtime unavailable or tensor shape mismatch");
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
                        m_backend->data_ptr(),
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
            input_dev.copy_to_device(m_backend->data_ptr());

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
                        m_backend->data_ptr(),
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
            input_dev.copy_to_device(m_backend->data_ptr());

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
                        m_backend->data_ptr(),
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
            input_dev.copy_to_device(m_backend->data_ptr());

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
                        m_backend->data_ptr(),
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
            input_dev.copy_to_device(m_backend->data_ptr());

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
                        m_backend->data_ptr(),
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
            input_dev.copy_to_device(m_backend->data_ptr());

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
        copy_host_to_device(ctx.get_queue(),
            m_gpu_buffer->buffer,
            m_backend->data_ptr(),
            size() * sizeof(float),
            what);
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
