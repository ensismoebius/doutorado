/**
 * @file src/core/tensor/opencl/OpenCLTensorBackend.cpp
 * @brief OpenCL-only tensor backend implementation.
 *
 * Tensor metadata and host synchronization staging are managed locally,
 * while math operations execute through OpenCL kernels only.
 */

#include "nn/tensor/opencl/OpenCLTensorBackend.hpp"

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

#include "nn/logging/Logger.hpp"
#include "nn/tensor/opencl/DeviceMemory.hpp"
#include "nn/tensor/opencl/KernelManager.hpp"
#include "nn/tensor/opencl/OpenCLContext.hpp"
#include "nn/tensor/opencl/OpenCLProfiling.hpp"

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
        const Index col_idx = d2 * m_shape[2] + d3;
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

        const Index col_idx = (d2 * (m_shape[2] * m_shape[3])) + (d3 * m_shape[3]) + d4;
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

auto to_opencl_tensor(const nn::Tensor& source) -> nn::OpenCLTensorBackend
{
    nn::OpenCLTensorBackend result(source.get_shape());
    for (Index i = 0; i < source.size(); ++i)
    {
        result.at(i) = source.at(i);
    }
    return result;
}

auto compute_opencl_reconstruction_mse(const nn::Tensor& prediction, const nn::Tensor& target)
    -> float
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
    other.sync_gpu();
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
        other.sync_gpu();
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
    sync_gpu();
    m_needs_sync_to_device = true;
    return m_backend->at(i);
}

const float& OpenCLTensorBackend::at(Index i) const
{
    sync_gpu();
    return m_backend->at(i);
}

float& OpenCLTensorBackend::at(Index row, Index col)
{
    sync_gpu();
    m_needs_sync_to_device = true;
    return m_backend->at(row, col);
}

const float& OpenCLTensorBackend::at(Index row, Index col) const
{
    sync_gpu();
    return m_backend->at(row, col);
}

float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3)
{
    sync_gpu();
    m_needs_sync_to_device = true;
    return m_backend->at(d1, d2, d3);
}

const float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3) const
{
    sync_gpu();
    return m_backend->at(d1, d2, d3);
}

float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3, Index d4)
{
    sync_gpu();
    m_needs_sync_to_device = true;
    return m_backend->at(d1, d2, d3, d4);
}

const float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3, Index d4) const
{
    sync_gpu();
    return m_backend->at(d1, d2, d3, d4);
}

float& OpenCLTensorBackend::at(const std::vector<Index>& indices)
{
    sync_gpu();
    m_needs_sync_to_device = true;
    return m_backend->at(indices);
}

const float& OpenCLTensorBackend::at(const std::vector<Index>& indices) const
{
    sync_gpu();
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

// In-place operations
void OpenCLTensorBackend::add_inplace(const OpenCLTensorBackend& other)
{
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
    m_backend->fill(value);
    m_needs_sync_to_device = true;
    m_needs_sync_to_host = false;
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

            if (m_gpu_resident && m_has_gpu_memory && m_gpu_buffer && col_vector.m_has_gpu_memory &&
                col_vector.m_gpu_buffer)
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

            if (m_gpu_resident && m_has_gpu_memory && pool)
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

            if (m_gpu_resident && m_has_gpu_memory && pool)
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
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("add", "OpenCL path requires matching tensor shapes");
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
                    if (m_gpu_resident)
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
                    if (m_gpu_resident)
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
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("multiply", "OpenCL path requires matching tensor shapes");
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
                    if (m_gpu_resident)
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
                    if (m_gpu_resident)
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

            if (m_gpu_resident && m_has_gpu_memory && m_gpu_buffer)
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

            if (m_gpu_resident && other.m_gpu_resident && m_has_gpu_memory &&
                other.m_has_gpu_memory && m_gpu_buffer && other.m_gpu_buffer)
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

            if (m_gpu_resident && other.m_gpu_resident && m_has_gpu_memory &&
                other.m_has_gpu_memory && m_gpu_buffer && other.m_gpu_buffer)
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

            if (m_gpu_resident && other.m_gpu_resident && m_has_gpu_memory &&
                other.m_has_gpu_memory && m_gpu_buffer && other.m_gpu_buffer)
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

        if (m_gpu_resident && other.m_gpu_resident && bias.m_gpu_resident && m_has_gpu_memory &&
            other.m_has_gpu_memory && bias.m_has_gpu_memory && m_gpu_buffer && other.m_gpu_buffer &&
            bias.m_gpu_buffer)
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

OpenCLTensorBackend OpenCLTensorBackend::transpose() const
{
    if (shape().size() != 2)
    {
        warn_opencl_cpu_fallback_once("transpose", "OpenCL path requires a rank-2 tensor");
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
                    if (m_gpu_resident)
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
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("compare_lt", "OpenCL path requires matching tensor shapes");
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

void OpenCLTensorBackend::verify_runtime_activity_or_throw(
    const Tensor& prediction, const Tensor& target, std::string_view gpu_busy_percent_path)
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
