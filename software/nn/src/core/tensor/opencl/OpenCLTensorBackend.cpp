/**
 * @file src/core/tensor/opencl/OpenCLTensorBackend.cpp
 * @brief OpenCL tensor backend implementation (Phase 1: CPU fallback).
 *
 * Phase 1 delegates all operations to EigenTensorBackend for correctness.
 * GPU implementations will be added incrementally as needed.
 */

#include "nn/tensor/opencl/OpenCLTensorBackend.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>

#include "nn/logging/Logger.hpp"
#include "nn/tensor/eigen/EigenTensorBackend.hpp"
#include "nn/tensor/opencl/DeviceMemory.hpp"
#include "nn/tensor/opencl/KernelManager.hpp"
#include "nn/tensor/opencl/OpenCLContext.hpp"
#include "nn/tensor/opencl/OpenCLProfiling.hpp"

namespace nn
{

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

class AsyncTransferManager
{
   public:
    static AsyncTransferManager& instance()
    {
        static AsyncTransferManager mgr;
        return mgr;
    }

    cl_event copy_host_to_device_async(
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

    cl_event copy_device_to_host_async(
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

    void enqueue_kernel(cl_command_queue queue,
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
            check_cl_error(clEnqueueWaitForEvents(queue, 1, &wait_evt), "wait_for_event");
            clReleaseEvent(evt);
        }
    }

    void wait_and_release_events(cl_command_queue queue, cl_event* events, cl_uint num_events)
    {
        if (num_events > 0 && events)
        {
            check_cl_error(
                clEnqueueWaitForEvents(queue, num_events, events), "wait_and_release_events");
            for (cl_uint i = 0; i < num_events; ++i)
            {
                if (events[i]) clReleaseEvent(events[i]);
            }
        }
    }

    void sync_queue(cl_command_queue queue)
    {
        check_cl_error(clFinish(queue), "sync_queue");
    }

   private:
    AsyncTransferManager() = default;
};

bool can_use_opencl()
{
    // OpenCL runtime is intentionally disabled under ASan to avoid false positives
    // from third-party drivers while preserving CPU fallback semantics.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
    return false;
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
    return false;
#endif
    return opencl::OpenCLContext::instance().is_available();
}

void warn_opencl_cpu_fallback_once(const std::string& operation, const std::string& reason)
{
    static std::mutex warned_mutex;
    static std::unordered_set<std::string> warned_messages;

    const std::string message =
        "OPENCL BACKEND FALLING BACK TO CPU for " + operation + ": " + reason;
    std::lock_guard<std::mutex> lock(warned_mutex);
    if (warned_messages.insert(message).second)
    {
        NN_LOG_WARN(message);
    }
}

bool can_use_opencl(const char* operation)
{
#if defined(__SANITIZE_ADDRESS__)
    warn_opencl_cpu_fallback_once(operation, "AddressSanitizer build disables OpenCL execution");
    return false;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
    warn_opencl_cpu_fallback_once(operation, "AddressSanitizer build disables OpenCL execution");
    return false;
#endif
    if (!can_use_opencl())
    {
        warn_opencl_cpu_fallback_once(operation, "OpenCL runtime or device is not available");
        return false;
    }
    return true;
#else
    if (!can_use_opencl())
    {
        warn_opencl_cpu_fallback_once(operation, "OpenCL runtime or device is not available");
        return false;
    }
    return true;
#endif
}

void warn_opencl_unimplemented_once(const char* operation)
{
    warn_opencl_cpu_fallback_once(
        operation, "operation is not implemented on the OpenCL backend yet");
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
    std::string context;
};

class EventTracker
{
   public:
    static EventTracker& instance()
    {
        static EventTracker tracker;
        return tracker;
    }

    void add_pending_event(cl_event evt, std::string context)
    {
        if (evt)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_pending_events.push_back({evt, std::move(context)});
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
            for (auto& pe : events)
            {
                cl_events.push_back(pe.event);
            }
            check_cl_error(clEnqueueWaitForEvents(
                               queue, static_cast<cl_uint>(cl_events.size()), cl_events.data()),
                "flush_all");
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

inline void enqueue_kernel_with_event(cl_command_queue queue,
    cl_kernel kernel,
    std::size_t global_size,
    std::size_t local_size,
    cl_event wait_event,
    cl_event* out_event)
{
    if (wait_event)
    {
        check_cl_error(
            clEnqueueNDRangeKernel(
                queue, kernel, 1, nullptr, &global_size, &local_size, 1, &wait_event, out_event),
            "enqueue_kernel_with_event");
    }
    else
    {
        check_cl_error(
            clEnqueueNDRangeKernel(
                queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, out_event),
            "enqueue_kernel_with_event");
    }
}

void flush_pending_events(cl_command_queue queue)
{
    EventTracker::instance().flush_all(queue);
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
    : m_backend(std::make_unique<EigenTensorBackend>(rows, cols))
{
    try_allocate_gpu_buffer(rows * cols);
}

OpenCLTensorBackend::OpenCLTensorBackend(Index d1, Index d2, Index d3, Index d4)
    : m_backend(std::make_unique<EigenTensorBackend>(d1, d2, d3, d4))
{
    try_allocate_gpu_buffer(d1 * d2 * d3 * d4);
}

OpenCLTensorBackend::OpenCLTensorBackend(const std::vector<Index>& shape)
    : m_backend(std::make_unique<EigenTensorBackend>(shape))
{
    Index total = 1;
    for (auto dim : shape) total *= dim;
    try_allocate_gpu_buffer(total);
}

OpenCLTensorBackend::OpenCLTensorBackend(const OpenCLTensorBackend& other)
{
    if (other.m_backend)
    {
        m_backend = std::make_unique<EigenTensorBackend>(*other.m_backend);
    }
    if (other.m_grad_backend)
    {
        m_grad_backend = std::make_unique<OpenCLTensorBackend>(*other.m_grad_backend);
    }
}

OpenCLTensorBackend& OpenCLTensorBackend::operator=(const OpenCLTensorBackend& other)
{
    if (this != &other)
    {
        m_backend =
            other.m_backend ? std::make_unique<EigenTensorBackend>(*other.m_backend) : nullptr;
        m_grad_backend = other.m_grad_backend
                             ? std::make_unique<OpenCLTensorBackend>(*other.m_grad_backend)
                             : nullptr;
    }
    return *this;
}

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
    *t.m_backend = EigenTensorBackend::zeros(rows, cols);
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::ones(Index rows, Index cols)
{
    OpenCLTensorBackend t(rows, cols);
    *t.m_backend = EigenTensorBackend::ones(rows, cols);
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::random(Index rows, Index cols)
{
    OpenCLTensorBackend t(rows, cols);
    *t.m_backend = EigenTensorBackend::random(rows, cols);
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::random(Index rows, Index cols, std::mt19937& rng)
{
    OpenCLTensorBackend t(rows, cols);
    *t.m_backend = EigenTensorBackend::random(rows, cols, rng);
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
    return m_backend->at(i);
}

const float& OpenCLTensorBackend::at(Index i) const
{
    return m_backend->at(i);
}

float& OpenCLTensorBackend::at(Index row, Index col)
{
    return m_backend->at(row, col);
}

const float& OpenCLTensorBackend::at(Index row, Index col) const
{
    return m_backend->at(row, col);
}

float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3, Index d4)
{
    return m_backend->at(d1, d2, d3, d4);
}

const float& OpenCLTensorBackend::at(Index d1, Index d2, Index d3, Index d4) const
{
    return m_backend->at(d1, d2, d3, d4);
}

float& OpenCLTensorBackend::at(const std::vector<Index>& indices)
{
    return m_backend->at(indices);
}

const float& OpenCLTensorBackend::at(const std::vector<Index>& indices) const
{
    return m_backend->at(indices);
}

// In-place operations
void OpenCLTensorBackend::add_inplace(const OpenCLTensorBackend& other)
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("add_inplace", "OpenCL path requires matching tensor shapes");
    }
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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());
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
            check_cl_error(clFinish(ctx.get_queue()), "add_inplace");

            a_dev.copy_from_device(m_backend->mutable_data_ptr());
            return;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL add_inplace fallback to CPU: ") + e.what());
        }
    }
    m_backend->add_inplace(*other.m_backend);
}

void OpenCLTensorBackend::subtract_inplace(const OpenCLTensorBackend& other)
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once(
            "subtract_inplace", "OpenCL path requires matching tensor shapes");
    }
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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());
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
            check_cl_error(clFinish(ctx.get_queue()), "subtract_inplace");

            a_dev.copy_from_device(m_backend->mutable_data_ptr());
            return;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL subtract_inplace fallback to CPU: ") + e.what());
        }
    }
    m_backend->subtract_inplace(*other.m_backend);
}

void OpenCLTensorBackend::multiply_inplace(const OpenCLTensorBackend& other)
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once(
            "multiply_inplace", "OpenCL path requires matching tensor shapes");
    }
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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());
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
            check_cl_error(clFinish(ctx.get_queue()), "multiply_inplace");

            a_dev.copy_from_device(m_backend->mutable_data_ptr());
            return;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL multiply_inplace fallback to CPU: ") + e.what());
        }
    }
    m_backend->multiply_inplace(*other.m_backend);
}

void OpenCLTensorBackend::divide_inplace(const OpenCLTensorBackend& other)
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once(
            "divide_inplace", "OpenCL path requires matching tensor shapes");
    }
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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());
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
            check_cl_error(clFinish(ctx.get_queue()), "divide_inplace");

            a_dev.copy_from_device(m_backend->mutable_data_ptr());
            return;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL divide_inplace fallback to CPU: ") + e.what());
        }
    }
    m_backend->divide_inplace(*other.m_backend);
}

void OpenCLTensorBackend::add_scalar_inplace(float val)
{
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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());
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
            check_cl_error(clFinish(ctx.get_queue()), "add_scalar_inplace");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            return;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL add_scalar_inplace fallback to CPU: ") + e.what());
        }
    }
    m_backend->add_scalar_inplace(val);
}

void OpenCLTensorBackend::multiply_scalar_inplace(float val)
{
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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());
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
            check_cl_error(clFinish(ctx.get_queue()), "multiply_scalar_inplace");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            return;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL multiply_scalar_inplace fallback to CPU: ") + e.what());
        }
    }
    m_backend->multiply_scalar_inplace(val);
}

void OpenCLTensorBackend::divide_scalar_inplace(float val)
{
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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());
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
            check_cl_error(clFinish(ctx.get_queue()), "divide_scalar_inplace");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            return;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL divide_scalar_inplace fallback to CPU: ") + e.what());
        }
    }
    m_backend->divide_scalar_inplace(val);
}

void OpenCLTensorBackend::sqrt_inplace()
{
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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());
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
            check_cl_error(clFinish(ctx.get_queue()), "sqrt_inplace");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            return;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL sqrt_inplace fallback to CPU: ") + e.what());
        }
    }
    m_backend->sqrt_inplace();
}

void OpenCLTensorBackend::square_inplace()
{
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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());
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
            check_cl_error(clFinish(ctx.get_queue()), "square_inplace");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            return;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL square_inplace fallback to CPU: ") + e.what());
        }
    }
    m_backend->square_inplace();
}

void OpenCLTensorBackend::add_col_vector_to_rows_inplace(const OpenCLTensorBackend& col_vector)
{
    if (shape().size() != 2 || col_vector.shape().size() != 2)
    {
        warn_opencl_cpu_fallback_once(
            "add_col_vector_to_rows_inplace", "OpenCL path requires rank-2 tensors");
    }
    else if (rows() != col_vector.rows() || col_vector.cols() != 1)
    {
        warn_opencl_cpu_fallback_once(
            "add_col_vector_to_rows_inplace", "OpenCL path requires col_vector to be (rows x 1)");
    }
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
            const std::size_t col_bytes = num_rows * sizeof(float);

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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());
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
            check_cl_error(clFinish(ctx.get_queue()), "add_col_vector_to_rows_inplace");

            data_dev.copy_from_device(m_backend->mutable_data_ptr());
            return;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(
                std::string("OpenCL add_col_vector_to_rows_inplace fallback to CPU: ") + e.what());
        }
    }
    m_backend->add_col_vector_to_rows_inplace(*col_vector.m_backend);
}

// Element-wise operations
OpenCLTensorBackend OpenCLTensorBackend::exp() const
{
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
            EigenTensorBackend out(shape());

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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(exp)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL exp fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->exp());
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::sqrt() const
{
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
            EigenTensorBackend out(shape());

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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(sqrt)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL sqrt fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->sqrt());
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::square() const
{
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
            EigenTensorBackend out(shape());

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
                    if (d2h_evt) clReleaseEvent(d2h_evt);

                    flush_pending_events(ctx.get_queue());

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(square)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL square fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->square());
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::add(const OpenCLTensorBackend& other) const
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("add", "OpenCL path requires matching tensor shapes");
    }
    else if (can_use_opencl("add"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(add)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(add, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(add)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL add fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->add(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::subtract(const OpenCLTensorBackend& other) const
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("subtract", "OpenCL path requires matching tensor shapes");
    }
    else if (can_use_opencl("subtract"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(subtract)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(subtract, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(subtract)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL subtract fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->subtract(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::multiply(const OpenCLTensorBackend& other) const
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("multiply", "OpenCL path requires matching tensor shapes");
    }
    else if (can_use_opencl("multiply"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(multiply)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(multiply, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(multiply)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL multiply fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->multiply(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::divide(const OpenCLTensorBackend& other) const
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("divide", "OpenCL path requires matching tensor shapes");
    }
    else if (can_use_opencl("divide"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(divide)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(divide, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(divide)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL divide fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->divide(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::add_scalar(float val) const
{
    if (can_use_opencl("add_scalar"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(add_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(add_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(add_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL add_scalar fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->add_scalar(val));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::multiply_scalar(float val) const
{
    if (can_use_opencl("multiply_scalar"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(multiply_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(multiply_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(multiply_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL multiply_scalar fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->multiply_scalar(val));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::divide_scalar(float val) const
{
    if (can_use_opencl("divide_scalar"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(divide_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(divide_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(divide_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL divide_scalar fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->divide_scalar(val));
    return t;
}

// Reduction
OpenCLTensorBackend OpenCLTensorBackend::rowwise_sum() const
{
    if (shape().size() != 2)
    {
        warn_opencl_cpu_fallback_once("rowwise_sum", "OpenCL path requires rank-2 tensors");
    }
    else if (can_use_opencl("rowwise_sum"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const Index num_rows = rows();
            const Index num_cols = cols();
            const std::size_t input_bytes = num_rows * num_cols * sizeof(float);
            const std::size_t output_bytes = num_rows * sizeof(float);

            EigenTensorBackend out(num_rows, 1);
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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(rowwise_sum)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        output_bytes,
                        "clEnqueueReadBuffer(rowwise_sum, output)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(rowwise_sum)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL rowwise_sum fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->rowwise_sum());
    return t;
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
            EigenTensorBackend out(m, n);

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(matmul)");

                    copy_device_to_host(ctx.get_queue(),
                        c_buf->buffer,
                        out.mutable_data_ptr(),
                        c_bytes,
                        "clEnqueueReadBuffer(matmul, c)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(matmul)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL matmul fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->matmul(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::matmul_transposed(const OpenCLTensorBackend& other) const
{
    warn_opencl_unimplemented_once("matmul_transposed");
    OpenCLTensorBackend t;
    t.m_backend =
        std::make_unique<EigenTensorBackend>(m_backend->matmul_transposed(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::transpose() const
{
    if (shape().size() != 2)
    {
        warn_opencl_cpu_fallback_once("transpose", "OpenCL path requires a rank-2 tensor");
    }
    else if (can_use_opencl("transpose"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const Index in_rows = rows();
            const Index in_cols = cols();
            const std::size_t bytes = in_rows * in_cols * sizeof(float);

            EigenTensorBackend out(in_cols, in_rows);
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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(transpose)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(transpose, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(transpose)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL transpose fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->transpose());
    return t;
}

// Comparisons
OpenCLTensorBackend OpenCLTensorBackend::compare_lt(const OpenCLTensorBackend& other) const
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("compare_lt", "OpenCL path requires matching tensor shapes");
    }
    else if (can_use_opencl("compare_lt"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_lt)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_lt, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_lt)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL compare_lt fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_lt(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_gt(const OpenCLTensorBackend& other) const
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("compare_gt", "OpenCL path requires matching tensor shapes");
    }
    else if (can_use_opencl("compare_gt"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_gt)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_gt, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_gt)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL compare_gt fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_gt(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_le(const OpenCLTensorBackend& other) const
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("compare_le", "OpenCL path requires matching tensor shapes");
    }
    else if (can_use_opencl("compare_le"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_le)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_le, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_le)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL compare_le fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_le(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_ge(const OpenCLTensorBackend& other) const
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("compare_ge", "OpenCL path requires matching tensor shapes");
    }
    else if (can_use_opencl("compare_ge"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_ge)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_ge, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_ge)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL compare_ge fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_ge(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_eq(const OpenCLTensorBackend& other) const
{
    if (shape() != other.shape())
    {
        warn_opencl_cpu_fallback_once("compare_eq", "OpenCL path requires matching tensor shapes");
    }
    else if (can_use_opencl("compare_eq"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_eq)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_eq, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_eq)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL compare_eq fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_eq(*other.m_backend));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_lt_scalar(float value) const
{
    if (can_use_opencl("compare_lt_scalar"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_lt_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_lt_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_lt_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL compare_lt_scalar fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_lt_scalar(value));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_gt_scalar(float value) const
{
    if (can_use_opencl("compare_gt_scalar"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_gt_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_gt_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_gt_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL compare_gt_scalar fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_gt_scalar(value));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_le_scalar(float value) const
{
    if (can_use_opencl("compare_le_scalar"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_le_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_le_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_le_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL compare_le_scalar fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_le_scalar(value));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_ge_scalar(float value) const
{
    if (can_use_opencl("compare_ge_scalar"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_ge_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_ge_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_ge_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL compare_ge_scalar fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_ge_scalar(value));
    return t;
}

OpenCLTensorBackend OpenCLTensorBackend::compare_eq_scalar(float value) const
{
    if (can_use_opencl("compare_eq_scalar"))
    {
        try
        {
            const auto& ctx = opencl::OpenCLContext::instance();
            const auto n = size();
            const std::size_t bytes = n * sizeof(float);
            EigenTensorBackend out(shape());

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
                    check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_eq_scalar)");

                    copy_device_to_host(ctx.get_queue(),
                        out_buf->buffer,
                        out.mutable_data_ptr(),
                        bytes,
                        "clEnqueueReadBuffer(compare_eq_scalar, out)");

                    OpenCLTensorBackend t;
                    t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
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
            check_cl_error(clFinish(ctx.get_queue()), "clFinish(compare_eq_scalar)");

            out_dev.copy_from_device(out.mutable_data_ptr());

            OpenCLTensorBackend t;
            t.m_backend = std::make_unique<EigenTensorBackend>(std::move(out));
            return t;
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("OpenCL compare_eq_scalar fallback to CPU: ") + e.what());
        }
    }

    OpenCLTensorBackend t;
    t.m_backend = std::make_unique<EigenTensorBackend>(m_backend->compare_eq_scalar(value));
    return t;
}

// Gradient management
OpenCLTensorBackend& OpenCLTensorBackend::grad_ref()
{
    if (!m_grad_backend)
    {
        m_grad_backend = std::make_unique<OpenCLTensorBackend>(shape());
        m_grad_backend->m_backend = std::make_unique<EigenTensorBackend>(shape());
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
        *m_grad_backend->m_backend = EigenTensorBackend::zeros(rows(), cols());
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
        auto& ctx = opencl::OpenCLContext::instance();
        AsyncTransferManager::instance().wait_and_release_events(
            ctx.get_queue(), m_pending_events, static_cast<cl_uint>(m_pending_events_count));
        m_pending_events_count = 0;
    }
}

void OpenCLTensorBackend::try_allocate_gpu_buffer(Index size)
{
    if (size == 0) return;
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
