/**
 * @file src/core/tensor/opencl/OpenCLTensorBackendRuntime.cpp
 * @brief Runtime lifecycle (initialize/verify/shutdown), the GPU buffer pool, Adam step, and the
 * resident-launch helper family (launch_binary_resident and friends).
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

namespace
{

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
