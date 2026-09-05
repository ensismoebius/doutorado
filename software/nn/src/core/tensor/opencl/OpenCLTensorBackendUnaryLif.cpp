/**
 * @file src/core/tensor/opencl/OpenCLTensorBackendUnaryLif.cpp
 * @brief Unary elementwise kernels (abs, relu, leaky_relu, clamp) and the LIF spiking-neuron
 * step/gradient kernels.
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

} // namespace nn
