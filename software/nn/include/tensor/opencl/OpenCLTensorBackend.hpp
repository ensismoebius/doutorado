/**
 * @file include/tensor/opencl/OpenCLTensorBackend.hpp
 * @brief OpenCL-only implementation of the tensor backend.
 *
 * **API contract:**
 * - Same public interface as the CPU tensor backend
 * - Move/copy construction and assignment supported
 * - Lazy gradient allocation (on grad_ref access)
 * - OpenCL kernels are the only execution path for tensor operations
 */

#ifndef OPENCL_TENSOR_BACKEND_HPP
#define OPENCL_TENSOR_BACKEND_HPP

#include <memory>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "device/Device.hpp"
#include "tensor/opencl/GPUBufferPool.hpp"
namespace nn
{

using Index = std::size_t;

class OpenCLHostStorage;

/**
 * @brief OpenCL tensor backend with OpenCL-owned execution and host staging storage.
 *
 * Host-side tensor metadata and synchronization staging are managed locally,
 * but all math operations execute through OpenCL kernels only.
 */
class OpenCLTensorBackend
{
   public:
    /**
     * @brief RAII scope that keeps OpenCL runtime resources active for a caller scope.
     *
     * On destruction, performs backend-owned runtime shutdown.
     */
    struct RuntimeScope
    {
        std::string device_name;
        bool active = false;

        RuntimeScope() = default;
        ~RuntimeScope();

        RuntimeScope(const RuntimeScope&) = delete;
        RuntimeScope& operator=(const RuntimeScope&) = delete;
        RuntimeScope(RuntimeScope&& other) noexcept;
        RuntimeScope& operator=(RuntimeScope&& other) noexcept;
    };

    // -----------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------
    OpenCLTensorBackend();

    explicit OpenCLTensorBackend(Index rows, Index cols);
    explicit OpenCLTensorBackend(Index d1, Index d2, Index d3);
    explicit OpenCLTensorBackend(Index d1, Index d2, Index d3, Index d4);
    explicit OpenCLTensorBackend(const std::vector<Index>& shape);

    // Construct from shape on a specific device (GPU)
    // Initializes GPU memory directly if OpenCL is available
    explicit OpenCLTensorBackend(const std::vector<Index>& shape, const Device& device);
    explicit OpenCLTensorBackend(Index rows, Index cols, const Device& device);

    // Copy/Move construction
    OpenCLTensorBackend(const OpenCLTensorBackend& other);
    OpenCLTensorBackend(OpenCLTensorBackend&& other) noexcept;

    // Copy/Move assignment
    OpenCLTensorBackend& operator=(const OpenCLTensorBackend& other);
    OpenCLTensorBackend& operator=(OpenCLTensorBackend&& other) noexcept;

    ~OpenCLTensorBackend();

    // -----------------------------------------------------------------
    // Static Factories
    // -----------------------------------------------------------------
    static OpenCLTensorBackend zeros(Index rows, Index cols);
    static OpenCLTensorBackend ones(Index rows, Index cols);
    static OpenCLTensorBackend random(Index rows, Index cols);
    static OpenCLTensorBackend random(Index rows, Index cols, std::mt19937& rng);
    static OpenCLTensorBackend random(Index d1, Index d2, Index d3);
    static OpenCLTensorBackend random(Index d1, Index d2, Index d3, std::mt19937& rng);

    // -----------------------------------------------------------------
    // Shape & Access
    // -----------------------------------------------------------------
    const std::vector<Index>& shape() const;
    void reshape(const std::vector<Index>& new_shape);

    Index rows() const;
    Index cols() const;
    Index size() const;

    // N-D access operators (requires synchronization from pending OpenCL work)
    float& at(Index i);
    const float& at(Index i) const;
    float& at(Index row, Index col);
    const float& at(Index row, Index col) const;
    float& at(Index d1, Index d2, Index d3);
    const float& at(Index d1, Index d2, Index d3) const;
    float& at(Index d1, Index d2, Index d3, Index d4);
    const float& at(Index d1, Index d2, Index d3, Index d4) const;
    float& at(const std::vector<Index>& indices);
    const float& at(const std::vector<Index>& indices) const;

    float* mutable_data_ptr();
    const float* data_ptr() const;

    // -----------------------------------------------------------------
    // Views / Slicing
    // -----------------------------------------------------------------
    OpenCLTensorBackend row(Index i) const;
    OpenCLTensorBackend col(Index j) const;
    OpenCLTensorBackend leftCols(Index n) const;
    OpenCLTensorBackend topRows(Index n) const;
    void setBlock(Index row, Index col, const OpenCLTensorBackend& block);
    OpenCLTensorBackend slice(std::span<const int> indices) const;
    OpenCLTensorBackend slice_batch(Index b) const;
    void set_batch_slice(Index b, const OpenCLTensorBackend& val);
    OpenCLTensorBackend slice_time(Index t) const;
    void set_time_slice(Index t, const OpenCLTensorBackend& val);

    // -----------------------------------------------------------------
    // In-place Operations
    // -----------------------------------------------------------------
    void add_inplace(const OpenCLTensorBackend& other);
    void subtract_inplace(const OpenCLTensorBackend& other);
    void multiply_inplace(const OpenCLTensorBackend& other);
    void divide_inplace(const OpenCLTensorBackend& other);
    void add_scalar_inplace(float val);
    void multiply_scalar_inplace(float val);
    void divide_scalar_inplace(float val);
    void set_zero();
    void set_ones();
    void fill(float value);
    void sqrt_inplace();
    void square_inplace();
    void add_row_broadcast_inplace(const OpenCLTensorBackend& row);
    void add_col_vector_to_rows_inplace(const OpenCLTensorBackend& col_vector);

    // -----------------------------------------------------------------
    // Element-wise Operations (const, create new tensor)
    // -----------------------------------------------------------------
    OpenCLTensorBackend exp() const;
    OpenCLTensorBackend sqrt() const;
    OpenCLTensorBackend square() const;
    OpenCLTensorBackend abs() const;
    OpenCLTensorBackend relu() const;
    OpenCLTensorBackend leaky_relu(float alpha) const;

    // OpenCL-only spiking helpers used by LIF layers.
    void lif_step_inplace(const OpenCLTensorBackend& input,
        OpenCLTensorBackend& output,
        OpenCLTensorBackend* adapt_a,
        float beta,
        float threshold,
        float reset_potential,
        bool reset_zero,
        float adapt_decay,
        float adapt_coupling,
        bool use_adaptation);
    OpenCLTensorBackend lif_grad(float threshold, float sharpness) const;

    // Fused Adam update: param/moment1/moment2 updated in place from grad in a
    // single kernel (adam_step_kernel) instead of ~15 elementwise ops with
    // intermediate tensors. Returns false when the fast path can't run
    // (OpenCL unavailable / buffer allocation failed) — caller falls back to
    // the generic tensor-op implementation. Decoupled weight decay is NOT
    // applied here; the caller layers it on top exactly as in the generic path.
    bool adam_step_inplace(OpenCLTensorBackend& moment1,
        OpenCLTensorBackend& moment2,
        const OpenCLTensorBackend& grad,
        float lr,
        float beta1,
        float beta2,
        float epsilon,
        float bias_correction1,
        float bias_correction2);

    OpenCLTensorBackend add(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend subtract(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend multiply(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend divide(const OpenCLTensorBackend& other) const;

    OpenCLTensorBackend add_scalar(float val) const;
    OpenCLTensorBackend multiply_scalar(float val) const;
    OpenCLTensorBackend divide_scalar(float val) const;
    OpenCLTensorBackend add_row_broadcast(const OpenCLTensorBackend& row) const;

    // -----------------------------------------------------------------
    // Reduction Operations
    // -----------------------------------------------------------------
    OpenCLTensorBackend rowwise_sum() const;
    OpenCLTensorBackend sum_rows() const;
    OpenCLTensorBackend sum_cols() const;
    float mean_squared_error(const OpenCLTensorBackend& target) const;
    float mean() const;
    float norm() const;
    float sum() const;
    bool hasNaN() const;
    bool operator==(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend clamp(float min_val, float max_val) const;
    void clamp_inplace(float min_val, float max_val);

    // -----------------------------------------------------------------
    // Linear Algebra
    // -----------------------------------------------------------------
    OpenCLTensorBackend matmul(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend matmul_lhs_transposed(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend matmul_transposed(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend matmul_transposed_add_col_bias(
        const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias) const;
    OpenCLTensorBackend matmul_transposed_add_col_bias_relu(
        const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias) const;
    OpenCLTensorBackend matmul_transposed_add_col_bias_leaky_relu(
        const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias, float alpha) const;
    OpenCLTensorBackend matmul_transposed_add_col_bias_sigmoid(
        const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias) const;
    OpenCLTensorBackend matmul_transposed_add_col_bias_tanh(
        const OpenCLTensorBackend& other, const OpenCLTensorBackend& bias) const;
    OpenCLTensorBackend transpose() const;
    OpenCLTensorBackend block(Index row, Index col, Index rows, Index cols) const;

    // -----------------------------------------------------------------
    // Comparisons
    // -----------------------------------------------------------------
    OpenCLTensorBackend compare_lt(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend compare_gt(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend compare_le(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend compare_ge(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend compare_eq(const OpenCLTensorBackend& other) const;

    OpenCLTensorBackend compare_lt_scalar(float value) const;
    OpenCLTensorBackend compare_gt_scalar(float value) const;
    OpenCLTensorBackend compare_le_scalar(float value) const;
    OpenCLTensorBackend compare_ge_scalar(float value) const;
    OpenCLTensorBackend compare_eq_scalar(float value) const;

    // -----------------------------------------------------------------
    // Gradient Management
    // -----------------------------------------------------------------
    /**
     * @brief Get a reference to the gradient backend (lazy allocation).
     *
     * First call allocates gradient storage; subsequent calls return existing.
     */
    auto grad_ref() -> OpenCLTensorBackend&;

    /**
     * @brief Check if gradient has been allocated.
     */
    auto has_grad() const -> bool
    {
        return m_grad_backend != nullptr;
    }

    /**
     * @brief Get gradient; returns a zero-initialised backend if no gradient has been allocated.
     */
    auto get_grad() const -> OpenCLTensorBackend;

    /**
     * @brief Replace stored gradient backend with a copy of the provided tensor.
     */
    void set_grad(const OpenCLTensorBackend& grad);

    /**
     * @brief Zero out all accumulated gradients.
     */
    void zero_grad();

    /**
     * @brief Initialize OpenCL backend runtime facilities.
     *
     * Performs backend-owned startup checks (sanitizer/runtime availability),
     * initializes the shared GPU buffer pool, configures profiling, and
     * returns the active OpenCL device name.
     *
     * @param opencl_profiling_enabled Whether event profiling should be enabled.
     * @return Device name reported by OpenCL runtime.
     * @throws std::runtime_error when OpenCL cannot be used.
     */
    static std::string initialize_runtime_or_throw(bool opencl_profiling_enabled);

    /**
     * @brief Start backend runtime and return an RAII scope for shutdown.
     *
     * This is the preferred lifecycle API for callers that need deterministic
     * runtime setup/teardown handled by the backend itself.
     */
    static RuntimeScope start_runtime_scope_or_throw(bool opencl_profiling_enabled);

    /**
     * @brief Verify that OpenCL execution is effectively active for a representative workload.
     *
     * Runs a small reconstruction-MSE probe fully through the OpenCL backend and, when
     * available, checks GPU busy percentage growth from a sysfs node.
     *
     * @param prediction Reconstruction tensor produced by the model.
     * @param target Ground-truth tensor used as reconstruction target.
     * @param gpu_busy_percent_path Sysfs path for gpu_busy_percent probing.
     * @throws std::runtime_error when OpenCL runtime is unavailable or probe indicates no activity.
     */
    static void verify_runtime_activity_or_throw(const OpenCLTensorBackend& prediction,
        const OpenCLTensorBackend& target,
        std::string_view gpu_busy_percent_path = "/sys/class/drm/card1/device/gpu_busy_percent");

    /**
     * @brief Initialize the static GPU buffer pool (call once at app startup).
     * @param context OpenCL context
     * @param queue OpenCL command queue
     */
    static void init_buffer_pool(void* context, void* queue);

    /**
     * @brief Shutdown the static GPU buffer pool (call at app shutdown).
     */
    static void shutdown_buffer_pool();

    /**
     * @brief Get the static buffer pool instance (nullptr if not initialized).
     */
    static tensor::GPUBufferPool* get_buffer_pool();

    // Check if this tensor has GPU memory allocated
    bool has_gpu_memory() const
    {
        return m_has_gpu_memory;
    }

    // Get the GPU buffer if available (for internal use)
    tensor::GPUBuffer* gpu_buffer()
    {
        return m_gpu_buffer.get();
    }

    // Synchronize pending GPU operations before CPU access
    void sync_gpu() const;

    // Lazy sync: only copy from GPU if data is dirty and CPU access is needed
    void sync_gpu_if_needed() const;

    // Check if there are pending GPU operations
    bool has_pending_gpu_ops() const
    {
        return m_pending_events_count > 0;
    }

    // Record pending GPU operation (for lazy synchronization)
    void record_pending_gpu_op(cl_event evt)
    {
        if (m_pending_events_count < max_pending_events && evt)
        {
            m_pending_events[m_pending_events_count++] = evt;
        }
        else if (evt)
        {
            clReleaseEvent(evt);
        }
    }

    // Flush: force synchronize all pending GPU operations
    // Call this when you need the result immediately or to batch multiple kernels
    void flush();

    static constexpr size_t max_pending_events = 16;

    // Helper to allocate persistent GPU buffer (internal use)
    void try_allocate_gpu_buffer(Index size);

    // GPU-resident mode: keep data on GPU between operations
    // When true, avoid copying back to CPU after operations
    void set_gpu_resident(bool resident)
    {
        m_gpu_resident = resident;
    }
    bool is_gpu_resident() const
    {
        return m_gpu_resident;
    }

    // Lazy sync: check if GPU data needs to be copied to CPU
    bool needs_sync_to_host() const
    {
        return m_needs_sync_to_host;
    }

    // Lazy sync: check if host data needs to be copied to GPU before GPU execution
    bool needs_sync_to_device() const
    {
        return m_needs_sync_to_device;
    }

    // Mark that GPU data has been modified and needs sync before CPU access
    void mark_dirty()
    {
        m_needs_sync_to_host = true;
        m_needs_sync_to_device = false;
    }

    // Mark that host data has been modified and needs sync before GPU access
    void mark_host_dirty()
    {
        m_needs_sync_to_device = true;
        m_needs_sync_to_host = false;
    }

    // Pipeline mode: queue multiple kernels before syncing
    void set_pipeline_mode(bool enable)
    {
        m_pipeline_mode = enable;
    }
    bool is_pipeline_mode() const
    {
        return m_pipeline_mode;
    }

   private:
    // ── Device-resident fast path ────────────────────────────────────────────
    // Every tensor owns a persistent GPU buffer (allocated in the constructors
    // via try_allocate_gpu_buffer). ensure_device_current() makes that buffer
    // hold the tensor's current data, uploading from host only when the host
    // copy is newer (m_needs_sync_to_device). Ops then feed operands' own
    // buffers straight to kernels — no per-op host round-trip — and leave the
    // result device-resident (host mirror synced lazily on first host access).
    // Returns false when no buffer could be allocated; callers fall back to
    // the legacy host-staged path.
    bool ensure_device_current(const char* what) const;

    // Kernel launchers over resident buffers. Each returns false (after
    // logging) when the fast path can't run — OpenCL unavailable, a buffer
    // missing, or a CL call failed — so call sites fall back to the legacy
    // path. Kernel arg orders match KernelManager.cpp:
    //   binary:        (A, B, out, n)      unary:         (in, out, n)
    //   unary_scalar:  (in, out, scalar, n)
    //   inplace_binary:(A, B, n)           inplace_scalar:(A, scalar, n)
    static bool launch_binary_resident(const char* kernel_name,
        const OpenCLTensorBackend& a,
        const OpenCLTensorBackend& b,
        OpenCLTensorBackend& out,
        const char* what);
    static bool launch_unary_resident(const char* kernel_name,
        const OpenCLTensorBackend& a,
        OpenCLTensorBackend& out,
        const char* what);
    static bool launch_unary_scalar_resident(const char* kernel_name,
        const OpenCLTensorBackend& a,
        float scalar,
        OpenCLTensorBackend& out,
        const char* what);
    static bool launch_inplace_binary_resident(const char* kernel_name,
        OpenCLTensorBackend& a,
        const OpenCLTensorBackend& b,
        const char* what);
    static bool launch_inplace_scalar_resident(
        const char* kernel_name, OpenCLTensorBackend& a, float scalar, const char* what);

    // Marks the result of a device kernel: device copy is authoritative,
    // host mirror stale until lazily synced.
    void mark_device_result()
    {
        m_gpu_resident = true;
        m_needs_sync_to_host = true;
        m_needs_sync_to_device = false;
    }

    std::unique_ptr<OpenCLHostStorage> m_backend;
    std::unique_ptr<OpenCLTensorBackend> m_grad_backend;
    std::unique_ptr<tensor::GPUBuffer> m_gpu_buffer;
    bool m_has_gpu_memory = false;
    bool m_gpu_resident = false;
    bool m_pipeline_mode = false;
    mutable bool m_needs_sync_to_host = false;
    mutable bool m_needs_sync_to_device = true;
    mutable cl_event m_pending_events[max_pending_events];
    mutable size_t m_pending_events_count = 0;
};

} // namespace nn

#endif // OPENCL_TENSOR_BACKEND_HPP
