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

#include <initializer_list>
#include <memory>
#include <optional>
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
    // binary_elementwise's two staging strategies, cheapest-transfer-first.
    // std::optional means "this strategy cannot run here"; the oneshot stage
    // has no successor and so always returns a value or throws.
    static void enqueue_binary_kernel(const char* kernel_name,
        const char* what,
        cl_mem a_mem,
        cl_mem b_mem,
        cl_mem out_mem,
        std::size_t n);
    std::optional<OpenCLTensorBackend> binary_stage_pooled(const char* kernel_name,
        const char* what,
        const OpenCLTensorBackend& other,
        std::size_t n) const;
    OpenCLTensorBackend binary_stage_oneshot(const char* kernel_name,
        const char* what,
        const OpenCLTensorBackend& other,
        std::size_t n) const;

    // One body for add/subtract/multiply/divide: they differ only in which
    // kernel runs, so they used to be four ~150-line near-copies. The copies
    // had silently drifted apart on shape-mismatch handling (two exception
    // types, one of them naming two unrelated causes); a single body cannot
    // drift. Arg order matches the binary kernels: (A, B, out, n).
    // Same story for square/sqrt/abs/exp/relu. Their five copies had drifted
    // into three different shapes (eager sync_gpu() vs lazy, and a
    // resident-output path present in three of them). Arg order: (in, out, n).
    OpenCLTensorBackend unary_elementwise(const char* kernel_name, const char* what) const;

    // matmul_transposed_add_col_bias_{relu,sigmoid,tanh,leaky_relu} were four
    // ~200-line bodies whose only differences were the kernel name and, for
    // leaky_relu alone, one extra float at argument 7. The pack carries that
    // float when there is one, so the size arguments keep their fixed indices
    // and the extras land after them.
    OpenCLTensorBackend matmul_transposed_bias_activated(const char* kernel_name,
        const char* what,
        const OpenCLTensorBackend& other,
        const OpenCLTensorBackend& bias,
        std::initializer_list<float> extra_scalars) const;

    // matmul()/matmul_transposed()'s shared staging strategies. Both launch
    // with the identical (a, b, c, m, n, k) arg order and a plain 2D NDRange
    // (no local size) -- they differ only in kernel_name and in how the
    // caller derives m/n/k from its own transpose semantics.
    // matmul_lhs_transposed uses a genuinely different, tiled launch
    // (local[2] = {16, 16}) and is decomposed separately below rather than
    // forced through this helper.
    static void enqueue_matmul_kernel(const char* kernel_name,
        const char* what,
        cl_mem a_mem,
        cl_mem b_mem,
        cl_mem c_mem,
        Index m,
        Index n,
        Index k);

    // Shared (a, b, bias, c, m, n, k, [extra scalars...]) arg-binding + launch used by all
    // three staging strategies (GPU-resident, pool, DeviceMemory fallback) inside
    // matmul_transposed_bias_activated() -- extracted so that ~90-line, near-identical block
    // exists once instead of three times.
    static void enqueue_matmul_bias_activated_kernel(const char* kernel_name,
        const char* what,
        cl_mem a_mem,
        cl_mem b_mem,
        cl_mem bias_mem,
        cl_mem c_mem,
        Index m,
        Index n,
        Index k,
        std::initializer_list<float> extra_scalars);

    // matmul_transposed_bias_activated()'s three staging strategies, tried cheapest-transfer
    // first: resident (nullopt if any operand isn't device-current), pooled (nullopt if the
    // pool or an acquire is unavailable), oneshot (always succeeds or throws). `out` is the
    // single host-side result buffer shared by pooled/oneshot, constructed once by the caller
    // — matching the pre-extraction code, which allocated it unconditionally even though only
    // one of the three stages ever consumes it.
    std::optional<OpenCLTensorBackend> matmul_bias_activated_stage_resident(const char* kernel_name,
        const char* what,
        const OpenCLTensorBackend& other,
        const OpenCLTensorBackend& bias,
        std::initializer_list<float> extra_scalars,
        Index m,
        Index n,
        Index k,
        std::size_t a_bytes,
        std::size_t b_bytes,
        std::size_t bias_bytes) const;
    std::optional<OpenCLTensorBackend> matmul_bias_activated_stage_pooled(const char* kernel_name,
        const char* what,
        const OpenCLTensorBackend& other,
        const OpenCLTensorBackend& bias,
        std::initializer_list<float> extra_scalars,
        Index m,
        Index n,
        Index k,
        std::size_t a_bytes,
        std::size_t b_bytes,
        std::size_t bias_bytes,
        std::size_t c_bytes,
        OpenCLHostStorage& out) const;
    OpenCLTensorBackend matmul_bias_activated_stage_oneshot(const char* kernel_name,
        const char* what,
        const OpenCLTensorBackend& other,
        const OpenCLTensorBackend& bias,
        std::initializer_list<float> extra_scalars,
        Index m,
        Index n,
        Index k,
        std::size_t a_bytes,
        std::size_t b_bytes,
        std::size_t bias_bytes,
        std::size_t c_bytes,
        OpenCLHostStorage& out) const;
    OpenCLTensorBackend matmul_stage_resident(const char* kernel_name,
        const char* what,
        const OpenCLTensorBackend& other,
        Index m,
        Index n,
        Index k,
        bool finish_after_launch) const;
    std::optional<OpenCLTensorBackend> matmul_stage_pooled(const char* kernel_name,
        const char* what,
        const OpenCLTensorBackend& other,
        Index m,
        Index n,
        Index k) const;
    OpenCLTensorBackend matmul_stage_oneshot(const char* kernel_name,
        const char* what,
        const OpenCLTensorBackend& other,
        Index m,
        Index n,
        Index k) const;
    OpenCLTensorBackend matmul_dispatch(const char* kernel_name,
        const char* what,
        const OpenCLTensorBackend& other,
        Index m,
        Index n,
        Index k,
        bool finish_after_launch) const;

    // matmul_lhs_transposed()'s own staging strategies -- tiled launch
    // (local[2] = {16, 16}), output shape (k, n), kept separate from the
    // matmul_dispatch family above rather than forced through it.
    static void enqueue_matmul_lhs_transposed_kernel(
        cl_mem a_mem, cl_mem b_mem, cl_mem c_mem, Index m, Index n, Index k);
    OpenCLTensorBackend matmul_lhs_transposed_stage_resident(
        const OpenCLTensorBackend& other, Index m, Index n, Index k) const;
    std::optional<OpenCLTensorBackend> matmul_lhs_transposed_stage_pooled(
        const OpenCLTensorBackend& other, Index m, Index n, Index k) const;
    OpenCLTensorBackend matmul_lhs_transposed_stage_oneshot(
        const OpenCLTensorBackend& other, Index m, Index n, Index k) const;

    // lif_step_inplace's own precondition check and kernel launch, split out
    // so each is short enough to read as a unit. There is no resident/pooled
    // variant here (see launch_lif_step_kernel's comment for why).
    void validate_lif_step_shapes(const OpenCLTensorBackend& input,
        const OpenCLTensorBackend& output,
        const OpenCLTensorBackend* adapt_a,
        bool use_adaptation) const;
    void launch_lif_step_kernel(const OpenCLTensorBackend& input,
        OpenCLTensorBackend& output,
        OpenCLTensorBackend* adapt_a,
        float beta,
        float threshold,
        float reset_potential,
        bool reset_zero,
        float adapt_decay,
        float adapt_coupling,
        bool use_adaptation,
        std::size_t n);

    // mean_squared_error()'s three staging strategies -- same shape as sum()'s
    // but over two operands (this, target).
    std::optional<float> mse_stage_resident(const OpenCLTensorBackend& target,
        std::size_t n,
        std::size_t local,
        std::size_t global,
        std::size_t num_groups) const;
    std::optional<float> mse_stage_pooled(const OpenCLTensorBackend& target,
        std::size_t n,
        std::size_t local,
        std::size_t global,
        std::size_t num_groups) const;
    float mse_stage_oneshot(const OpenCLTensorBackend& target,
        std::size_t n,
        std::size_t local,
        std::size_t global,
        std::size_t num_groups) const;

    // sum()'s three staging strategies. Each returns std::nullopt (or, for
    // the last, never) when it cannot run; reduce_partials() is the one place
    // sum_kernel's per-work-group partials become the final float, so all
    // three stages are guaranteed to reduce them the same way.
    static float reduce_partials(const std::vector<float>& partials);
    std::optional<float> sum_stage_resident(
        std::size_t n, std::size_t local, std::size_t global, std::size_t num_groups) const;
    std::optional<float> sum_stage_pooled(
        std::size_t n, std::size_t local, std::size_t global, std::size_t num_groups) const;
    float sum_stage_oneshot(
        std::size_t n, std::size_t local, std::size_t global, std::size_t num_groups) const;

    // rowwise_sum()'s three staging strategies, tried cheapest-transfer-first.
    // Arg order: (in, out, rows, cols), 1D NDRange over rows.
    static void enqueue_rowwise_sum_kernel(
        cl_mem in_mem, cl_mem out_mem, Index num_rows, Index num_cols);
    bool rowwise_sum_stage_resident(OpenCLTensorBackend& result) const;
    std::optional<OpenCLTensorBackend> rowwise_sum_stage_pooled() const;
    OpenCLTensorBackend rowwise_sum_stage_oneshot() const;

    // transpose()'s own two staging strategies. Kept separate from the
    // (in,out,scalars,n) unary family above because transpose_kernel's arg
    // shape is (in,out,rows,cols) with a 2D NDRange, not a 1D one -- sharing
    // would mean forcing a 4-argument kernel through a helper built for a
    // 2-4 argument 1D one.
    static void enqueue_transpose_kernel(
        cl_mem in_mem, cl_mem out_mem, Index in_rows, Index in_cols);
    std::optional<OpenCLTensorBackend> transpose_stage_pooled(
        Index in_rows, Index in_cols, std::size_t bytes) const;
    OpenCLTensorBackend transpose_stage_oneshot(
        Index in_rows, Index in_cols, std::size_t bytes) const;

    // The three staging strategies behind both unary entry points, tried
    // cheapest-transfer-first. std::nullopt means "this strategy cannot run
    // here"; the last one has no successor and so returns a value or throws.
    static void enqueue_unary_kernel(const char* kernel_name,
        const char* what,
        cl_mem in_mem,
        cl_mem out_mem,
        std::initializer_list<float> scalars,
        std::size_t n,
        const cl_event* wait_event,
        cl_event* out_event);
    std::optional<OpenCLTensorBackend> unary_stage_resident(const char* kernel_name,
        const char* what,
        std::initializer_list<float> scalars,
        std::size_t n) const;
    std::optional<OpenCLTensorBackend> unary_stage_pooled(const char* kernel_name,
        const char* what,
        std::initializer_list<float> scalars,
        std::size_t n,
        OpenCLHostStorage& out) const;
    OpenCLTensorBackend unary_stage_oneshot(const char* kernel_name,
        const char* what,
        std::initializer_list<float> scalars,
        std::size_t n,
        OpenCLHostStorage& out) const;
    OpenCLTensorBackend run_unary_stages(
        const char* kernel_name, const char* what, std::initializer_list<float> scalars) const;

    // square_inplace / sqrt_inplace. Arg order: (A, n), written back into A.
    void unary_inplace(const char* kernel_name, const char* what);

    // leaky_relu(alpha) and clamp(min, max) differ ONLY in how many float
    // arguments sit between `out` and `n`, so the scalars travel as a pack and
    // the size argument lands at index 2 + scalars.size(). Adding a third
    // scalar op now costs one line, not another 180.
    OpenCLTensorBackend unary_scalars_elementwise(
        const char* kernel_name, const char* what, std::initializer_list<float> scalars) const;

    // compare_elementwise's two staging strategies -- same shape as
    // binary_elementwise's, reusing its enqueue_binary_kernel since the arg
    // order is identical (A, B, out, n).
    std::optional<OpenCLTensorBackend> compare_stage_pooled(const char* kernel_name,
        const char* what,
        const OpenCLTensorBackend& other,
        std::size_t n) const;
    OpenCLTensorBackend compare_stage_oneshot(const char* kernel_name,
        const char* what,
        const OpenCLTensorBackend& other,
        std::size_t n) const;

    // compare_lt's own row/column broadcasting path -- see definition for why
    // this one comparison tolerates a shape mismatch when the others refuse it.
    OpenCLTensorBackend compare_lt_broadcast(const OpenCLTensorBackend& other) const;

    // compare_gt/le/ge/eq were four byte-identical bodies (modulo the kernel
    // name). compare_lt is deliberately NOT here: it alone supports row/column
    // broadcasting, so folding it in would either lose that or force it on the
    // other four. Arg order: (A, B, out, n).
    OpenCLTensorBackend compare_elementwise(
        const char* kernel_name, const char* what, const OpenCLTensorBackend& other) const;

    // scalar_inplace / unary_inplace share this: one buffer, an optional
    // scalar pack, cheapest-transfer-first staging. Same shape as the
    // unary_elementwise family above, mirrored for the in-place case.
    static void enqueue_inplace_kernel(const char* kernel_name,
        const char* what,
        cl_mem data_mem,
        std::initializer_list<float> scalars,
        std::size_t n,
        const cl_event* wait_event,
        cl_event* out_event);
    bool inplace_elementwise_stage_resident(const char* kernel_name,
        const char* what,
        std::initializer_list<float> scalars,
        std::size_t n);
    bool inplace_elementwise_stage_pooled(const char* kernel_name,
        const char* what,
        std::initializer_list<float> scalars,
        std::size_t n);
    void inplace_elementwise_stage_oneshot(const char* kernel_name,
        const char* what,
        std::initializer_list<float> scalars,
        std::size_t n);
    void run_inplace_elementwise_stages(
        const char* kernel_name, const char* what, std::initializer_list<float> scalars);

    // add_row_broadcast_inplace() and add_col_vector_to_rows_inplace() share
    // these: a (1,M) row and an (M,1) col_vector are the same flat M values,
    // so both launch the identical add_col_vector_to_rows_kernel(data, vec,
    // rows, cols). The pooled stage chains upload/kernel/download on events
    // (up to 2 wait events in), unlike the other *_stage_pooled families in
    // this file, which block between steps -- preserved as found.
    static void enqueue_broadcast_vector_kernel(cl_mem data_mem,
        cl_mem vec_mem,
        Index num_rows,
        Index num_cols,
        cl_uint num_wait_events,
        const cl_event* wait_events,
        cl_event* out_event);
    bool broadcast_vector_stage_resident(const OpenCLTensorBackend& vec, const char* what);
    bool broadcast_vector_stage_pooled(const OpenCLTensorBackend& vec, const char* what);
    void broadcast_vector_stage_oneshot(const OpenCLTensorBackend& vec, const char* what);
    void run_broadcast_vector_stages(const OpenCLTensorBackend& vec, const char* what);

    // binary_inplace's three staging strategies, tried cheapest-transfer-first.
    // Each returns false when it cannot run, so the caller falls to the next;
    // the last one has no successor and therefore returns void.
    bool inplace_stage_resident(
        const char* kernel_name, const char* what, const OpenCLTensorBackend& other, std::size_t n);
    bool inplace_stage_pooled(
        const char* kernel_name, const char* what, const OpenCLTensorBackend& other, std::size_t n);
    void inplace_stage_oneshot(
        const char* kernel_name, const char* what, const OpenCLTensorBackend& other, std::size_t n);
    static void enqueue_inplace_binary_kernel(const char* kernel_name,
        const char* what,
        cl_mem a_mem,
        cl_mem b_mem,
        std::size_t n,
        const cl_event* wait_events,
        cl_uint wait_count,
        cl_event* out_event);

    // add/subtract/multiply/divide_inplace: four more byte-identical bodies,
    // ~215 lines each. Arg order: (A, B, n), result written back into A.
    void binary_inplace(
        const char* kernel_name, const char* what, const OpenCLTensorBackend& other);

    // add/multiply/divide_scalar_inplace. Arg order: (A, scalar, n).
    void scalar_inplace(const char* kernel_name, const char* what, float val);

    OpenCLTensorBackend binary_elementwise(
        const char* kernel_name, const char* what, const OpenCLTensorBackend& other) const;

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

    // ── Lazy host access ─────────────────────────────────────────────────────
    // Sync-on-read: pull the device copy down only at the moment host bytes are
    // actually needed, instead of eagerly at the top of every op. The eager form
    // cost a blocking clEnqueueReadBuffer (~200 us for 1 KiB on rusticl) on every
    // op whose operands were device-resident — including ops that then went on to
    // use the device buffer and never touched the host bytes at all.
    //
    // Both are no-ops unless the tensor is device-resident with a stale host
    // mirror, so the fallback (host-staged) paths keep working unchanged.
    //
    // NOT for use inside sync_gpu(), ensure_device_current(), or the copy
    // ctor/assignment — those manage the sync flags themselves and would recurse.
    //
    // Defined in the .cpp: OpenCLHostStorage is incomplete here.
    const float* host_data() const;
    float* mutable_host_data();

    // ── Device-side view/slice ───────────────────────────────────────────────
    // Rectangular copy between two strided views, run on the device via
    // strided_copy_2d_kernel. Backs block/setBlock/row/col/topRows/leftCols and
    // the rank-3 time/batch slice ops.
    //
    // These were the single largest cost in the backend: each was a full
    // blocking readback plus a host element loop plus a forced full re-upload,
    // and LSTMLayer calls them once per timestep (T=256). Measured on a real
    // Guayaquil run before this change: 517k blocking writes + 227k blocking reads,
    // 85% of all OpenCL time, against 2.4% in actual kernels.
    //
    // Element (i,j) maps to base + i*stride_i + j*stride_j on each side.
    // Returns false when the device path can't run; callers fall back to the
    // host loop.
    struct StridedRegion
    {
        Index base = 0;
        Index stride_i = 0;
        Index stride_j = 0;
    };

    static bool launch_strided_copy(const OpenCLTensorBackend& src,
        const StridedRegion& src_region,
        OpenCLTensorBackend& dst,
        const StridedRegion& dst_region,
        Index ni,
        Index nj,
        const char* what);

    // Marks the result of a device kernel: device copy is authoritative,
    // host mirror stale until lazily synced.
    void mark_device_result()
    {
        m_gpu_resident = true;
        m_needs_sync_to_host = true;
        m_needs_sync_to_device = false;
    }

    // ── In-flight asynchronous upload ────────────────────────────────────────
    // Inside a BatchScope, ensure_device_current() issues the host→device copy
    // non-blocking (~2.5 us vs ~43 us blocking, 1 KiB on rusticl) because a
    // blocking transfer is a full pipeline flush and would defeat the batching.
    //
    // OpenCL requires the host source to stay valid and unmodified until an
    // async transfer completes, and the source here is this tensor's own host
    // storage. So the event is owned by the tensor: anything that would free or
    // overwrite that storage must wait on it first.
    void wait_for_upload() const;

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
    // Outstanding async upload reading from m_backend, or nullptr. Owned.
    mutable cl_event m_upload_event = nullptr;
};

} // namespace nn

#endif // OPENCL_TENSOR_BACKEND_HPP
