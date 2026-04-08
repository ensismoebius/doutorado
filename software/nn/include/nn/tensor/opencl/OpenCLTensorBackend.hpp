/**
 * @file include/nn/tensor/OpenCLTensorBackend.hpp
 * @brief OpenCL implementation of tensor backend.
 *
 * **API contract:**
 * - Same interface and semantics as EigenTensorBackend
 * - Move/copy construction and assignment supported
 * - Lazy gradient allocation (on grad_ref access)
 * - Row-major storage order
 */

#ifndef OPENCL_TENSOR_BACKEND_HPP
#define OPENCL_TENSOR_BACKEND_HPP

#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "nn/device/Device.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/eigen/EigenTensorBackend.hpp"
#include "nn/tensor/opencl/GPUBufferPool.hpp"
namespace nn
{

using Index = std::size_t;

// Forward-declare Eigen backend for Phase 1 fallback
class EigenTensorBackend;

/**
 * @brief OpenCL tensor backend (Phase 1: CPU fallback).
 *
 * Currently delegates all operations to EigenTensorBackend for correctness.
 * GPU implementations will be added incrementally in later phases
 * when performance targets are met.
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
    OpenCLTensorBackend() = default;

    explicit OpenCLTensorBackend(Index rows, Index cols);
    explicit OpenCLTensorBackend(Index d1, Index d2, Index d3, Index d4);
    explicit OpenCLTensorBackend(const std::vector<Index>& shape);

    // Construct from shape on a specific device (GPU)
    // Initializes GPU memory directly if OpenCL is available
    explicit OpenCLTensorBackend(const std::vector<Index>& shape, const Device& device);
    explicit OpenCLTensorBackend(Index rows, Index cols, const Device& device);

    // Copy/Move construction: delegate to Eigen backend
    OpenCLTensorBackend(const OpenCLTensorBackend& other);
    OpenCLTensorBackend(OpenCLTensorBackend&& other) noexcept = default;

    // Copy/Move assignment
    OpenCLTensorBackend& operator=(const OpenCLTensorBackend& other);
    OpenCLTensorBackend& operator=(OpenCLTensorBackend&& other) noexcept = default;

    ~OpenCLTensorBackend();

    // -----------------------------------------------------------------
    // Static Factories
    // -----------------------------------------------------------------
    static OpenCLTensorBackend zeros(Index rows, Index cols);
    static OpenCLTensorBackend ones(Index rows, Index cols);
    static OpenCLTensorBackend random(Index rows, Index cols);
    static OpenCLTensorBackend random(Index rows, Index cols, std::mt19937& rng);

    // -----------------------------------------------------------------
    // Shape & Access
    // -----------------------------------------------------------------
    const std::vector<Index>& shape() const;
    void reshape(const std::vector<Index>& new_shape);

    Index rows() const;
    Index cols() const;
    Index size() const;

    // N-D access operators (delegates to Eigen backend)
    float& at(Index i);
    const float& at(Index i) const;
    float& at(Index row, Index col);
    const float& at(Index row, Index col) const;
    float& at(Index d1, Index d2, Index d3, Index d4);
    const float& at(Index d1, Index d2, Index d3, Index d4) const;
    float& at(const std::vector<Index>& indices);
    const float& at(const std::vector<Index>& indices) const;

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
    void sqrt_inplace();
    void square_inplace();
    void add_col_vector_to_rows_inplace(const OpenCLTensorBackend& col_vector);

    // -----------------------------------------------------------------
    // Element-wise Operations (const, create new tensor)
    // -----------------------------------------------------------------
    OpenCLTensorBackend exp() const;
    OpenCLTensorBackend sqrt() const;
    OpenCLTensorBackend square() const;

    OpenCLTensorBackend add(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend subtract(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend multiply(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend divide(const OpenCLTensorBackend& other) const;

    OpenCLTensorBackend add_scalar(float val) const;
    OpenCLTensorBackend multiply_scalar(float val) const;
    OpenCLTensorBackend divide_scalar(float val) const;

    // -----------------------------------------------------------------
    // Reduction Operations
    // -----------------------------------------------------------------
    OpenCLTensorBackend rowwise_sum() const;

    // -----------------------------------------------------------------
    // Linear Algebra
    // -----------------------------------------------------------------
    OpenCLTensorBackend matmul(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend matmul_transposed(const OpenCLTensorBackend& other) const;
    OpenCLTensorBackend transpose() const;

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
     * @brief Get const reference to gradient (throws if not allocated).
     */
    auto get_grad() const -> const OpenCLTensorBackend&;

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
    static void verify_runtime_activity_or_throw(const Tensor& prediction,
        const Tensor& target,
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

    static constexpr size_t max_pending_events = 16;

    // Helper to allocate persistent GPU buffer (internal use)
    void try_allocate_gpu_buffer(Index size);

   private:
    std::unique_ptr<EigenTensorBackend> m_backend;
    std::unique_ptr<OpenCLTensorBackend> m_grad_backend;
    std::unique_ptr<tensor::GPUBuffer> m_gpu_buffer;
    bool m_has_gpu_memory = false;
    mutable cl_event m_pending_events[max_pending_events];
    mutable size_t m_pending_events_count = 0;
};

} // namespace nn

#endif // OPENCL_TENSOR_BACKEND_HPP
