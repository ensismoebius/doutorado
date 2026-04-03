/**
 * @file include/nn/tensor/OpenCLTensorBackend.hpp
 * @brief OpenCL implementation of tensor backend.
 *
 * **PHASE 1 (MVP):** CPU fallback wrapper — delegates all operations to
 * Eigen backend. Validates API contract, tests, and build integration.
 *
 * **Future phases:** Implement GPU kernels (matmul, element-wise, etc.)
 * when performance measurements justify the complexity.
 *
 * **Hardware assumptions:**
 * - AMD Renoir APU (7 CUs, 64 KiB LDS, no UMA)
 * - Explicit buffer synchronization required
 * - Kernel launch overhead is critical
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
#include <vector>

#include "nn/tensor/GPUBufferPool.hpp"
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
    // -----------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------
    OpenCLTensorBackend() = default;

    explicit OpenCLTensorBackend(Index rows, Index cols);
    explicit OpenCLTensorBackend(Index d1, Index d2, Index d3, Index d4);
    explicit OpenCLTensorBackend(const std::vector<Index>& shape);

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

   private:
    std::unique_ptr<EigenTensorBackend> m_backend;
    std::unique_ptr<OpenCLTensorBackend> m_grad_backend;
};

} // namespace nn

#endif // OPENCL_TENSOR_BACKEND_HPP
