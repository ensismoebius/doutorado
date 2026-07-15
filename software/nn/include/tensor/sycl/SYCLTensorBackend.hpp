#ifndef SYCL_TENSOR_BACKEND_HPP
#define SYCL_TENSOR_BACKEND_HPP

#include <random>
#include <string>
#include <vector>

#include "tensor/xtensor/XTensorBackend.hpp"

namespace nn
{

/**
 * @brief SYCL (Khronos) tensor backend.
 *
 * Execution model
 * - Hot math (matmul family, elementwise, activations, reductions) runs as
 *   SYCL 2020 kernels through a process-wide in-order queue; every accelerated
 *   op is copy-in/copy-out (host mirror → USM device buffer → kernel → host),
 *   so the host mirror stays authoritative at all times.
 * - Everything shape/metadata related (slicing, blocks, element access,
 *   gradients, serialization) delegates to the XTensorBackend host mirror,
 *   exactly like DeviceTensorBackend — see that skeleton for the contract.
 * - If no SYCL device can be created at runtime, every accelerated op falls
 *   back to the host mirror implementation. The backend is therefore always
 *   functional; SYCL is an accelerator, not a requirement, at runtime.
 *
 * Build model
 * - This header contains no SYCL includes and is compilable by any C++20
 *   compiler. All SYCL code lives in src/core/tensor/sycl/SYCLTensorBackend.cpp,
 *   which requires a SYCL implementation (AdaptiveCpp or oneAPI DPC++) and is
 *   only compiled when NN_BACKEND=SYCL.
 *
 * Numerical notes
 * - Device reductions (sum/norm/MSE) accumulate in a different order than the
 *   host serial loop; expect float rounding differences within ~1e-4 relative,
 *   same as the OpenCL backend. Parity tests use tolerances accordingly.
 */
class SYCLTensorBackend
{
   public:
    // -- Constructors ------------------------------------------------------
    SYCLTensorBackend() = default;
    explicit SYCLTensorBackend(Index rows, Index cols) : m_host(rows, cols) {}
    explicit SYCLTensorBackend(Index d1, Index d2, Index d3) : m_host(d1, d2, d3) {}
    explicit SYCLTensorBackend(Index d1, Index d2, Index d3, Index d4) : m_host(d1, d2, d3, d4) {}
    explicit SYCLTensorBackend(const std::vector<Index>& shape) : m_host(shape) {}

    explicit SYCLTensorBackend(const XTensorBackend& host) : m_host(host) {}
    explicit SYCLTensorBackend(XTensorBackend&& host) noexcept : m_host(std::move(host)) {}

    SYCLTensorBackend(const SYCLTensorBackend&) = default;
    SYCLTensorBackend(SYCLTensorBackend&&) noexcept = default;
    SYCLTensorBackend& operator=(const SYCLTensorBackend&) = default;
    SYCLTensorBackend& operator=(SYCLTensorBackend&&) noexcept = default;

    // -- Static factories ---------------------------------------------------
    static SYCLTensorBackend zeros(Index rows, Index cols)
    {
        return SYCLTensorBackend(XTensorBackend::zeros(rows, cols));
    }
    static SYCLTensorBackend ones(Index rows, Index cols)
    {
        return SYCLTensorBackend(XTensorBackend::ones(rows, cols));
    }
    static SYCLTensorBackend random(Index rows, Index cols)
    {
        return SYCLTensorBackend(XTensorBackend::random(rows, cols));
    }
    static SYCLTensorBackend random(Index rows, Index cols, std::mt19937& rng)
    {
        return SYCLTensorBackend(XTensorBackend::random(rows, cols, rng));
    }
    static SYCLTensorBackend random(Index d1, Index d2, Index d3)
    {
        return SYCLTensorBackend(XTensorBackend::random(d1, d2, d3));
    }
    static SYCLTensorBackend random(Index d1, Index d2, Index d3, std::mt19937& rng)
    {
        return SYCLTensorBackend(XTensorBackend::random(d1, d2, d3, rng));
    }

    // -- Runtime diagnostics -------------------------------------------------
    /// True when a SYCL queue could be created (device present, runtime OK).
    static bool sycl_runtime_available();
    /// Human-readable device description ("<vendor> <name>"), or "host fallback".
    static std::string device_description();

    // -- Shape / sizing (host mirror) ----------------------------------------
    std::vector<Index> shape() const
    {
        return m_host.shape();
    }
    void reshape(const std::vector<Index>& new_shape)
    {
        m_host.reshape(new_shape);
    }
    Index rows() const
    {
        return m_host.rows();
    }
    Index cols() const
    {
        return m_host.cols();
    }
    Index size() const
    {
        return m_host.size();
    }

    // -- Element access (host mirror) ----------------------------------------
    float& at(Index i)
    {
        return m_host.at(i);
    }
    const float& at(Index i) const
    {
        return m_host.at(i);
    }
    float& at(Index r, Index c)
    {
        return m_host.at(r, c);
    }
    const float& at(Index r, Index c) const
    {
        return m_host.at(r, c);
    }
    float& at(Index d1, Index d2, Index d3)
    {
        return m_host.at(d1, d2, d3);
    }
    const float& at(Index d1, Index d2, Index d3) const
    {
        return m_host.at(d1, d2, d3);
    }
    float& at(Index d1, Index d2, Index d3, Index d4)
    {
        return m_host.at(d1, d2, d3, d4);
    }
    const float& at(Index d1, Index d2, Index d3, Index d4) const
    {
        return m_host.at(d1, d2, d3, d4);
    }
    float& at(const std::vector<Index>& idx)
    {
        return m_host.at(idx);
    }
    const float& at(const std::vector<Index>& idx) const
    {
        return m_host.at(idx);
    }

    // -- Elementwise binary ops (SYCL kernels; defined in .cpp) ---------------
    SYCLTensorBackend add(const SYCLTensorBackend& other) const;
    SYCLTensorBackend subtract(const SYCLTensorBackend& other) const;
    SYCLTensorBackend multiply(const SYCLTensorBackend& other) const;
    SYCLTensorBackend divide(const SYCLTensorBackend& other) const;
    void add_inplace(const SYCLTensorBackend& other);
    void subtract_inplace(const SYCLTensorBackend& other);
    void multiply_inplace(const SYCLTensorBackend& other);
    void divide_inplace(const SYCLTensorBackend& other);

    // -- Scalar ops (SYCL kernels) --------------------------------------------
    SYCLTensorBackend add_scalar(float val) const;
    SYCLTensorBackend multiply_scalar(float val) const;
    SYCLTensorBackend divide_scalar(float val) const;
    void add_scalar_inplace(float val);
    void multiply_scalar_inplace(float val);
    void divide_scalar_inplace(float val);

    // -- Elementwise unary ops (SYCL kernels) ----------------------------------
    SYCLTensorBackend exp() const;
    SYCLTensorBackend sqrt() const;
    SYCLTensorBackend square() const;
    SYCLTensorBackend abs() const;
    SYCLTensorBackend relu() const;
    SYCLTensorBackend leaky_relu(float alpha) const;
    void sqrt_inplace();
    void square_inplace();

    // -- Matmul family / transpose (SYCL kernels) -------------------------------
    SYCLTensorBackend matmul(const SYCLTensorBackend& other) const;
    SYCLTensorBackend matmul_transposed(const SYCLTensorBackend& other) const;
    SYCLTensorBackend transpose() const;

    // -- Reductions (SYCL kernels) -----------------------------------------------
    float sum() const;
    float norm() const;
    float mean_squared_error(const SYCLTensorBackend& target) const;
    SYCLTensorBackend sum_rows() const;
    SYCLTensorBackend sum_cols() const;
    SYCLTensorBackend rowwise_sum() const;

    // -- Broadcasting (host mirror — xtensor broadcasting semantics) -----------
    SYCLTensorBackend add_row_broadcast(const SYCLTensorBackend& row) const
    {
        return SYCLTensorBackend(m_host.add_row_broadcast(row.m_host));
    }
    void add_row_broadcast_inplace(const SYCLTensorBackend& row)
    {
        m_host.add_row_broadcast_inplace(row.m_host);
    }
    void add_col_vector_to_rows_inplace(const SYCLTensorBackend& col_vector)
    {
        m_host.add_col_vector_to_rows_inplace(col_vector.m_host);
    }

    // -- Comparisons / clamp / mean (host mirror; cold paths — masks, clipping) -
    SYCLTensorBackend compare_lt(const SYCLTensorBackend& other) const
    {
        return SYCLTensorBackend(m_host.compare_lt(other.m_host));
    }
    SYCLTensorBackend compare_gt(const SYCLTensorBackend& other) const
    {
        return SYCLTensorBackend(m_host.compare_gt(other.m_host));
    }
    SYCLTensorBackend compare_le(const SYCLTensorBackend& other) const
    {
        return SYCLTensorBackend(m_host.compare_le(other.m_host));
    }
    SYCLTensorBackend compare_ge(const SYCLTensorBackend& other) const
    {
        return SYCLTensorBackend(m_host.compare_ge(other.m_host));
    }
    SYCLTensorBackend compare_eq(const SYCLTensorBackend& other) const
    {
        return SYCLTensorBackend(m_host.compare_eq(other.m_host));
    }
    SYCLTensorBackend compare_lt_scalar(float v) const
    {
        return SYCLTensorBackend(m_host.compare_lt_scalar(v));
    }
    SYCLTensorBackend compare_gt_scalar(float v) const
    {
        return SYCLTensorBackend(m_host.compare_gt_scalar(v));
    }
    SYCLTensorBackend compare_le_scalar(float v) const
    {
        return SYCLTensorBackend(m_host.compare_le_scalar(v));
    }
    SYCLTensorBackend compare_ge_scalar(float v) const
    {
        return SYCLTensorBackend(m_host.compare_ge_scalar(v));
    }
    SYCLTensorBackend compare_eq_scalar(float v) const
    {
        return SYCLTensorBackend(m_host.compare_eq_scalar(v));
    }
    SYCLTensorBackend clamp(float min_val, float max_val) const
    {
        return SYCLTensorBackend(m_host.clamp(min_val, max_val));
    }
    void clamp_inplace(float min_val, float max_val)
    {
        m_host.clamp_inplace(min_val, max_val);
    }
    float mean() const
    {
        return m_host.mean();
    }

    // -- Slicing / blocks (host mirror) -----------------------------------------
    SYCLTensorBackend row(Index i) const
    {
        return SYCLTensorBackend(m_host.row(i));
    }
    SYCLTensorBackend col(Index j) const
    {
        return SYCLTensorBackend(m_host.col(j));
    }
    SYCLTensorBackend leftCols(Index n) const
    {
        return SYCLTensorBackend(m_host.leftCols(n));
    }
    SYCLTensorBackend topRows(Index n) const
    {
        return SYCLTensorBackend(m_host.topRows(n));
    }
    SYCLTensorBackend block(Index r, Index c, Index rows, Index cols) const
    {
        return SYCLTensorBackend(m_host.block(r, c, rows, cols));
    }
    void setBlock(Index r, Index c, const SYCLTensorBackend& other)
    {
        m_host.setBlock(r, c, other.m_host);
    }
    SYCLTensorBackend slice(std::span<const int> indices) const
    {
        return SYCLTensorBackend(m_host.slice(indices));
    }
    SYCLTensorBackend slice_batch(Index b) const
    {
        return SYCLTensorBackend(m_host.slice_batch(b));
    }
    void set_batch_slice(Index b, const SYCLTensorBackend& val)
    {
        m_host.set_batch_slice(b, val.m_host);
    }
    SYCLTensorBackend slice_time(Index t) const
    {
        return SYCLTensorBackend(m_host.slice_time(t));
    }
    void set_time_slice(Index t, const SYCLTensorBackend& val)
    {
        m_host.set_time_slice(t, val.m_host);
    }

    // -- Mutators (host mirror) ----------------------------------------------
    void fill(float v)
    {
        m_host.fill(v);
    }
    void set_zero()
    {
        m_host.set_zero();
    }
    void set_ones()
    {
        m_host.set_ones();
    }

    // -- Raw data access (host mirror is authoritative) ------------------------
    const float* data_ptr() const
    {
        return m_host.data_ptr();
    }
    float* mutable_data_ptr()
    {
        return m_host.mutable_data_ptr();
    }

    // -- Misc predicates (host mirror) -----------------------------------------
    bool hasNaN() const
    {
        return m_host.hasNaN();
    }
    bool operator==(const SYCLTensorBackend& other) const
    {
        return m_host == other.m_host;
    }
    bool operator!=(const SYCLTensorBackend& other) const
    {
        return !(*this == other);
    }

    // -- Gradient API (host mirror; lazy allocation inside XTensorBackend) ------
    SYCLTensorBackend get_grad() const
    {
        return SYCLTensorBackend(m_host.get_grad());
    }
    void set_grad(const SYCLTensorBackend& other)
    {
        m_host.set_grad(other.m_host);
    }
    void zero_grad()
    {
        m_host.zero_grad();
    }

    // -- Host mirror escape hatch (backend-internal use only) -------------------
    XTensorBackend& host()
    {
        return m_host;
    }
    const XTensorBackend& host() const
    {
        return m_host;
    }

   private:
    XTensorBackend m_host;
};

} // namespace nn

#endif // SYCL_TENSOR_BACKEND_HPP
