#/**
 * @file include/nn/tensor/Tensor.hpp
 * @brief Backend-driven tensor wrapper used throughout the project.
 *
 * **PHASE:** Core stable API — central value-type tensor wrapper parameterized by
 * backend.
 *
 * **Key ideas:**
 * - Backend-driven value type with zero-overhead abstraction.
 * - `TensorImpl` is a value type; operations typically return new tensors.
 * - Gradients are backend-managed; see backend `grad_ref()` semantics.
 *
 * **Contract:**
 * - 2D mapping: rows/cols map to xtensor storage row/col (row-major logical).
 * - 3D mapping: rows = d1; cols = d2; storage cols = d2 * d3.
 * - 4D mapping: rows = d1; cols = d2; storage cols = d2 * d3 * d4.
 * - Public APIs throw `std::invalid_argument` or `std::out_of_range` on misuse.
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <algorithm>
#include <span>
#include <vector>

// Forward declare Device to avoid circular include
namespace nn
{
struct Device;
class OpenCLTensorBackend;
} // namespace nn

// XTensorBackend.hpp must be available in include path.
#include "nn/tensor/xtensor/XTensorBackend.hpp"

// -----------------------------------------------------------------------------
// Lightweight Tensor wrapper (Templated)
// - Thin, backend-driven container for numeric data used across the library.
// - Replaces virtual ITensorBackend with a template policy `Backend` for
//   compile-time inlining and zero-overhead abstraction.
//
// Key idea for users of the library:
// - `TensorImpl` is a *value type*. Most operations return new tensors (copies), not views.
// - “Views” like row()/block() still return a new TensorImpl constructed from a backend value.
//   Whether that is a cheap view or an owning copy depends on the backend implementation.
//
// Shapes:
// - This API supports 2D and a “4D-on-top-of-2D-storage” convention via the backend.
// - For 4D tensors, the backend typically stores data in an xt::xarray<float> with shape:
//     rows = dim1
//     cols = dim2 * dim3 * dim4
//   and `at(d1,d2,d3,d4)` computes the flattened column index.
//
// Gradients (important gotcha):
// - `grad()` currently returns a *copy* of the stored gradient (backend `get_grad()` is by value).
// - Optimizers should read gradients via `p->grad()` (copy) and update parameters via `param =
// ...`.
// - To *set* gradients during backward, use `set_grad()` or backend-provided mechanisms.
//   There is no public “grad_ref” on TensorImpl right now.
// -----------------------------------------------------------------------------
namespace nn
{

// Forward declare for CommaInitializer
template <typename Backend>
class TensorImpl;

template <typename Backend = XTensorBackend>
class TensorImpl
{
   public:
    using index_type = size_t; // Alignment with std::size_t usually

    // Note on index types:
    // - The public API uses `nn::Index` (declared by the backend header) for shape/indices.
    // - `index_type` is kept for compatibility but is not the primary index type.

    // -----------------------------------------------------------------
    // Constructors / Factories
    // -----------------------------------------------------------------
    /// Default empty tensor.
    TensorImpl() = default;

    /// Construct from explicit backend instance
    explicit TensorImpl(const Backend& backend) : backend_(backend) {}
    explicit TensorImpl(Backend&& backend) noexcept : backend_(std::move(backend)) {}

    /// Construct from a tensor backed by a different backend via element-wise copy.
    template <typename OtherBackend>
    explicit TensorImpl(const TensorImpl<OtherBackend>& other) : backend_(other.get_shape())
    {
        for (Index i = 0; i < other.size(); ++i)
        {
            at(i) = other.at(i);
        }
    }

    /// Construct a 2-D tensor.
    TensorImpl(Index rows, Index cols) : backend_(rows, cols) {}

    /// Construct a 3-D tensor. Storage: (d1, d2*d3).
    TensorImpl(Index d1, Index d2, Index d3) : backend_(d1, d2, d3) {}

    /// Construct a 4-D tensor.
    TensorImpl(Index dim1, Index d2, Index d3, Index d4) : backend_(dim1, d2, d3, d4) {}

    /// Construct from shape vector.
    explicit TensorImpl(const std::vector<Index>& shape) : backend_(shape) {}

    /// Create a tensor filled with `value`.
    static auto constant(Index rows, Index cols, float value) -> TensorImpl
    {
        TensorImpl t(rows, cols);
        t.fill(value);
        return t;
    }
    /// Create a 3-D tensor filled with `value`.
    static auto constant(Index d1, Index d2, Index d3, float value) -> TensorImpl
    {
        TensorImpl t(d1, d2, d3);
        t.fill(value);
        return t;
    }
    /// Create a zeros tensor.
    static auto zeros(Index rows, Index cols) -> TensorImpl
    {
        // Require Backend::zeros to return Backend value
        return TensorImpl(Backend::zeros(rows, cols));
    }
    /// Create a 3-D zeros tensor.
    static auto zeros(Index d1, Index d2, Index d3) -> TensorImpl
    {
        TensorImpl t(d1, d2, d3);
        t.fill(0.0f);
        return t;
    }
    /// Create a ones tensor.
    static auto ones(Index rows, Index cols) -> TensorImpl
    {
        return TensorImpl(Backend::ones(rows, cols));
    }
    /// Create a 3-D ones tensor.
    static auto ones(Index d1, Index d2, Index d3) -> TensorImpl
    {
        TensorImpl t(d1, d2, d3);
        t.fill(1.0f);
        return t;
    }

    /// Create a tensor with uniform random values in [0,1).
    static auto rand(Index rows, Index cols) -> TensorImpl
    {
        return TensorImpl(Backend::random(rows, cols));
    }
    /// Create a tensor with uniform random values in [0,1) using provided RNG.
    static auto rand(Index rows, Index cols, std::mt19937& rng) -> TensorImpl
    {
        return TensorImpl(Backend::random(rows, cols, rng));
    }
    /// Create a 3-D tensor with uniform random values in [0,1).
    static auto rand(Index d1, Index d2, Index d3) -> TensorImpl
    {
        return TensorImpl(Backend::random(d1, d2, d3));
    }
    /// Create a 3-D tensor with uniform random values in [0,1) using provided RNG.
    static auto rand(Index d1, Index d2, Index d3, std::mt19937& rng) -> TensorImpl
    {
        return TensorImpl(Backend::random(d1, d2, d3, rng));
    }

    // -----------------------------------------------------------------
    // Copy / Move (Defaulted - rules of 5 handled by Backend)
    // -----------------------------------------------------------------
    TensorImpl(const TensorImpl&) = default;
    TensorImpl& operator=(const TensorImpl&) = default;
    TensorImpl(TensorImpl&&) = default;
    TensorImpl& operator=(TensorImpl&&) = default;
    ~TensorImpl() = default;

    // -----------------------------------------------------------------
    // Shape / sizing helpers
    // -----------------------------------------------------------------
    auto get_shape() const -> std::vector<Index>
    {
        return backend_.shape();
    }
    void reshape(const std::vector<Index>& new_shape)
    {
        backend_.reshape(new_shape);
    }

    auto reshape(const std::vector<Index>& new_shape) const -> TensorImpl
    {
        return TensorImpl(backend_.reshape(new_shape));
    }
    [[nodiscard]] auto rows() const noexcept -> Index
    {
        return backend_.rows();
    }
    [[nodiscard]] auto cols() const noexcept -> Index
    {
        return backend_.cols();
    }
    [[nodiscard]] auto size() const noexcept -> Index
    {
        return backend_.size();
    }

    // Shape caveat:
    // - For 2D tensors, rows()/cols() match the underlying matrix.
    // - For 4D tensors, rows()/cols() reflect dim1/dim2 (not the flattened storage columns).
    //   Use get_shape() and at(d1,d2,d3,d4) for 4D indexing.

    // -----------------------------------------------------------------
    // Element access (Inline)
    // -----------------------------------------------------------------
    auto at(Index i) -> float&
    {
        return backend_.at(i);
    }
    [[nodiscard]] auto at(Index i) const -> const float&
    {
        return backend_.at(i);
    }

    auto at(Index row, Index col) -> float&
    {
        return backend_.at(row, col);
    }
    [[nodiscard]] auto at(Index row, Index col) const -> const float&
    {
        return backend_.at(row, col);
    }

    auto at(Index d1, Index d2, Index d3) -> float&
    {
        if constexpr (requires(Backend& b) { b.at(d1, d2, d3); })
        {
            return backend_.at(d1, d2, d3);
        }
        return backend_.at(std::vector<Index>{d1, d2, d3});
    }
    [[nodiscard]] auto at(Index d1, Index d2, Index d3) const -> const float&
    {
        if constexpr (requires(const Backend& b) { b.at(d1, d2, d3); })
        {
            return backend_.at(d1, d2, d3);
        }
        return backend_.at(std::vector<Index>{d1, d2, d3});
    }

    auto at(Index d1, Index d2, Index d3, Index d4) -> float&
    {
        return backend_.at(d1, d2, d3, d4);
    }
    [[nodiscard]] auto at(Index d1, Index d2, Index d3, Index d4) const -> const float&
    {
        return backend_.at(d1, d2, d3, d4);
    }

    auto at(const std::vector<Index>& indices) -> float&
    {
        return backend_.at(indices);
    }
    [[nodiscard]] auto at(const std::vector<Index>& indices) const -> const float&
    {
        return backend_.at(indices);
    }

    // -----------------------------------------------------------------
    // Views / Slicing (Backend returns Value)
    // -----------------------------------------------------------------
    auto row(Index i) const -> TensorImpl
    {
        if constexpr (requires(const Backend& b) { b.row(i); })
        {
            return TensorImpl(backend_.row(i));
        }

        TensorImpl out(1, cols());
        for (Index c = 0; c < cols(); ++c)
        {
            out.at(0, c) = at(i, c);
        }
        return out;
    }
    auto col(Index j) const -> TensorImpl
    {
        if constexpr (requires(const Backend& b) { b.col(j); })
        {
            return TensorImpl(backend_.col(j));
        }

        TensorImpl out(rows(), 1);
        for (Index r = 0; r < rows(); ++r)
        {
            out.at(r, 0) = at(r, j);
        }
        return out;
    }
    auto leftCols(Index n) const -> TensorImpl
    {
        if constexpr (requires(const Backend& b) { b.leftCols(n); })
        {
            return TensorImpl(backend_.leftCols(n));
        }

        if (n > cols()) throw std::out_of_range("leftCols exceeds tensor width");
        TensorImpl out(rows(), n);
        for (Index r = 0; r < rows(); ++r)
        {
            for (Index c = 0; c < n; ++c)
            {
                out.at(r, c) = at(r, c);
            }
        }
        return out;
    }
    auto topRows(Index n) const -> TensorImpl
    {
        if constexpr (requires(const Backend& b) { b.topRows(n); })
        {
            return TensorImpl(backend_.topRows(n));
        }

        if (n > rows()) throw std::out_of_range("topRows exceeds tensor height");
        TensorImpl out(n, cols());
        for (Index r = 0; r < n; ++r)
        {
            for (Index c = 0; c < cols(); ++c)
            {
                out.at(r, c) = at(r, c);
            }
        }
        return out;
    }

    auto block(Index row, Index col, Index rows, Index cols) const -> TensorImpl
    {
        if constexpr (requires(const Backend& b) { b.block(row, col, rows, cols); })
        {
            return TensorImpl(backend_.block(row, col, rows, cols));
        }

        if (row + rows > this->rows() || col + cols > this->cols())
            throw std::out_of_range("block exceeds tensor bounds");
        TensorImpl out(rows, cols);
        for (Index r = 0; r < rows; ++r)
        {
            for (Index c = 0; c < cols; ++c)
            {
                out.at(r, c) = at(row + r, col + c);
            }
        }
        return out;
    }

    void setBlock(Index row, Index col, const TensorImpl& block)
    {
        if constexpr (requires(Backend& b) { b.setBlock(row, col, block.backend_); })
        {
            backend_.setBlock(row, col, block.backend_);
            return;
        }

        if (row + block.rows() > rows() || col + block.cols() > cols())
            throw std::out_of_range("setBlock exceeds tensor bounds");
        for (Index r = 0; r < block.rows(); ++r)
        {
            for (Index c = 0; c < block.cols(); ++c)
            {
                at(row + r, col + c) = block.at(r, c);
            }
        }
    }

    [[nodiscard]] auto slice(std::span<const int> indices) const -> TensorImpl
    {
        if constexpr (requires(const Backend& b) { b.slice(indices); })
        {
            return TensorImpl(backend_.slice(indices));
        }

        if (get_shape().size() != 2)
            throw std::invalid_argument("slice(span<int>) fallback requires rank-2 tensor");
        TensorImpl out(indices.size(), cols());
        for (Index i = 0; i < indices.size(); ++i)
        {
            const auto src_r = static_cast<Index>(indices[i]);
            if (src_r >= rows()) throw std::out_of_range("slice index out of range");
            for (Index c = 0; c < cols(); ++c)
            {
                out.at(i, c) = at(src_r, c);
            }
        }
        return out;
    }

    // Extract 2D slice [b, :, :] from a 3D (B, T, D) tensor → (T, D).
    [[nodiscard]] auto slice_batch(Index b) const -> TensorImpl
    {
        if constexpr (requires(const Backend& be) { be.slice_batch(b); })
        {
            return TensorImpl(backend_.slice_batch(b));
        }

        const auto shape = get_shape();
        if (shape.size() != 3)
            throw std::invalid_argument("slice_batch fallback requires rank-3 tensor");
        if (b >= shape[0]) throw std::out_of_range("slice_batch index out of range");

        TensorImpl out(shape[1], shape[2]);
        for (Index t = 0; t < shape[1]; ++t)
        {
            for (Index d = 0; d < shape[2]; ++d)
            {
                out.at(t, d) = at(b, t, d);
            }
        }
        return out;
    }

    // Write 2D (T, D) tensor into [b, :, :] of a 3D (B, T, D) tensor.
    void set_batch_slice(Index b, const TensorImpl& val)
    {
        if constexpr (requires(Backend& be) { be.set_batch_slice(b, val.backend_); })
        {
            backend_.set_batch_slice(b, val.backend_);
            return;
        }

        const auto shape = get_shape();
        if (shape.size() != 3)
            throw std::invalid_argument("set_batch_slice fallback requires rank-3 tensor");
        if (b >= shape[0]) throw std::out_of_range("set_batch_slice index out of range");
        if (val.rows() != shape[1] || val.cols() != shape[2])
            throw std::invalid_argument("set_batch_slice value shape mismatch");

        for (Index t = 0; t < shape[1]; ++t)
        {
            for (Index d = 0; d < shape[2]; ++d)
            {
                at(b, t, d) = val.at(t, d);
            }
        }
    }

    // Extract (B, D) slice at time t from a 3D (B, T, D) tensor.
    [[nodiscard]] auto slice_time(Index t) const -> TensorImpl
    {
        if constexpr (requires(const Backend& be) { be.slice_time(t); })
        {
            return TensorImpl(backend_.slice_time(t));
        }

        const auto shape = get_shape();
        if (shape.size() != 3)
            throw std::invalid_argument("slice_time fallback requires rank-3 tensor");
        if (t >= shape[1]) throw std::out_of_range("slice_time index out of range");

        TensorImpl out(shape[0], shape[2]);
        for (Index b = 0; b < shape[0]; ++b)
        {
            for (Index d = 0; d < shape[2]; ++d)
            {
                out.at(b, d) = at(b, t, d);
            }
        }
        return out;
    }

    // Write (B, D) tensor into [:, t, :] of a 3D (B, T, D) tensor.
    void set_time_slice(Index t, const TensorImpl& val)
    {
        if constexpr (requires(Backend& be) { be.set_time_slice(t, val.backend_); })
        {
            backend_.set_time_slice(t, val.backend_);
            return;
        }

        const auto shape = get_shape();
        if (shape.size() != 3)
            throw std::invalid_argument("set_time_slice fallback requires rank-3 tensor");
        if (t >= shape[1]) throw std::out_of_range("set_time_slice index out of range");
        if (val.rows() != shape[0] || val.cols() != shape[2])
            throw std::invalid_argument("set_time_slice value shape mismatch");

        for (Index b = 0; b < shape[0]; ++b)
        {
            for (Index d = 0; d < shape[2]; ++d)
            {
                at(b, t, d) = val.at(b, d);
            }
        }
    }

    // -----------------------------------------------------------------
    // Element-wise & matrix ops
    // -----------------------------------------------------------------
    auto add(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(backend_.add(other.backend_));
    }

    auto multiply(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(backend_.multiply(other.backend_));
    }

    auto add_scalar(float scalar) const -> TensorImpl
    {
        return TensorImpl(backend_.add_scalar(scalar));
    }

    auto multiply_scalar(float scalar) const -> TensorImpl
    {
        return TensorImpl(backend_.multiply_scalar(scalar));
    }

    auto matmul(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(backend_.matmul(other.backend_));
    }

    // -----------------------------------------------------------------
    // In-place mutating operations
    // -----------------------------------------------------------------
    void add_inplace(const TensorImpl& other)
    {
        backend_.add_inplace(other.backend_);
    }
    void add_col_vector_to_rows_inplace(const TensorImpl& col_vector)
    {
        backend_.add_col_vector_to_rows_inplace(col_vector.backend_);
    }
    auto add_row_broadcast(const TensorImpl& row) const -> TensorImpl
    {
        if constexpr (requires(const Backend& b) { b.add_row_broadcast(row.backend_); })
        {
            return TensorImpl(backend_.add_row_broadcast(row.backend_));
        }

        if (row.rows() != 1 || row.cols() != cols())
            throw std::invalid_argument("add_row_broadcast requires row shape (1, cols)");

        TensorImpl out(rows(), cols());
        for (Index r = 0; r < rows(); ++r)
        {
            for (Index c = 0; c < cols(); ++c)
            {
                out.at(r, c) = at(r, c) + row.at(0, c);
            }
        }
        return out;
    }
    void add_row_broadcast_inplace(const TensorImpl& row)
    {
        if constexpr (requires(Backend& b) { b.add_row_broadcast_inplace(row.backend_); })
        {
            backend_.add_row_broadcast_inplace(row.backend_);
            return;
        }

        if (row.rows() != 1 || row.cols() != cols())
            throw std::invalid_argument("add_row_broadcast_inplace requires row shape (1, cols)");

        for (Index r = 0; r < rows(); ++r)
        {
            for (Index c = 0; c < cols(); ++c)
            {
                at(r, c) += row.at(0, c);
            }
        }
    }
    void subtract_inplace(const TensorImpl& other)
    {
        backend_.subtract_inplace(other.backend_);
    }
    void multiply_inplace(const TensorImpl& other)
    {
        backend_.multiply_inplace(other.backend_);
    }
    void divide_inplace(const TensorImpl& other)
    {
        backend_.divide_inplace(other.backend_);
    }
    void add_scalar_inplace(float scalar)
    {
        backend_.add_scalar_inplace(scalar);
    }
    void multiply_scalar_inplace(float scalar)
    {
        backend_.multiply_scalar_inplace(scalar);
    }
    void divide_scalar_inplace(float scalar)
    {
        backend_.divide_scalar_inplace(scalar);
    }
    void sqrt_inplace()
    {
        backend_.sqrt_inplace();
    }
    void square_inplace()
    {
        backend_.square_inplace();
    }

    auto exp() const -> TensorImpl
    {
        return TensorImpl(backend_.exp());
    }
    auto rowwise_sum() const -> TensorImpl
    {
        return TensorImpl(backend_.rowwise_sum());
    }
    auto matmul_transposed(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(backend_.matmul_transposed(other.backend_));
    }

    auto transpose() const -> TensorImpl
    {
        return TensorImpl(backend_.transpose());
    }

    auto relu() const -> TensorImpl
    {
        return TensorImpl(backend_.relu());
    }

    auto leaky_relu(float alpha = 0.01f) const -> TensorImpl
    {
        return TensorImpl(backend_.leaky_relu(alpha));
    }

    // -----------------------------------------------------------------
    // Reductions / losses / validation
    // -----------------------------------------------------------------
    auto mean_squared_error(const TensorImpl& target) const -> float
    {
        return backend_.mean_squared_error(target.backend_);
    }
    /**
     * Mean of all elements in the tensor. Delegates to the backend.
     */
    auto mean() const -> float
    {
        return backend_.mean();
    }
    auto norm() const -> float
    {
        return backend_.norm();
    }
    auto sum() const -> float
    {
        return backend_.sum();
    }

    auto sum_rows() const -> TensorImpl
    {
        if constexpr (requires(const Backend& b) { b.sum_rows(); })
        {
            return TensorImpl(backend_.sum_rows());
        }
        if constexpr (requires(const Backend& b) { b.rowwise_sum(); })
        {
            return TensorImpl(backend_.rowwise_sum());
        }

        TensorImpl out(rows(), 1);
        for (Index r = 0; r < rows(); ++r)
        {
            float acc = 0.0f;
            for (Index c = 0; c < cols(); ++c)
            {
                acc += at(r, c);
            }
            out.at(r, 0) = acc;
        }
        return out;
    }
    auto sum_cols() const -> TensorImpl
    {
        return TensorImpl(backend_.sum_cols());
    }

    auto hasNaN() const -> bool
    {
        return backend_.hasNaN();
    }

    // -----------------------------------------------------------------
    // Convenience elementwise math
    // -----------------------------------------------------------------
    auto sqrt() const -> TensorImpl
    {
        return TensorImpl(backend_.sqrt());
    }
    auto square() const -> TensorImpl
    {
        return TensorImpl(backend_.square());
    }
    auto abs() const -> TensorImpl
    {
        return TensorImpl(backend_.abs());
    }

    /**
     * Element-wise clamp: returns a new tensor where each element x is
     * clamped to the interval [min_val, max_val].
     */
    auto clamp(float min_val, float max_val) const -> TensorImpl
    {
        return TensorImpl(backend_.clamp(min_val, max_val));
    }

    /**
     * In-place clamp: mutates this tensor's storage.
     */
    void clamp_inplace(float min_val, float max_val)
    {
        backend_.clamp_inplace(min_val, max_val);
    }

    auto divide(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(backend_.divide(other.backend_));
    }

    auto divide_scalar(float scalar) const -> TensorImpl
    {
        return TensorImpl(backend_.divide_scalar(scalar));
    }

    // -----------------------------------------------------------------
    // Initialization helpers (mutating)
    // -----------------------------------------------------------------
    void fill(float value)
    {
        if constexpr (requires(Backend& b) { b.fill(value); })
        {
            backend_.fill(value);
            return;
        }

        for (Index i = 0; i < size(); ++i)
        {
            at(i) = value;
        }
    }
    void set_zero()
    {
        if constexpr (requires(Backend& b) { b.set_zero(); })
        {
            backend_.set_zero();
            return;
        }
        fill(0.0f);
    }
    void set_ones()
    {
        if constexpr (requires(Backend& b) { b.set_ones(); })
        {
            backend_.set_ones();
            return;
        }
        fill(1.0f);
    }
    void setZero()
    {
        set_zero();
    }
    void setOnes()
    {
        set_ones();
    }
    void setConstant(float value)
    {
        fill(value);
    }

    // -----------------------------------------------------------------
    // Raw data access
    // -----------------------------------------------------------------
    const float* data() const noexcept
    {
        return data_ptr();
    }
    float* mutable_data() noexcept
    {
        return mutable_data_ptr();
    }

    // Raw data pointers:
    // - These expose the backend's contiguous buffer (xtensor storage for the default backend).
    // - The memory order is backend-defined (xtensor is row-major by default).
    //   When you treat it as a flat array (e.g., gradient clipping), you are operating in
    //   that underlying order.

    float& operator()(Index i, Index j)
    {
        return at(i, j);
    }
    const float& operator()(Index i, Index j) const
    {
        return at(i, j);
    }
    float& operator()(Index d1, Index d2, Index d3)
    {
        return at(d1, d2, d3);
    }
    const float& operator()(Index d1, Index d2, Index d3) const
    {
        return at(d1, d2, d3);
    }

    const float* data_ptr() const noexcept
    {
        return backend_.data_ptr();
    }
    float* mutable_data_ptr() noexcept
    {
        return backend_.mutable_data_ptr();
    }

    // -----------------------------------------------------------------
    // Gradients
    // -----------------------------------------------------------------
    auto grad() const -> TensorImpl
    {
        return TensorImpl(backend_.get_grad());
    }
    // Mutable grad version?
    auto grad() -> TensorImpl
    {
        // Returns a copy.
        // If you need to mutate gradients, use set_grad()/zero_grad().
        return TensorImpl(backend_.get_grad());
    }

    void set_grad(const TensorImpl& new_grad)
    {
        if constexpr (requires(Backend& b) { b.set_grad(new_grad.backend_); })
        {
            backend_.set_grad(new_grad.backend_);
        }
        else
        {
            backend_.grad_ref() = new_grad.backend_;
        }
    }

    void zero_grad()
    {
        backend_.zero_grad();
    }

    // -----------------------------------------------------------------
    // Operators
    // -----------------------------------------------------------------
    auto operator+(const TensorImpl& other) const -> TensorImpl
    {
        return add(other);
    }
    auto operator-(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(backend_.subtract(other.backend_));
    }

    auto operator*(const TensorImpl& other) const -> TensorImpl
    {
        return multiply(other);
    }

    auto operator*(float scalar) const -> TensorImpl
    {
        return multiply_scalar(scalar);
    }
    auto operator+(float scalar) const -> TensorImpl
    {
        return add_scalar(scalar);
    }
    auto operator-(float scalar) const -> TensorImpl
    {
        return add_scalar(-scalar);
    }
    auto operator/(float scalar) const -> TensorImpl
    {
        return divide_scalar(scalar);
    }

    auto operator==(const TensorImpl& other) const -> bool
    {
        return backend_ == other.backend_;
    }
    auto operator!=(const TensorImpl& other) const -> bool
    {
        return !(*this == other);
    }

    // Elementwise comparisons returning tensors of 0.0/1.0 floats.
    auto operator<(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(backend_.compare_lt(other.backend_));
    }

    auto operator>(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(backend_.compare_gt(other.backend_));
    }

    auto operator<=(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(backend_.compare_le(other.backend_));
    }

    auto operator>=(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(backend_.compare_ge(other.backend_));
    }

    // Elementwise equality returning 0/1 tensor (not to be confused with overall equality)
    auto equal(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(backend_.compare_eq(other.backend_));
    }

    // Scalar comparisons
    auto operator<(float scalar) const -> TensorImpl
    {
        return TensorImpl(backend_.compare_lt_scalar(scalar));
    }
    auto operator>(float scalar) const -> TensorImpl
    {
        return TensorImpl(backend_.compare_gt_scalar(scalar));
    }
    auto operator<=(float scalar) const -> TensorImpl
    {
        return TensorImpl(backend_.compare_le_scalar(scalar));
    }
    auto operator>=(float scalar) const -> TensorImpl
    {
        return TensorImpl(backend_.compare_ge_scalar(scalar));
    }

    auto equal(float scalar) const -> TensorImpl
    {
        return TensorImpl(backend_.compare_eq_scalar(scalar));
    }

    template <typename vector_type>
    [[nodiscard]] static auto toVector() -> std::vector<vector_type>
    {
        return {};
    } // Placeholder

    // -----------------------------------------------------------------
    // Comma initialization
    // -----------------------------------------------------------------
    class CommaInitializer;
    auto operator<<(float value) -> CommaInitializer;

    template <typename TargetBackend>
    auto to_backend() const -> TensorImpl<TargetBackend>
    {
        return TensorImpl<TargetBackend>(*this);
    }

    /// Direct access to backend
    auto get_backend() const noexcept -> const Backend&
    {
        return backend_;
    }
    auto get_backend() noexcept -> Backend&
    {
        return backend_;
    }

   private:
    Backend backend_;
};

// Default type alias
using Tensor = TensorImpl<XTensorBackend>;
using OpenCLTensor = TensorImpl<OpenCLTensorBackend>;

// -----------------------------------------------------------------------------
// CommaInitializer Implementation
// -----------------------------------------------------------------------------
template <typename Backend>
class TensorImpl<Backend>::CommaInitializer
{
   public:
    CommaInitializer(TensorImpl<Backend>& tensor, float first_value) : m_tensor(tensor)
    {
        m_values.reserve(static_cast<size_t>(m_tensor.size()));
        m_values.push_back(first_value);
    }

    auto operator,(float value) -> CommaInitializer&
    {
        m_values.push_back(value);
        return *this;
    }

    ~CommaInitializer() noexcept(false)
    {
        // Comma-initialization copies the provided values into the underlying buffer.
        // To maintain compatibility with Eigen-style column-major filling:
        // Populate column by column.
        if (m_values.size() <= static_cast<size_t>(m_tensor.size()))
        {
            Index rows = m_tensor.rows();
            Index cols = m_tensor.cols();
            for (size_t i = 0; i < m_values.size(); ++i)
            {
                Index r = i % rows;
                Index c = i / rows;
                if (r < rows && c < cols)
                {
                    m_tensor.at(r, c) = m_values[i];
                }
            }
        }
    }

   private:
    TensorImpl<Backend>& m_tensor;
    std::vector<float> m_values;
};

template <typename Backend>
auto TensorImpl<Backend>::operator<<(float value) -> CommaInitializer
{
    return CommaInitializer(*this, value);
}

} // namespace nn

#endif // TENSOR_HPP
