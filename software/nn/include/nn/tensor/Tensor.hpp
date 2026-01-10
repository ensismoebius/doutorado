/**
 * @file Tensor.hpp
 * @brief Backend-driven tensor wrapper used throughout the project.
 *
 * This file is intentionally “dense”: it defines the core numeric container type.
 * The most common pitfall for new contributors is gradient ownership/aliasing.
 * Read the “Gradients (important gotcha)” block below before touching optimizers.
 */

#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <algorithm>
#include <span>
#include <vector>

// EigenTensorBackend.hpp must be available in include path.
#include "nn/tensor/EigenTensorBackend.hpp"

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
// - For 4D tensors, the backend typically stores data in an Eigen::MatrixXf with shape:
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

template <typename Backend = EigenTensorBackend>
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

    /// Construct from explicit backend instance (Moved).
    TensorImpl(Backend backend) : m_backend(std::move(backend)) {}

    /// Construct a 2-D tensor.
    TensorImpl(Index rows, Index cols) : m_backend(rows, cols) {}

    /// Construct a 4-D tensor.
    TensorImpl(Index dim1, Index d2, Index d3, Index d4) : m_backend(dim1, d2, d3, d4) {}

    /// Construct from shape vector.
    TensorImpl(const std::vector<Index>& shape) : m_backend(shape) {}

    /// Create a tensor filled with `value`.
    static auto constant(Index rows, Index cols, float value) -> TensorImpl
    {
        TensorImpl t(rows, cols);
        t.fill(value);
        return t;
    }
    /// Create a zeros tensor.
    static auto zeros(Index rows, Index cols) -> TensorImpl
    {
        // Require Backend::zeros to return Backend value
        return TensorImpl(Backend::zeros(rows, cols));
    }
    /// Create a ones tensor.
    static auto ones(Index rows, Index cols) -> TensorImpl
    {
        return TensorImpl(Backend::ones(rows, cols));
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
    auto get_shape() const -> const std::vector<Index>&
    {
        return m_backend.shape();
    }
    void reshape(const std::vector<Index>& new_shape)
    {
        m_backend.reshape(new_shape);
    }
    [[nodiscard]] auto rows() const noexcept -> Index
    {
        return m_backend.rows();
    }
    [[nodiscard]] auto cols() const noexcept -> Index
    {
        return m_backend.cols();
    }
    [[nodiscard]] auto size() const noexcept -> Index
    {
        return m_backend.size();
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
        return m_backend.at(i);
    }
    [[nodiscard]] auto at(Index i) const -> const float&
    {
        return m_backend.at(i);
    }

    auto at(Index row, Index col) -> float&
    {
        return m_backend.at(row, col);
    }
    [[nodiscard]] auto at(Index row, Index col) const -> const float&
    {
        return m_backend.at(row, col);
    }

    auto at(Index d1, Index d2, Index d3, Index d4) -> float&
    {
        return m_backend.at(d1, d2, d3, d4);
    }
    [[nodiscard]] auto at(Index d1, Index d2, Index d3, Index d4) const -> const float&
    {
        return m_backend.at(d1, d2, d3, d4);
    }

    auto at(const std::vector<Index>& indices) -> float&
    {
        return m_backend.at(indices);
    }
    [[nodiscard]] auto at(const std::vector<Index>& indices) const -> const float&
    {
        return m_backend.at(indices);
    }

    // -----------------------------------------------------------------
    // Views / Slicing (Backend returns Value)
    // -----------------------------------------------------------------
    auto row(Index i) const -> TensorImpl
    {
        return TensorImpl(m_backend.row(i));
    }
    auto col(Index j) const -> TensorImpl
    {
        return TensorImpl(m_backend.col(j));
    }
    auto leftCols(Index n) const -> TensorImpl
    {
        return TensorImpl(m_backend.leftCols(n));
    }
    auto topRows(Index n) const -> TensorImpl
    {
        return TensorImpl(m_backend.topRows(n));
    }

    auto block(Index row, Index col, Index rows, Index cols) const -> TensorImpl
    {
        return TensorImpl(m_backend.block(row, col, rows, cols));
    }

    void setBlock(Index row, Index col, const TensorImpl& block)
    {
        m_backend.setBlock(row, col, block.m_backend);
    }

    [[nodiscard]] auto slice(std::span<const int> indices) const -> TensorImpl
    {
        return TensorImpl(m_backend.slice(indices));
    }

    // -----------------------------------------------------------------
    // Element-wise & matrix ops
    // -----------------------------------------------------------------
    auto add(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(m_backend.add(other.m_backend));
    }

    auto multiply(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(m_backend.multiply(other.m_backend));
    }

    auto add_scalar(float scalar) const -> TensorImpl
    {
        return TensorImpl(m_backend.add_scalar(scalar));
    }

    auto multiply_scalar(float scalar) const -> TensorImpl
    {
        return TensorImpl(m_backend.multiply_scalar(scalar));
    }

    auto matmul(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(m_backend.matmul(other.m_backend));
    }

    auto transpose() const -> TensorImpl
    {
        return TensorImpl(m_backend.transpose());
    }

    auto relu() const -> TensorImpl
    {
        return TensorImpl(m_backend.relu());
    }

    auto leaky_relu(float alpha = 0.01f) const -> TensorImpl
    {
        return TensorImpl(m_backend.leaky_relu(alpha));
    }

    // -----------------------------------------------------------------
    // Reductions / losses / validation
    // -----------------------------------------------------------------
    auto mean_squared_error(const TensorImpl& target) const -> float
    {
        return m_backend.mean_squared_error(target.m_backend);
    }
    auto norm() const -> float
    {
        return m_backend.norm();
    }
    auto sum() const -> float
    {
        return m_backend.sum();
    }

    auto sum_rows() const -> TensorImpl
    {
        return TensorImpl(m_backend.sum_rows());
    }
    auto sum_cols() const -> TensorImpl
    {
        return TensorImpl(m_backend.sum_cols());
    }

    auto hasNaN() const -> bool
    {
        return m_backend.hasNaN();
    }

    // -----------------------------------------------------------------
    // Convenience elementwise math
    // -----------------------------------------------------------------
    auto sqrt() const -> TensorImpl
    {
        return TensorImpl(m_backend.sqrt());
    }
    auto square() const -> TensorImpl
    {
        return TensorImpl(m_backend.square());
    }
    auto abs() const -> TensorImpl
    {
        return TensorImpl(m_backend.abs());
    }

    auto divide(const TensorImpl& other) const -> TensorImpl
    {
        return TensorImpl(m_backend.divide(other.m_backend));
    }

    auto divide_scalar(float scalar) const -> TensorImpl
    {
        return TensorImpl(m_backend.divide_scalar(scalar));
    }

    // -----------------------------------------------------------------
    // Initialization helpers (mutating)
    // -----------------------------------------------------------------
    void fill(float value)
    {
        m_backend.fill(value);
    }
    void set_zero()
    {
        m_backend.set_zero();
    }
    void set_ones()
    {
        m_backend.set_ones();
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
    const float* data() const
    {
        return data_ptr();
    }
    float* mutable_data()
    {
        return mutable_data_ptr();
    }

    // Raw data pointers:
    // - These expose the backend's contiguous buffer (Eigen storage for the default backend).
    // - The memory order is backend-defined (Eigen is column-major by default).
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

    const float* data_ptr() const
    {
        return m_backend.data_ptr();
    }
    float* mutable_data_ptr()
    {
        return m_backend.mutable_data_ptr();
    }

    // -----------------------------------------------------------------
    // Gradients
    // -----------------------------------------------------------------
    auto grad() const -> TensorImpl
    {
        return TensorImpl(m_backend.get_grad());
    }
    // Mutable grad version?
    auto grad() -> TensorImpl
    {
        // Returns a copy.
        // If you need to mutate gradients, use set_grad()/zero_grad().
        return TensorImpl(m_backend.get_grad());
    }

    void set_grad(const TensorImpl& new_grad)
    {
        m_backend.set_grad(new_grad.m_backend);
    }

    void zero_grad()
    {
        m_backend.zero_grad();
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
        return TensorImpl(m_backend.subtract(other.m_backend));
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
        return m_backend == other.m_backend;
    }
    auto operator!=(const TensorImpl& other) const -> bool
    {
        return !(*this == other);
    }

    template <typename vector_type>
    [[nodiscard]] auto toVector() const -> std::vector<vector_type>
    {
        return {};
    } // Placeholder

    // -----------------------------------------------------------------
    // Comma initialization
    // -----------------------------------------------------------------
    class CommaInitializer;
    auto operator<<(float value) -> CommaInitializer;

    /// Direct access to backend
    auto get_backend() const -> const Backend&
    {
        return m_backend;
    }
    auto get_backend() -> Backend&
    {
        return m_backend;
    }

   private:
    Backend m_backend;
};

// Default type alias
using Tensor = TensorImpl<EigenTensorBackend>;

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
        // If fewer than `tensor.size()` values are provided, the remaining elements keep
        // their previous values.
        if (m_values.size() <= static_cast<size_t>(m_tensor.size()))
        {
            float* ptr = m_tensor.mutable_data();
            std::copy(m_values.begin(), m_values.end(), ptr);
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
