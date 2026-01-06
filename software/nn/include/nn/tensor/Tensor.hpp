#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <algorithm>
#include <memory>
#include <span>
#include <vector>

#include "nn/tensor/ITensorBackend.hpp"

// -----------------------------------------------------------------------------
// Lightweight Tensor wrapper
// - Thin, backend-driven container for numeric data used across the library.
// - Delegates storage and heavy operations to `ITensorBackend` (Eigen-based by
//   default). This header provides a compact, stable API surface for callers
//   while keeping backend details encapsulated.
// -----------------------------------------------------------------------------
namespace nn
{

class Tensor
{
   public:
    // -----------------------------------------------------------------
    // Constructors / Factories
    // -----------------------------------------------------------------
    /// Default empty tensor (backend will be null until constructed).
    Tensor();
    /// Take ownership of an existing backend implementation.
    Tensor(std::unique_ptr<ITensorBackend> backend);
    /// Construct a 2-D tensor with `rows x cols` shape.
    Tensor(Index rows, Index cols);
    /// Construct a 4-D tensor with provided dimensions.
    Tensor(Index dim1, Index d2, Index d3, Index d4);
    /// Construct from an explicit shape vector.
    Tensor(const std::vector<Index>& shape);

    /// Create a tensor filled with `value`.
    static auto constant(Index rows, Index cols, float value) -> Tensor;
    /// Create a zeros tensor.
    static auto zeros(Index rows, Index cols) -> Tensor;
    /// Create a ones tensor.
    static auto ones(Index rows, Index cols) -> Tensor;

    // -----------------------------------------------------------------
    // Copy / Move (defaulted where appropriate)
    // -----------------------------------------------------------------
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) = default;
    Tensor& operator=(Tensor&& other) = default;
    ~Tensor() = default;

    // -----------------------------------------------------------------
    // Shape / sizing helpers
    // Implementation note: These are defined inline to allow the compiler
    // to eliminate function call overhead in tight loops.
    // -----------------------------------------------------------------
    /// Return the tensor shape as a vector of dimension sizes.
    auto get_shape() const -> const std::vector<Index>&
    {
        return m_backend->shape();
    }
    /// Reshape the tensor (backend dependent; may reallocate or reinterpret).
    void reshape(const std::vector<Index>& new_shape);
    [[nodiscard]] auto rows() const noexcept -> Index
    {
        return m_backend->rows();
    }
    [[nodiscard]] auto cols() const noexcept -> Index
    {
        return m_backend->cols();
    }
    [[nodiscard]] auto size() const noexcept -> Index
    {
        return m_backend->size();
    }

    // -----------------------------------------------------------------
    // Element access
    // Implementation note: These are defined inline to allow the compiler
    // to eliminate function call overhead. Bounds checking is handled
    // by the backend.
    // -----------------------------------------------------------------
    /// 1-D element access
    auto at(Index i) -> float&
    {
        return m_backend->at(i);
    }
    [[nodiscard]] auto at(Index i) const -> const float&
    {
        return m_backend->at(i);
    }
    /// 2-D element access (row, col)
    auto at(Index row, Index col) -> float&
    {
        return m_backend->at(row, col);
    }
    [[nodiscard]] auto at(Index row, Index col) const -> const float&
    {
        return m_backend->at(row, col);
    }
    /// 4-D element access
    auto at(Index d1, Index d2, Index d3, Index d4) -> float&
    {
        return m_backend->at(d1, d2, d3, d4);
    }
    [[nodiscard]] auto at(Index d1, Index d2, Index d3, Index d4) const -> const float&
    {
        return m_backend->at(d1, d2, d3, d4);
    }
    /// N-D element access using an indices vector.
    auto at(const std::vector<Index>& indices) -> float&
    {
        return m_backend->at(indices);
    }
    [[nodiscard]] auto at(const std::vector<Index>& indices) const -> const float&
    {
        return m_backend->at(indices);
    }

    // -----------------------------------------------------------------
    // Views / Slicing
    // -----------------------------------------------------------------
    auto row(Index i) const -> Tensor;
    auto col(Index j) const -> Tensor;
    auto leftCols(Index n) const -> Tensor;
    auto topRows(Index n) const -> Tensor;

    auto block(Index row, Index col, Index rows, Index cols) const -> Tensor;
    void setBlock(Index row, Index col, const Tensor& block);

    /// Return a new tensor containing the selected indices. The argument is a
    /// non-owning span of integer indices (caller-owned memory).
    [[nodiscard]] auto slice(std::span<const int> indices) const -> Tensor;

    // -----------------------------------------------------------------
    // Element-wise & matrix ops
    // -----------------------------------------------------------------
    auto add(const Tensor& other) const -> Tensor;
    auto multiply(const Tensor& other) const -> Tensor;
    auto add_scalar(float scalar) const -> Tensor;
    auto multiply_scalar(float scalar) const -> Tensor;

    auto matmul(const Tensor& other) const -> Tensor;
    auto transpose() const -> Tensor;

    auto relu() const -> Tensor;
    auto leaky_relu(float alpha = 0.01f) const -> Tensor;

    // -----------------------------------------------------------------
    // Reductions / losses / validation
    // -----------------------------------------------------------------
    auto mean_squared_error(const Tensor& target) const -> float;
    auto norm() const -> float;
    auto sum() const -> float;
    /// Sum across columns -> returns (rows, 1)
    auto sum_rows() const -> Tensor;
    /// Sum across rows -> returns (1, cols)
    auto sum_cols() const -> Tensor;

    auto hasNaN() const -> bool;

    // -----------------------------------------------------------------
    // Convenience elementwise math
    // -----------------------------------------------------------------
    auto sqrt() const -> Tensor;
    auto square() const -> Tensor;
    auto abs() const -> Tensor;
    auto divide(const Tensor& other) const -> Tensor;
    auto divide_scalar(float scalar) const -> Tensor;

    // -----------------------------------------------------------------
    // Initialization helpers (mutating)
    // -----------------------------------------------------------------
    void fill(float value);
    void set_zero();
    void set_ones();
    // Backwards-compatible aliases
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
    // Raw data access (legacy helpers)
    // -----------------------------------------------------------------
    /// Pointer to read-only data (backend owned). Prefer backend-safe accessors.
    const float* data() const
    {
        return data_ptr();
    }
    /// Mutable pointer to backend data (use cautiously).
    float* mutable_data()
    {
        return mutable_data_ptr();
    }
    /// Convenience operator for 2-D indexing.
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
        return m_backend->data_ptr();
    }
    float* mutable_data_ptr()
    {
        return m_backend->mutable_data_ptr();
    }

    // -----------------------------------------------------------------
    // Gradients and backend access
    // -----------------------------------------------------------------
    /// Returns a copy or view of the gradient tensor (backend-defined behavior).
    auto grad() const -> Tensor;
    auto grad() -> Tensor;
    void set_grad(const Tensor& new_grad);

    /// Zero the gradient buffer on the backend.
    void zero_grad();

    // -----------------------------------------------------------------
    // Miscellaneous utilities
    // -----------------------------------------------------------------
    template <typename vector_type>
    [[nodiscard]] auto toVector() const -> std::vector<vector_type>;

    // -----------------------------------------------------------------
    // Operators / comparisons
    // -----------------------------------------------------------------
    auto operator+(const Tensor& other) const -> Tensor
    {
        return add(other);
    }
    auto operator-(const Tensor& other) const -> Tensor;
    auto operator*(const Tensor& other) const -> Tensor
    {
        return multiply(other);
    }
    auto operator*(float scalar) const -> Tensor;
    auto operator+(float scalar) const -> Tensor;
    auto operator-(float scalar) const -> Tensor;
    auto operator/(float scalar) const -> Tensor;

    auto operator==(const Tensor& other) const -> bool;
    auto operator!=(const Tensor& other) const -> bool;

    // -----------------------------------------------------------------
    // Comma initializer helper (stream-like syntax)
    // -----------------------------------------------------------------
    class CommaInitializer;
    auto operator<<(float value) -> CommaInitializer;

    /// Access the underlying backend pointer (read-only).
    auto get_backend() const -> const ITensorBackend*
    {
        return m_backend.get();
    }

   private:
    std::unique_ptr<ITensorBackend> m_backend;
};

// -----------------------------------------------------------------------------
// CommaInitializer: small utility used by `operator<<` for value-list initialization
// The implementation writes values directly into the backend memory when the
// initializer is destroyed.
// -----------------------------------------------------------------------------
class Tensor::CommaInitializer
{
   public:
    CommaInitializer(Tensor& tensor, float first_value) : m_tensor(tensor)
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
        const auto total = static_cast<size_t>(m_tensor.size());
        if (m_values.size() > total)
        {
            throw std::out_of_range("Tensor comma initializer received too many values");
        }
        float* data = m_tensor.mutable_data_ptr();
        std::copy(m_values.begin(), m_values.end(), data);
    }

   private:
    Tensor& m_tensor;
    std::vector<float> m_values;
};

inline auto Tensor::operator<<(float value) -> CommaInitializer
{
    return CommaInitializer(*this, value);
}

// Template implementations must be available in the header, outside the class but inside
// the namespace. Keep the implementation minimal here; backend will provide data access.
template <typename vector_type>
auto Tensor::toVector() const -> std::vector<vector_type>
{
    // Placeholder: concrete backends should enable an efficient transfer path.
    return std::vector<vector_type>();
}

} // namespace nn

#endif // TENSOR_HPP