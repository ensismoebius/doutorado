#include "nn/tensor/Tensor.hpp"

#include <span>

#include "nn/tensor/TensorBackendFactory.hpp"

namespace nn
{

// ---------------------------------------------------------------------------
// Tensor - Implementation
// This file implements the small, high-level `Tensor` wrapper which delegates
// storage and numerical operations to a pluggable `ITensorBackend` (default
// provided by `TensorBackendFactory`). The implementation below is intentionally
// thin: it forwards calls to the backend and provides convenient factory
// methods and operator overloads used by the rest of the codebase.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------
// Construct an empty tensor backed by the default backend.
Tensor::Tensor() : m_backend(TensorBackendFactory::create_backend()) {}

// Construct a tensor from an existing backend (takes ownership).
Tensor::Tensor(std::unique_ptr<ITensorBackend> backend) : m_backend(std::move(backend)) {}

// Construct common shapes. These allocate storage via the backend.
Tensor::Tensor(Index rows, Index cols) : m_backend(TensorBackendFactory::create_backend())
{
    m_backend->construct(rows, cols);
}

Tensor::Tensor(Index dim1, Index dim2, Index dim3, Index dim4)
    : m_backend(TensorBackendFactory::create_backend())
{
    m_backend->construct(dim1, dim2, dim3, dim4);
}

Tensor::Tensor(const std::vector<Index>& shape) : m_backend(TensorBackendFactory::create_backend())
{
    m_backend->construct(shape);
}

// ---------------------------------------------------------------------------
// Factory helpers
// Convenient static constructors for common initialized tensors.
// ---------------------------------------------------------------------------
auto Tensor::constant(Index rows, Index cols, float value) -> Tensor
{
    Tensor t(rows, cols);
    t.fill(value);
    return t;
}

auto Tensor::zeros(Index rows, Index cols) -> Tensor
{
    Tensor t(rows, cols);
    t.set_zero();
    return t;
}

auto Tensor::ones(Index rows, Index cols) -> Tensor
{
    Tensor t(rows, cols);
    t.set_ones();
    return t;
}

// ---------------------------------------------------------------------------
// Copy / assignment
// The tensor performs deep copies of the backend via `clone()` to preserve
// value semantics while keeping the backend implementation opaque.
// ---------------------------------------------------------------------------
Tensor::Tensor(const Tensor& other)
{
    if (other.m_backend)
    {
        m_backend = other.m_backend->clone();
    }
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if (this != &other)
    {
        if (other.m_backend)
        {
            m_backend = other.m_backend->clone();
        }
        else
        {
            m_backend.reset();
        }
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Shape and size information
// These methods forward to the backend which owns the concrete shape info.
// ---------------------------------------------------------------------------
auto Tensor::get_shape() const -> const std::vector<Index>&
{
    return m_backend->shape();
}

void Tensor::reshape(const std::vector<Index>& new_shape)
{
    m_backend->reshape(new_shape);
}

auto Tensor::rows() const noexcept -> Index
{
    return m_backend->rows();
}

auto Tensor::cols() const noexcept -> Index
{
    return m_backend->cols();
}

auto Tensor::size() const noexcept -> Index
{
    return m_backend->size();
}

// ---------------------------------------------------------------------------
// Element accessors
// Thin forwarding accessors for common indexing patterns. Bounds checking and
// indexing semantics are implemented by the backend; Tensor simply delegates.
// ---------------------------------------------------------------------------
auto Tensor::at(Index row, Index col) -> float&
{
    return m_backend->at(row, col);
}

auto Tensor::at(Index row, Index col) const -> const float&
{
    return m_backend->at(row, col);
}

auto Tensor::at(Index d1, Index d2, Index d3, Index d4) -> float&
{
    return m_backend->at(d1, d2, d3, d4);
}

auto Tensor::at(Index d1, Index d2, Index d3, Index d4) const -> const float&
{
    return m_backend->at(d1, d2, d3, d4);
}

// 1D access
auto Tensor::at(Index i) -> float&
{
    return m_backend->at(i);
}

auto Tensor::at(Index i) const -> const float&
{
    return m_backend->at(i);
}

// General N-D access (delegates to backend's span-based implementation)
auto Tensor::at(const std::vector<Index>& indices) -> float&
{
    return m_backend->at(indices);
}

auto Tensor::at(const std::vector<Index>& indices) const -> const float&
{
    return m_backend->at(indices);
}

// ---------------------------------------------------------------------------
// Views and block operations
// Return new `Tensor` instances that own the extracted sub-blocks or columns/rows.
// ---------------------------------------------------------------------------
auto Tensor::row(Index i) const -> Tensor
{
    return Tensor(m_backend->row(i));
}

auto Tensor::col(Index j) const -> Tensor
{
    return Tensor(m_backend->col(j));
}

auto Tensor::leftCols(Index n) const -> Tensor
{
    return Tensor(m_backend->leftCols(n));
}

auto Tensor::topRows(Index n) const -> Tensor
{
    return Tensor(m_backend->topRows(n));
}

auto Tensor::block(Index row, Index col, Index rows, Index cols) const -> Tensor
{
    return Tensor(m_backend->block(row, col, rows, cols));
}

void Tensor::setBlock(Index row, Index col, const Tensor& block)
{
    m_backend->setBlock(row, col, *block.m_backend);
}

// ---------------------------------------------------------------------------
// Element-wise operations
// These create new `Tensor` results by delegating to the backend.
// ---------------------------------------------------------------------------
auto Tensor::add(const Tensor& other) const -> Tensor
{
    return Tensor(m_backend->add(*other.m_backend));
}

auto Tensor::multiply(const Tensor& other) const -> Tensor
{
    return Tensor(m_backend->multiply(*other.m_backend));
}

auto Tensor::add_scalar(float scalar) const -> Tensor
{
    Tensor result = *this;
    result.m_backend->add_scalar(scalar);
    return result;
}

auto Tensor::multiply_scalar(float scalar) const -> Tensor
{
    Tensor result = *this;
    result.m_backend->multiply_scalar(scalar);
    return result;
}

// ---------------------------------------------------------------------------
// Matrix / linear algebra operations
// ---------------------------------------------------------------------------
auto Tensor::matmul(const Tensor& other) const -> Tensor
{
    return Tensor(m_backend->matmul(*other.m_backend));
}

auto Tensor::transpose() const -> Tensor
{
    return Tensor(m_backend->transpose());
}

// ---------------------------------------------------------------------------
// Activations, losses and reductions
// Simple forwarding helpers for common neural-network operations.
// ---------------------------------------------------------------------------
auto Tensor::relu() const -> Tensor
{
    return Tensor(m_backend->relu());
}

auto Tensor::leaky_relu(float alpha) const -> Tensor
{
    return Tensor(m_backend->leaky_relu(alpha));
}

auto Tensor::mean_squared_error(const Tensor& target) const -> float
{
    return m_backend->mean_squared_error(*target.m_backend);
}

auto Tensor::norm() const -> float
{
    return m_backend->norm();
}

auto Tensor::sum() const -> float
{
    return m_backend->sum();
}

auto Tensor::sum_rows() const -> Tensor
{
    return Tensor(m_backend->sum_rows());
}

auto Tensor::sum_cols() const -> Tensor
{
    return Tensor(m_backend->sum_cols());
}
// ---------------------------------------------------------------------------
// Slice & gradient helpers
// `slice` creates a new tensor by selecting rows (backend-defined). `zero_grad`
// forwards gradient clearing to the backend.
// ---------------------------------------------------------------------------
auto Tensor::slice(std::span<const int> indices) const -> Tensor
{
    if (!m_backend)
    {
        throw std::runtime_error("Tensor backend is null for slice operation.");
    }
    return Tensor(m_backend->slice(indices));
}

void Tensor::zero_grad()
{
    m_backend->zero_grad();
}

// Operator overload implementations
auto Tensor::operator-(const Tensor& other) const -> Tensor
{
    // Subtraction: this - other = this + (-1 * other)
    return add(other.multiply_scalar(-1.0f));
}

auto Tensor::operator*(float scalar) const -> Tensor
{
    return multiply_scalar(scalar);
}

auto Tensor::operator+(float scalar) const -> Tensor
{
    return add_scalar(scalar);
}

auto Tensor::operator-(float scalar) const -> Tensor
{
    return add_scalar(-scalar);
}

auto Tensor::operator/(float scalar) const -> Tensor
{
    return divide_scalar(scalar);
}

// Element-wise math operations
auto Tensor::sqrt() const -> Tensor
{
    return Tensor(m_backend->sqrt());
}

auto Tensor::square() const -> Tensor
{
    return Tensor(m_backend->square());
}

auto Tensor::abs() const -> Tensor
{
    return Tensor(m_backend->abs());
}

auto Tensor::divide(const Tensor& other) const -> Tensor
{
    return Tensor(m_backend->divide(*other.m_backend));
}

auto Tensor::divide_scalar(float scalar) const -> Tensor
{
    return Tensor(m_backend->divide_scalar(scalar));
}

// Initialization
void Tensor::fill(float value)
{
    m_backend->fill(value);
}

void Tensor::set_zero()
{
    m_backend->set_zero();
}

void Tensor::set_ones()
{
    m_backend->set_ones();
}

// Data pointer access
const float* Tensor::data_ptr() const
{
    return m_backend->data_ptr();
}

float* Tensor::mutable_data_ptr()
{
    return m_backend->mutable_data_ptr();
}

// Gradient access
auto Tensor::grad() const -> Tensor
{
    // Return a tensor wrapping the gradient backend
    return Tensor(m_backend->grad().clone());
}

auto Tensor::grad() -> Tensor
{
    // For mutable access, we need to return a reference
    // This is tricky with the current design - for now create a static wrapper
    return Tensor(m_backend->grad().clone());
}

void Tensor::set_grad(const Tensor& new_grad)
{
    m_backend->set_grad(*new_grad.m_backend);
}

auto Tensor::hasNaN() const -> bool
{
    return m_backend->hasNaN();
}

auto Tensor::operator==(const Tensor& other) const -> bool
{
    return m_backend->equals(*other.m_backend);
}

auto Tensor::operator!=(const Tensor& other) const -> bool
{
    return !(*this == other);
}

} // namespace nn