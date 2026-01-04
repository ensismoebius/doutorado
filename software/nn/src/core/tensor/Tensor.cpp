#include "Tensor.hpp"

#include <span>

#include "TensorBackendFactory.hpp"

namespace nn
{

// Constructors
Tensor::Tensor() : m_backend(TensorBackendFactory::create_backend()) {}

Tensor::Tensor(std::unique_ptr<ITensorBackend> backend) : m_backend(std::move(backend)) {}

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

// Static factory methods for creating initialized tensors
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

// Copy constructor
Tensor::Tensor(const Tensor& other)
{
    if (other.m_backend)
    {
        m_backend = other.m_backend->clone();
    }
}

// Copy assignment operator
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

// Shape and size information
auto Tensor::get_shape() const -> const std::vector<Index>&
{
    return m_backend->shape();
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

// Element access for 2D and 4D tensors
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

// General N-D access
auto Tensor::at(const std::vector<Index>& indices) -> float&
{
    return m_backend->at(indices);
}

auto Tensor::at(const std::vector<Index>& indices) const -> const float&
{
    return m_backend->at(indices);
}

// Row and column access
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

// Block operations
auto Tensor::block(Index row, Index col, Index rows, Index cols) const -> Tensor
{
    return Tensor(m_backend->block(row, col, rows, cols));
}

void Tensor::setBlock(Index row, Index col, const Tensor& block)
{
    m_backend->setBlock(row, col, *block.m_backend);
}

// Element-wise operations
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

// Matrix operations
auto Tensor::matmul(const Tensor& other) const -> Tensor
{
    return Tensor(m_backend->matmul(*other.m_backend));
}

auto Tensor::transpose() const -> Tensor
{
    return Tensor(m_backend->transpose());
}

// Activation functions
auto Tensor::relu() const -> Tensor
{
    return Tensor(m_backend->relu());
}

auto Tensor::leaky_relu(float alpha) const -> Tensor
{
    return Tensor(m_backend->leaky_relu(alpha));
}

// Loss functions
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
// Slice operation
auto Tensor::slice(std::span<const int> indices) const -> Tensor
{
    if (!m_backend)
    {
        throw std::runtime_error("Tensor backend is null for slice operation.");
    }
    return Tensor(m_backend->slice(indices));
}

// Zero out the gradient
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

auto Tensor::grad() -> Tensor&
{
    // For mutable access, we need to return a reference
    // This is tricky with the current design - for now create a static wrapper
    static thread_local Tensor grad_wrapper;
    grad_wrapper = Tensor(m_backend->grad().clone());
    return grad_wrapper;
}

void Tensor::set_grad(const Tensor& new_grad)
{
    m_backend->set_grad(*new_grad.m_backend);
}

} // namespace nn