#include "Tensor.hpp"

#include <iostream>
#include <span>

#include "EigenTensorBackend.hpp"
#include "TensorBackendFactory.hpp"

namespace nn
{

// Constructors
Tensor::Tensor() : m_backend(TensorBackendFactory::create_backend()) {}

Tensor::Tensor(std::unique_ptr<ITensorBackend> backend) : m_backend(std::move(backend)) {}

Tensor::Tensor(const Eigen::MatrixXf& data)
{
    m_backend = std::make_unique<EigenTensorBackend>(data);
}

Tensor::Tensor(Eigen::MatrixXf&& data)
{
    m_backend = std::make_unique<EigenTensorBackend>(std::move(data));
}

Tensor::Tensor(Index rows, Index cols)
{
    m_backend = TensorBackendFactory::create_backend();
    m_backend->construct(rows, cols);
}

Tensor::Tensor(Index dim1, Index dim2, Index dim3, Index dim4)
{
    m_backend = TensorBackendFactory::create_backend();
    m_backend->construct(dim1, dim2, dim3, dim4);
}

Tensor::Tensor(const std::vector<Index>& shape)
{
    m_backend = TensorBackendFactory::create_backend();
    m_backend->construct(shape);
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

// Getters for data and gradient (backward compatibility - return Eigen matrices)
auto Tensor::get_data_ref() const -> const Eigen::MatrixXf&
{
    // Cast to EigenTensorBackend for backward compatibility
    auto* eigen_backend = dynamic_cast<EigenTensorBackend*>(m_backend.get());
    if (!eigen_backend)
    {
        throw std::runtime_error("Tensor backend is not Eigen-based");
    }
    return eigen_backend->get_data();
}

auto Tensor::get_grad_ref() const -> const Eigen::MatrixXf&
{
    auto* eigen_backend = dynamic_cast<EigenTensorBackend*>(m_backend.get());
    if (!eigen_backend)
    {
        throw std::runtime_error("Tensor backend is not Eigen-based");
    }
    return eigen_backend->get_grad();
}

auto Tensor::get_data_ref() -> Eigen::MatrixXf&
{
    auto* eigen_backend = dynamic_cast<EigenTensorBackend*>(m_backend.get());
    if (!eigen_backend)
    {
        throw std::runtime_error("Tensor backend is not Eigen-based");
    }
    return eigen_backend->get_data();
}

auto Tensor::get_grad_ref() -> Eigen::MatrixXf&
{
    auto* eigen_backend = dynamic_cast<EigenTensorBackend*>(m_backend.get());
    if (!eigen_backend)
    {
        throw std::runtime_error("Tensor backend is not Eigen-based");
    }
    return eigen_backend->get_grad();
}

// Setter for gradient
void Tensor::set_grad(const Eigen::MatrixXf& grad)
{
    // Ensure m_backend exists and is of type EigenTensorBackend
    auto* eigen_backend = dynamic_cast<EigenTensorBackend*>(m_backend.get());
    if (!eigen_backend) {
        throw std::runtime_error("Tensor backend is not Eigen-based for set_grad");
    }
    // Set the gradient directly in the EigenTensorBackend
    eigen_backend->set_grad(EigenTensorBackend(grad));
}

void Tensor::set_grad(Eigen::MatrixXf&& grad)
{
    auto* eigen_backend = dynamic_cast<EigenTensorBackend*>(m_backend.get());
    if (!eigen_backend) {
        throw std::runtime_error("Tensor backend is not Eigen-based for set_grad");
    }
    eigen_backend->set_grad(EigenTensorBackend(std::move(grad)));
}

// Setter for data
void Tensor::set_data(const Eigen::MatrixXf& data)
{
    auto* eigen_backend = dynamic_cast<EigenTensorBackend*>(m_backend.get());
    if (!eigen_backend) {
        throw std::runtime_error("Tensor backend is not Eigen-based for set_data");
    }
    eigen_backend->get_data() = data; // Directly assign to the Eigen::MatrixXf in the backend
}

// Shape and size information
auto Tensor::get_shape() const -> const std::vector<Index>&
{
    return m_backend->shape();
}

auto Tensor::rows() const -> Index
{
    return m_backend->rows();
}

auto Tensor::cols() const -> Index
{
    return m_backend->cols();
}

auto Tensor::size() const -> Index
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

auto Tensor::add_scalar(float scalar) -> Tensor&
{
    m_backend->add_scalar(scalar);
    return *this;
}

auto Tensor::multiply_scalar(float scalar) -> Tensor&
{
    m_backend->multiply_scalar(scalar);
    return *this;
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

} // namespace nn