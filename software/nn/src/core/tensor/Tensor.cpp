#include "Tensor.hpp"

#include <span>

namespace nn
{
// Constructors
Tensor::Tensor(const Eigen::MatrixXf& data)
    : m_data(data), m_grad(Eigen::MatrixXf::Zero(data.rows(), data.cols()))
{
}
Tensor::Tensor(Eigen::MatrixXf&& data)
    : m_data(std::move(data)), m_grad(Eigen::MatrixXf::Zero(m_data.rows(), m_data.cols()))
{
}
Tensor::Tensor(Eigen::Index rows, Eigen::Index cols)
    : m_data(Eigen::MatrixXf::Zero(rows, cols)), m_grad(Eigen::MatrixXf::Zero(rows, cols))
{
}

// Getters for data and gradient
auto Tensor::get_data_ref() -> Eigen::MatrixXf&
{
    return m_data;
}
auto Tensor::get_data_ref() const -> const Eigen::MatrixXf&
{
    return m_data;
}
auto Tensor::get_grad_ref() -> Eigen::MatrixXf&
{
    return m_grad;
}
auto Tensor::get_grad_ref() const -> const Eigen::MatrixXf&
{
    return m_grad;
}

// Setter for gradient
void Tensor::set_grad(const Eigen::MatrixXf& grad)
{
    m_grad = grad;
}
void Tensor::set_grad(Eigen::MatrixXf&& grad)
{
    m_grad = std::move(grad);
}

// Setter for data
void Tensor::set_data(const Eigen::MatrixXf& data)
{
    m_data = data;
}

// Shape and size information
auto Tensor::get_shape() const -> std::vector<Eigen::Index>
{
    return {m_data.rows(), m_data.cols()};
}
auto Tensor::rows() const -> Eigen::Index
{
    return m_data.rows();
}
auto Tensor::cols() const -> Eigen::Index
{
    return m_data.cols();
}
auto Tensor::size() const -> Eigen::Index
{
    return m_data.size();
}

// Slice operation
auto Tensor::slice(std::span<const int> indices) const -> Tensor
{
    auto n = static_cast<Eigen::Index>(indices.size());
    Eigen::MatrixXf sliced_data(n, m_data.cols());
    for (Eigen::Index i = 0; i < n; ++i)
    {
        sliced_data.row(i) = m_data.row(static_cast<Eigen::Index>(indices[static_cast<size_t>(i)]));
    }
    return Tensor{sliced_data};
}

void Tensor::zero_grad()
{
    if (m_grad.size() != m_data.size())
    {
        m_grad.resize(m_data.rows(), m_data.cols());
    }
    m_grad.setZero();
}
} // namespace nn
