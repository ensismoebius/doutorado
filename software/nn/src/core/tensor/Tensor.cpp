#include "Tensor.hpp"

#include <span>

namespace nn
{
// Constructors
Tensor::Tensor(const Eigen::MatrixXf& data)
    : m_data(data),
      m_grad(Eigen::MatrixXf::Zero(data.rows(), data.cols())),
      m_shape({data.rows(), data.cols()})
{
}
Tensor::Tensor(Eigen::MatrixXf&& data)
    : m_data(std::move(data)),
      m_grad(Eigen::MatrixXf::Zero(m_data.rows(), m_data.cols())),
      m_shape({m_data.rows(), m_data.cols()})
{
}
Tensor::Tensor(Eigen::Index rows, Eigen::Index cols)
    : m_data(Eigen::MatrixXf::Zero(rows, cols)),
      m_grad(Eigen::MatrixXf::Zero(rows, cols)),
      m_shape({rows, cols})
{
}

Tensor::Tensor(Eigen::Index dim1, Eigen::Index dim2, Eigen::Index dim3, Eigen::Index dim4)
    : m_shape({dim1, dim2, dim3, dim4})
{
    Eigen::Index total_size = dim1 * dim2 * dim3 * dim4;
    m_data = Eigen::MatrixXf::Zero(total_size, 1);
    m_grad = Eigen::MatrixXf::Zero(total_size, 1);
}

Tensor::Tensor(const std::vector<Eigen::Index>& shape) : m_shape(shape)
{
    Eigen::Index total_size = 1;
    for (Eigen::Index dim : shape)
    {
        total_size *= dim;
    }
    m_data = Eigen::MatrixXf::Zero(total_size, 1);
    m_grad = Eigen::MatrixXf::Zero(total_size, 1);
}

// Getters for data and gradient
auto Tensor::get_data_ref() const -> const Eigen::MatrixXf&
{
    return m_data;
}
auto Tensor::get_grad_ref() const -> const Eigen::MatrixXf&
{
    return m_grad;
}
auto Tensor::get_data_ref() -> Eigen::MatrixXf&
{
    return m_data;
}
auto Tensor::get_grad_ref() -> Eigen::MatrixXf&
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
    return m_shape;
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

// Element access for 2D and 4D tensors
float& Tensor::at(Eigen::Index row, Eigen::Index col)
{
    if (m_shape.size() == 2)
    {
        return m_data(row, col);
    }
    return m_data(row, 0);
}

const float& Tensor::at(Eigen::Index row, Eigen::Index col) const
{
    if (m_shape.size() == 2)
    {
        return m_data(row, col);
    }
    return m_data(row, 0);
}

float& Tensor::at(Eigen::Index d1, Eigen::Index d2, Eigen::Index d3, Eigen::Index d4)
{
    Eigen::Index channels = m_shape[1];
    Eigen::Index height = m_shape[2];
    Eigen::Index width = m_shape[3];

    Eigen::Index index =
        (d1 * (channels * height * width)) + (d2 * (height * width)) + (d3 * width) + d4;
    return m_data(index, 0);
}

const float& Tensor::at(Eigen::Index d1, Eigen::Index d2, Eigen::Index d3, Eigen::Index d4) const
{
    Eigen::Index channels = m_shape[1];
    Eigen::Index height = m_shape[2];
    Eigen::Index width = m_shape[3];

    Eigen::Index index =
        (d1 * (channels * height * width)) + (d2 * (height * width)) + (d3 * width) + d4;
    return m_data(index, 0);
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
    // This constructor needs to handle the shape correctly.
    // Assuming if the slice is still 2D, its shape is (n, m_data.cols())
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