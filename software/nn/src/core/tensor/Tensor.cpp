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
    : m_data(Eigen::MatrixXf::Zero(dim1 * dim2 * dim3 * dim4, 1)),
      m_grad(Eigen::MatrixXf::Zero(dim1 * dim2 * dim3 * dim4, 1)),
      m_shape({dim1, dim2, dim3, dim4})
{
}

Tensor::Tensor(const std::vector<Eigen::Index>& shape)
    : m_data(Eigen::MatrixXf::Zero(calculate_total_size(shape), 1)),
      m_grad(Eigen::MatrixXf::Zero(calculate_total_size(shape), 1)),
      m_shape(shape)
{
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
auto Tensor::get_shape() const -> const std::vector<Eigen::Index>&
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
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("at(row, col) is only valid for 2D tensors");
    }
    if (row < 0 || row >= m_shape[0] || col < 0 || col >= m_shape[1])
    {
        throw std::out_of_range("Index out of range");
    }
    return m_data(row, col);
}

const float& Tensor::at(Eigen::Index row, Eigen::Index col) const
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("at(row, col) is only valid for 2D tensors");
    }
    if (row < 0 || row >= m_shape[0] || col < 0 || col >= m_shape[1])
    {
        throw std::out_of_range("Index out of range");
    }
    return m_data(row, col);
}

float& Tensor::at(Eigen::Index d1, Eigen::Index d2, Eigen::Index d3, Eigen::Index d4)
{
    if (m_shape.size() != 4)
    {
        throw std::invalid_argument("at(d1, d2, d3, d4) is only valid for 4D tensors");
    }
    if (d1 < 0 || d1 >= m_shape[0] || d2 < 0 || d2 >= m_shape[1] || d3 < 0 || d3 >= m_shape[2] ||
        d4 < 0 || d4 >= m_shape[3])
    {
        throw std::out_of_range("Index out of range");
    }
    Eigen::Index channels = m_shape[1];
    Eigen::Index height = m_shape[2];
    Eigen::Index width = m_shape[3];

    Eigen::Index index =
        (d1 * (channels * height * width)) + (d2 * (height * width)) + (d3 * width) + d4;
    return m_data(index, 0);
}

const float& Tensor::at(Eigen::Index d1, Eigen::Index d2, Eigen::Index d3, Eigen::Index d4) const
{
    if (m_shape.size() != 4)
    {
        throw std::invalid_argument("at(d1, d2, d3, d4) is only valid for 4D tensors");
    }
    if (d1 < 0 || d1 >= m_shape[0] || d2 < 0 || d2 >= m_shape[1] || d3 < 0 || d3 >= m_shape[2] ||
        d4 < 0 || d4 >= m_shape[3])
    {
        throw std::out_of_range("Index out of range");
    }
    Eigen::Index channels = m_shape[1];
    Eigen::Index height = m_shape[2];
    Eigen::Index width = m_shape[3];

    Eigen::Index index =
        (d1 * (channels * height * width)) + (d2 * (height * width)) + (d3 * width) + d4;
    return m_data(index, 0);
}

// General N-D access
float& Tensor::at(const std::vector<Eigen::Index>& indices)
{
    if (indices.size() != m_shape.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    Eigen::Index index = 0;
    Eigen::Index stride = 1;
    for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i)
    {
        if (indices[static_cast<size_t>(i)] < 0 ||
            indices[static_cast<size_t>(i)] >= m_shape[static_cast<size_t>(i)])
        {
            throw std::out_of_range("Index out of range");
        }
        index += indices[static_cast<size_t>(i)] * stride;
        stride *= m_shape[static_cast<size_t>(i)];
    }
    return m_data(index, 0);
}

const float& Tensor::at(const std::vector<Eigen::Index>& indices) const
{
    if (indices.size() != m_shape.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    Eigen::Index index = 0;
    Eigen::Index stride = 1;
    for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i)
    {
        if (indices[static_cast<size_t>(i)] < 0 ||
            indices[static_cast<size_t>(i)] >= m_shape[static_cast<size_t>(i)])
        {
            throw std::out_of_range("Index out of range");
        }
        index += indices[static_cast<size_t>(i)] * stride;
        stride *= m_shape[static_cast<size_t>(i)];
    }
    return m_data(index, 0);
}

// Row and column access
auto Tensor::row(Eigen::Index i) const -> Tensor
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("row() only valid for 2D tensors");
    }
    if (i < 0 || i >= m_shape[0])
    {
        throw std::out_of_range("Row index out of range");
    }
    Eigen::MatrixXf row_data = m_data.row(i);
    return Tensor(row_data);
}

auto Tensor::col(Eigen::Index j) const -> Tensor
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("col() only valid for 2D tensors");
    }
    if (j < 0 || j >= m_shape[1])
    {
        throw std::out_of_range("Column index out of range");
    }
    Eigen::MatrixXf col_data = m_data.col(j);
    return Tensor(col_data);
}

auto Tensor::leftCols(Eigen::Index n) const -> Tensor
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("leftCols() only valid for 2D tensors");
    }
    if (n < 0 || n > m_shape[1])
    {
        throw std::out_of_range("Invalid number of columns");
    }
    Eigen::MatrixXf cols_data = m_data.leftCols(n);
    return Tensor(cols_data);
}

auto Tensor::topRows(Eigen::Index n) const -> Tensor
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("topRows() only valid for 2D tensors");
    }
    if (n < 0 || n > m_shape[0])
    {
        throw std::out_of_range("Invalid number of rows");
    }
    Eigen::MatrixXf rows_data = m_data.topRows(n);
    return Tensor(rows_data);
}

// Block operations
auto Tensor::block(Eigen::Index row, Eigen::Index col, Eigen::Index rows, Eigen::Index cols) const
    -> Tensor
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("block() only valid for 2D tensors");
    }
    if (row < 0 || col < 0 || rows < 0 || cols < 0 || row + rows > m_shape[0] ||
        col + cols > m_shape[1])
    {
        throw std::out_of_range("Block dimensions out of range");
    }
    Eigen::MatrixXf block_data = m_data.block(row, col, rows, cols);
    return Tensor(block_data);
}

void Tensor::setBlock(Eigen::Index row, Eigen::Index col, const Tensor& block)
{
    if (m_shape.size() != 2 || block.get_shape().size() != 2)
    {
        throw std::invalid_argument("setBlock() only valid for 2D tensors");
    }
    if (row < 0 || col < 0 || row + block.get_shape()[0] > m_shape[0] ||
        col + block.get_shape()[1] > m_shape[1])
    {
        throw std::out_of_range("Block position out of range");
    }
    m_data.block(row, col, block.get_shape()[0], block.get_shape()[1]) = block.get_data_ref();
}

// Element-wise operations
auto Tensor::add(const Tensor& other) const -> Tensor
{
    if (m_shape != other.get_shape())
    {
        throw std::invalid_argument("Shape mismatch in add");
    }
    Eigen::MatrixXf result = m_data + other.get_data_ref();
    return Tensor(result);
}

auto Tensor::multiply(const Tensor& other) const -> Tensor
{
    if (m_shape != other.get_shape())
    {
        throw std::invalid_argument("Shape mismatch in multiply");
    }
    Eigen::MatrixXf result = m_data.cwiseProduct(other.get_data_ref());
    return Tensor(result);
}

auto Tensor::add_scalar(float scalar) -> Tensor&
{
    m_data.array() += scalar;
    return *this;
}

auto Tensor::multiply_scalar(float scalar) -> Tensor&
{
    m_data.array() *= scalar;
    return *this;
}

// Matrix operations
auto Tensor::matmul(const Tensor& other) const -> Tensor
{
    if (m_shape.size() != 2 || other.get_shape().size() != 2)
    {
        throw std::invalid_argument("matmul() only valid for 2D tensors");
    }
    if (m_shape[1] != other.get_shape()[0])
    {
        throw std::invalid_argument("Matrix dimension mismatch for matmul");
    }
    Eigen::MatrixXf result = m_data * other.get_data_ref();
    return Tensor(result);
}

auto Tensor::transpose() const -> Tensor
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("transpose() only valid for 2D tensors");
    }
    Eigen::MatrixXf result = m_data.transpose();
    return Tensor(result);
}

// Activation functions
auto Tensor::relu() const -> Tensor
{
    Eigen::MatrixXf result = m_data.array().max(0.0f);
    return Tensor(result);
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