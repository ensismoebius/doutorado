#include "EigenTensorBackend.hpp"

#include <cassert>
#include <stdexcept>

namespace nn
{

// Helper function
Index EigenTensorBackend::calculate_total_size(std::span<const Index> shape)
{
    Index total = 1;
    for (Index dim : shape)
    {
        total *= dim;
    }
    return total;
}

// Constructors
EigenTensorBackend::EigenTensorBackend() : m_data(), m_grad_backend(nullptr), m_shape() {}

EigenTensorBackend::EigenTensorBackend(const Eigen::MatrixXf& data)
    : m_data(data),
      m_grad_backend(nullptr),
      m_shape({static_cast<Index>(data.rows()), static_cast<Index>(data.cols())})
{
}

EigenTensorBackend::EigenTensorBackend(Eigen::MatrixXf&& data)
    : m_data(std::move(data)),
      m_grad_backend(nullptr),
      m_shape({static_cast<Index>(m_data.rows()), static_cast<Index>(m_data.cols())})
{
}

// Construction methods
void EigenTensorBackend::construct(Index rows, Index cols)
{
    m_data =
        Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
    m_grad_backend = nullptr;
    m_shape = {rows, cols};
}

void EigenTensorBackend::construct(Index d1, Index d2, Index d3, Index d4)
{
    Index total_size = d1 * d2 * d3 * d4;
    m_data = Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(total_size), 1);
    m_grad_backend = nullptr;
    m_shape = {d1, d2, d3, d4};
}

void EigenTensorBackend::construct(std::span<const Index> shape)
{
    if (shape.size() == 2)
    {
        construct(shape[0], shape[1]);
        return;
    }
    Index total_size = calculate_total_size(shape);
    m_data = Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(total_size), 1);
    m_grad_backend = nullptr;
    m_shape.assign(shape.begin(), shape.end());
}

// Data access methods
float& EigenTensorBackend::at(Index row, Index col)
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("at(row, col) is only valid for 2D tensors");
    }
    if (row >= m_shape[0] || col >= m_shape[1])
    {
        throw std::out_of_range("Index out of range");
    }
    return m_data(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
}

const float& EigenTensorBackend::at(Index row, Index col) const
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("at(row, col) is only valid for 2D tensors");
    }
    if (row >= m_shape[0] || col >= m_shape[1])
    {
        throw std::out_of_range("Index out of range");
    }
    return m_data(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
}

float& EigenTensorBackend::at(Index d1, Index d2, Index d3, Index d4)
{
    if (m_shape.size() != 4)
    {
        throw std::invalid_argument("at(d1, d2, d3, d4) is only valid for 4D tensors");
    }
    if (d1 >= m_shape[0] || d2 >= m_shape[1] || d3 >= m_shape[2] || d4 >= m_shape[3])
    {
        throw std::out_of_range("Index out of range");
    }
    Index channels = m_shape[1];
    Index height = m_shape[2];
    Index width = m_shape[3];
    Index index = (d1 * (channels * height * width)) + (d2 * (height * width)) + (d3 * width) + d4;
    return m_data(static_cast<Eigen::Index>(index));
}

const float& EigenTensorBackend::at(Index d1, Index d2, Index d3, Index d4) const
{
    if (m_shape.size() != 4)
    {
        throw std::invalid_argument("at(d1, d2, d3, d4) is only valid for 4D tensors");
    }
    if (d1 >= m_shape[0] || d2 >= m_shape[1] || d3 >= m_shape[2] || d4 >= m_shape[3])
    {
        throw std::out_of_range("Index out of range");
    }
    Index channels = m_shape[1];
    Index height = m_shape[2];
    Index width = m_shape[3];
    Index index = (d1 * (channels * height * width)) + (d2 * (height * width)) + (d3 * width) + d4;
    return m_data(static_cast<Eigen::Index>(index));
}

// 1D access
float& EigenTensorBackend::at(Index i)
{
    if (i >= static_cast<Index>(m_data.size()))
    {
        throw std::out_of_range("Index out of range");
    }
    return m_data(static_cast<Eigen::Index>(i));
}

const float& EigenTensorBackend::at(Index i) const
{
    if (i >= static_cast<Index>(m_data.size()))
    {
        throw std::out_of_range("Index out of range");
    }
    return m_data(static_cast<Eigen::Index>(i));
}

// N-D access
float& EigenTensorBackend::at(std::span<const Index> indices)
{
    if (indices.size() != m_shape.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    Index index = 0;
    Index stride = 1;
    for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i)
    {
        if (indices[static_cast<size_t>(i)] >= m_shape[static_cast<size_t>(i)])
        {
            throw std::out_of_range("Index out of range");
        }
        index += indices[static_cast<size_t>(i)] * stride;
        stride *= m_shape[static_cast<size_t>(i)];
    }
    return m_data(static_cast<Eigen::Index>(index), 0);
}

const float& EigenTensorBackend::at(std::span<const Index> indices) const
{
    if (indices.size() != m_shape.size())
    {
        throw std::invalid_argument("Number of indices must match tensor dimensions");
    }
    Index index = 0;
    Index stride = 1;
    for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i)
    {
        if (indices[static_cast<size_t>(i)] >= m_shape[static_cast<size_t>(i)])
        {
            throw std::out_of_range("Index out of range");
        }
        index += indices[static_cast<size_t>(i)] * stride;
        stride *= m_shape[static_cast<size_t>(i)];
    }
    return m_data(static_cast<Eigen::Index>(index), 0);
}

// Shape and size methods
const std::vector<Index>& EigenTensorBackend::shape() const
{
    return m_shape;
}

Index EigenTensorBackend::rows() const
{
    if (m_shape.empty()) return 0;
    return m_shape[0];
}

Index EigenTensorBackend::cols() const
{
    if (m_shape.size() < 2) return 0; // Or handle as 1 for 1D vectors
    return m_shape[1];                // For 2D tensors, this is the column count
}

Index EigenTensorBackend::size() const
{
    return static_cast<Index>(m_data.size());
}

void EigenTensorBackend::reshape(std::span<const Index> new_shape)
{
    if (calculate_total_size(new_shape) != size())
    {
        throw std::invalid_argument("New shape size must match old shape size");
    }
    m_shape.assign(new_shape.begin(), new_shape.end());
}

// Row/column operations
std::unique_ptr<ITensorBackend> EigenTensorBackend::row(Index i) const
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("row() only valid for 2D tensors");
    }
    if (i >= m_shape[0])
    {
        throw std::out_of_range("Row index out of range");
    }
    Eigen::MatrixXf row_data = m_data.row(static_cast<Eigen::Index>(i));
    return std::make_unique<EigenTensorBackend>(row_data);
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::col(Index j) const
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("col() only valid for 2D tensors");
    }
    if (j >= m_shape[1])
    {
        throw std::out_of_range("Column index out of range");
    }
    Eigen::MatrixXf col_data = m_data.col(static_cast<Eigen::Index>(j));
    return std::make_unique<EigenTensorBackend>(col_data);
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::leftCols(Index n) const
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("leftCols() only valid for 2D tensors");
    }
    if (n > m_shape[1])
    {
        throw std::out_of_range("Invalid number of columns");
    }
    Eigen::MatrixXf cols_data = m_data.leftCols(static_cast<Eigen::Index>(n));
    return std::make_unique<EigenTensorBackend>(cols_data);
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::topRows(Index n) const
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("topRows() only valid for 2D tensors");
    }
    if (n > m_shape[0])
    {
        throw std::out_of_range("Invalid number of rows");
    }
    Eigen::MatrixXf rows_data = m_data.topRows(static_cast<Eigen::Index>(n));
    return std::make_unique<EigenTensorBackend>(rows_data);
}

// Block operations
std::unique_ptr<ITensorBackend> EigenTensorBackend::block(Index row, Index col, Index rows,
                                                          Index cols) const
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("block() only valid for 2D tensors");
    }
    if (row + rows > m_shape[0] || col + cols > m_shape[1])
    {
        throw std::out_of_range("Block dimensions out of range");
    }
    Eigen::MatrixXf block_data = m_data.block(static_cast<Eigen::Index>(row),
                                              static_cast<Eigen::Index>(col),
                                              static_cast<Eigen::Index>(rows),
                                              static_cast<Eigen::Index>(cols));
    return std::make_unique<EigenTensorBackend>(block_data);
}

void EigenTensorBackend::setBlock(Index row, Index col, const ITensorBackend& block)
{
    if (m_shape.size() != 2 || block.shape().size() != 2)
    {
        throw std::invalid_argument("setBlock() only valid for 2D tensors");
    }
    if (row + block.shape()[0] > m_shape[0] || col + block.shape()[1] > m_shape[1])
    {
        // Treat oversized blocks as invalid argument to align with test expectations
        throw std::invalid_argument("Block dimensions do not fit target tensor");
    }
    const auto& eigen_block = dynamic_cast<const EigenTensorBackend&>(block);
    m_data.block(static_cast<Eigen::Index>(row),
                 static_cast<Eigen::Index>(col),
                 static_cast<Eigen::Index>(block.shape()[0]),
                 static_cast<Eigen::Index>(block.shape()[1])) = eigen_block.m_data;
}

// Element-wise operations
std::unique_ptr<ITensorBackend> EigenTensorBackend::add(const ITensorBackend& other) const
{
    if (m_shape != other.shape())
    {
        throw std::invalid_argument("Shape mismatch in add");
    }
    const auto& eigen_other = dynamic_cast<const EigenTensorBackend&>(other);
    Eigen::MatrixXf result = m_data + eigen_other.m_data;
    return std::make_unique<EigenTensorBackend>(result);
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::multiply(const ITensorBackend& other) const
{
    if (m_shape != other.shape())
    {
        throw std::invalid_argument("Shape mismatch in multiply");
    }
    const auto& eigen_other = dynamic_cast<const EigenTensorBackend&>(other);
    Eigen::MatrixXf result = m_data.cwiseProduct(eigen_other.m_data);
    return std::make_unique<EigenTensorBackend>(result);
}

void EigenTensorBackend::add_scalar(float scalar)
{
    m_data.array() += scalar;
}

void EigenTensorBackend::multiply_scalar(float scalar)
{
    m_data.array() *= scalar;
}

// Matrix operations
std::unique_ptr<ITensorBackend> EigenTensorBackend::matmul(const ITensorBackend& other) const
{
    if (m_shape.size() != 2 || other.shape().size() != 2)
    {
        throw std::invalid_argument("matmul() only valid for 2D tensors");
    }
    if (m_shape[1] != other.shape()[0])
    {
        throw std::invalid_argument("Matrix dimension mismatch for matmul");
    }
    const auto& eigen_other = dynamic_cast<const EigenTensorBackend&>(other);
    Eigen::MatrixXf result = m_data * eigen_other.m_data;
    return std::make_unique<EigenTensorBackend>(result);
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::transpose() const
{
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("transpose() only valid for 2D tensors");
    }
    Eigen::MatrixXf result = m_data.transpose();
    return std::make_unique<EigenTensorBackend>(result);
}

// Activation functions
std::unique_ptr<ITensorBackend> EigenTensorBackend::relu() const
{
    Eigen::MatrixXf result = m_data.array().max(0.0f);
    return std::make_unique<EigenTensorBackend>(result);
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::leaky_relu(float alpha) const
{
    Eigen::MatrixXf result = m_data.array().max(0.0f) + (m_data.array().min(0.0f) * alpha);
    return std::make_unique<EigenTensorBackend>(result);
}

// Loss functions
float EigenTensorBackend::mean_squared_error(const ITensorBackend& target) const
{
    if (m_shape != target.shape())
    {
        throw std::invalid_argument("Shape mismatch in mean_squared_error");
    }
    const auto& eigen_target = dynamic_cast<const EigenTensorBackend&>(target);
    Eigen::MatrixXf diff = m_data - eigen_target.m_data;
    float sum_squared = diff.array().square().sum();
    return sum_squared / static_cast<float>(m_data.size());
}

float EigenTensorBackend::norm() const
{
    return m_data.norm();
}

float EigenTensorBackend::sum() const
{
    return m_data.sum();
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::sum_rows() const
{
    // Sum across columns, return column vector (rows, 1)
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("sum_rows() only valid for 2D tensors");
    }
    Eigen::VectorXf summed = m_data.rowwise().sum();
    Eigen::MatrixXf result(summed.size(), 1);
    result.col(0) = summed;
    return std::make_unique<EigenTensorBackend>(result);
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::sum_cols() const
{
    // Sum across rows, return row vector (1, cols)
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument("sum_cols() only valid for 2D tensors");
    }
    Eigen::RowVectorXf summed = m_data.colwise().sum();
    Eigen::MatrixXf result(1, summed.size());
    result.row(0) = summed;
    return std::make_unique<EigenTensorBackend>(result);
}

// Gradient operations
void EigenTensorBackend::zero_grad()
{
    if (!m_grad_backend)
    {
        m_grad_backend = std::make_unique<EigenTensorBackend>();
        m_grad_backend->construct(m_shape);
    }
    else if (m_grad_backend->shape() != m_shape)
    {
        m_grad_backend->construct(m_shape);
    }
    // Zero out the gradient data
    m_grad_backend->m_data.setZero();
}

void EigenTensorBackend::set_grad(const ITensorBackend& grad)
{
    const auto& eigen_grad = dynamic_cast<const EigenTensorBackend&>(grad);
    if (!m_grad_backend)
    {
        m_grad_backend = std::make_unique<EigenTensorBackend>();
    }
    m_grad_backend->m_data = eigen_grad.m_data;
    m_grad_backend->m_shape = eigen_grad.m_shape;
}

const ITensorBackend& EigenTensorBackend::grad() const
{
    if (!m_grad_backend)
    {
        // Lazily initialize m_grad_backend if it doesn't exist.
        // The 'mutable' keyword on m_grad_backend allows this modification in a const method.
        // We ensure it's properly sized and zeroed.
        // Call the non-const zero_grad, which will correctly set up m_grad_backend
        // This const_cast is safe because m_grad_backend is mutable.
        const_cast<EigenTensorBackend*>(this)->zero_grad();
    }
    // At this point m_grad_backend is guaranteed to be non-null
    assert(m_grad_backend != nullptr);
    return *m_grad_backend;
}

ITensorBackend& EigenTensorBackend::grad()
{
    if (!m_grad_backend)
    {
        zero_grad();
    }
    return *m_grad_backend;
}

// Utility methods
std::unique_ptr<ITensorBackend> EigenTensorBackend::clone() const
{
    auto backend = std::make_unique<EigenTensorBackend>(m_data);
    backend->m_shape = m_shape; // Preserve the logical shape
    if (m_grad_backend)
    {
        backend->m_grad_backend = std::unique_ptr<EigenTensorBackend>(
            static_cast<EigenTensorBackend*>(m_grad_backend->clone().release()));
        backend->m_grad_backend->m_shape = m_grad_backend->m_shape;
    }
    return backend;
}

void EigenTensorBackend::copy_from(const ITensorBackend& other)
{
    const auto& eigen_other = dynamic_cast<const EigenTensorBackend&>(other);
    m_data = eigen_other.m_data;
    m_shape = eigen_other.m_shape;
    if (eigen_other.m_grad_backend)
    {
        if (!m_grad_backend)
        {
            m_grad_backend = std::make_unique<EigenTensorBackend>();
        }
        m_grad_backend->copy_from(*eigen_other.m_grad_backend);
    }
    else
    {
        m_grad_backend = nullptr;
    }
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::slice(std::span<const int> indices) const
{
    // Ensure the tensor is 2D for slicing rows
    if (m_shape.size() != 2)
    {
        throw std::invalid_argument(
            "Slicing by indices is only supported for 2D tensors (row selection).");
    }

    // Determine the number of rows in the new sliced tensor
    const Eigen::Index new_rows = indices.size();
    const Eigen::Index num_cols = m_data.cols();

    Eigen::MatrixXf sliced_data(new_rows, num_cols);

    for (Eigen::Index i = 0; i < new_rows; ++i)
    {
        const int original_row_idx = indices[static_cast<std::size_t>(i)];
        if (original_row_idx < 0 || original_row_idx >= static_cast<int>(m_data.rows()))
        {
            throw std::out_of_range("Slice index out of range.");
        }
        sliced_data.row(i) = m_data.row(original_row_idx);
    }

    // Create a new EigenTensorBackend with the sliced data
    return std::make_unique<EigenTensorBackend>(sliced_data);
}

Index EigenTensorBackend::data_rows() const
{
    return m_data.rows();
}

Index EigenTensorBackend::data_cols() const
{
    return m_data.cols();
}

// Element-wise math operations
std::unique_ptr<ITensorBackend> EigenTensorBackend::sqrt() const
{
    Eigen::MatrixXf result = m_data.array().sqrt().matrix();
    return std::make_unique<EigenTensorBackend>(result);
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::square() const
{
    Eigen::MatrixXf result = m_data.array().square().matrix();
    return std::make_unique<EigenTensorBackend>(result);
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::abs() const
{
    Eigen::MatrixXf result = m_data.array().abs().matrix();
    return std::make_unique<EigenTensorBackend>(result);
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::divide(const ITensorBackend& other) const
{
    const auto* other_eigen = dynamic_cast<const EigenTensorBackend*>(&other);
    if (!other_eigen)
    {
        throw std::invalid_argument("Backend type mismatch in divide operation");
    }
    if (m_data.rows() != other_eigen->m_data.rows() || m_data.cols() != other_eigen->m_data.cols())
    {
        throw std::invalid_argument("Shape mismatch in divide operation");
    }
    Eigen::MatrixXf result = m_data.array() / other_eigen->m_data.array();
    return std::make_unique<EigenTensorBackend>(result.matrix());
}

std::unique_ptr<ITensorBackend> EigenTensorBackend::divide_scalar(float scalar) const
{
    if (scalar == 0.0f)
    {
        throw std::invalid_argument("Division by zero");
    }
    Eigen::MatrixXf result = m_data / scalar;
    return std::make_unique<EigenTensorBackend>(result);
}

// Initialization
void EigenTensorBackend::fill(float value)
{
    m_data.fill(value);
}

void EigenTensorBackend::set_zero()
{
    m_data.setZero();
}

void EigenTensorBackend::set_ones()
{
    m_data.setOnes();
}

// Data pointer access
const float* EigenTensorBackend::data_ptr() const
{
    return m_data.data();
}

float* EigenTensorBackend::mutable_data_ptr()
{
    return m_data.data();
}

// Validation and comparison
bool EigenTensorBackend::hasNaN() const
{
    return m_data.hasNaN();
}

bool EigenTensorBackend::equals(const ITensorBackend& other) const
{
    const auto* eigen_other = dynamic_cast<const EigenTensorBackend*>(&other);
    if (!eigen_other)
    {
        return false;
    }
    if (m_data.rows() != eigen_other->m_data.rows() || m_data.cols() != eigen_other->m_data.cols())
    {
        return false;
    }
    return m_data.isApprox(eigen_other->m_data);
}

} // namespace nn