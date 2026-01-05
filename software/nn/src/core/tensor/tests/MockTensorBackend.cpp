#include "MockTensorBackend.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace nn
{

namespace
{
constexpr float kEpsilon = 1e-6f;
}

MockTensorBackend::MockTensorBackend() : m_data(), m_shape(), m_grad(nullptr), m_calls()
{
    log_call("construct:empty");
}

MockTensorBackend::MockTensorBackend(const std::vector<Index>& shape)
    : m_data(), m_shape(shape), m_grad(nullptr), m_calls()
{
    const Index total = shape.empty() ? 0 : [&]()
    {
        Index t = 1;
        for (Index d : shape) t *= d;
        return t;
    }();
    m_data.assign(static_cast<std::size_t>(total), 0.0f);
    log_call("construct:" + shape_to_string());
}

MockTensorBackend::MockTensorBackend(const std::vector<Index>& shape, std::vector<float> data)
    : m_data(std::move(data)), m_shape(shape), m_grad(nullptr), m_calls()
{
    const Index total = shape.empty() ? 0 : [&]()
    {
        Index t = 1;
        for (Index d : shape) t *= d;
        return t;
    }();
    if (static_cast<Index>(m_data.size()) != total)
    {
        throw std::invalid_argument("Data size does not match shape in MockTensorBackend");
    }
    log_call("construct:data:" + shape_to_string());
}

void MockTensorBackend::construct(Index rows, Index cols)
{
    m_shape = {rows, cols};
    m_data.assign(static_cast<std::size_t>(rows * cols), 0.0f);
    m_grad.reset();
    log_call("construct:" + shape_to_string());
}

void MockTensorBackend::construct(Index d1, Index d2, Index d3, Index d4)
{
    m_shape = {d1, d2, d3, d4};
    m_data.assign(static_cast<std::size_t>(d1 * d2 * d3 * d4), 0.0f);
    m_grad.reset();
    log_call("construct:" + shape_to_string());
}

void MockTensorBackend::construct(const std::vector<Index>& shape)
{
    Index total = 1;
    for (Index d : shape) total *= d;
    m_shape = shape;
    m_data.assign(static_cast<std::size_t>(total), 0.0f);
    m_grad.reset();
    log_call("construct:" + shape_to_string());
}

float& MockTensorBackend::at(Index i)
{
    log_call("at1d-write");
    if (i >= m_data.size())
    {
        throw std::out_of_range("Index out of range");
    }
    return m_data[i];
}

const float& MockTensorBackend::at(Index i) const
{
    log_call("at1d-read");
    if (i >= m_data.size())
    {
        throw std::out_of_range("Index out of range");
    }
    return m_data[i];
}

float& MockTensorBackend::at(Index row, Index col)
{
    ensure_shape(2);
    log_call("at2d-write");
    return m_data.at(static_cast<std::size_t>(offset_2d(row, col)));
}

const float& MockTensorBackend::at(Index row, Index col) const
{
    ensure_shape(2);
    log_call("at2d-read");
    return m_data.at(static_cast<std::size_t>(offset_2d(row, col)));
}

float& MockTensorBackend::at(Index d1, Index d2, Index d3, Index d4)
{
    ensure_shape(4);
    log_call("at4d-write");
    return m_data.at(static_cast<std::size_t>(offset_nd({d1, d2, d3, d4})));
}

const float& MockTensorBackend::at(Index d1, Index d2, Index d3, Index d4) const
{
    ensure_shape(4);
    log_call("at4d-read");
    return m_data.at(static_cast<std::size_t>(offset_nd({d1, d2, d3, d4})));
}

float& MockTensorBackend::at(const std::vector<Index>& indices)
{
    log_call("atnd-write");
    return m_data.at(static_cast<std::size_t>(offset_nd(indices)));
}

const float& MockTensorBackend::at(const std::vector<Index>& indices) const
{
    log_call("atnd-read");
    return m_data.at(static_cast<std::size_t>(offset_nd(indices)));
}

const std::vector<Index>& MockTensorBackend::shape() const
{
    return m_shape;
}

void MockTensorBackend::reshape(const std::vector<Index>& new_shape)
{
    log_call("reshape:" + shape_to_string());
    Index new_size = 1;
    for (Index dim : new_shape)
    {
        new_size *= dim;
    }

    if (new_size != size())
    {
        throw std::invalid_argument("New shape size must match old shape size");
    }
    m_shape = new_shape;
}

Index MockTensorBackend::rows() const
{
    return m_shape.size() >= 1 ? m_shape[0] : 0;
}

Index MockTensorBackend::cols() const
{
    return m_shape.size() >= 2 ? m_shape[1] : 0;
}

Index MockTensorBackend::size() const
{
    return static_cast<Index>(m_data.size());
}

std::unique_ptr<ITensorBackend> MockTensorBackend::row(Index i) const
{
    ensure_shape(2);
    if (i >= rows()) throw std::out_of_range("row index out of range");
    std::vector<float> out(static_cast<std::size_t>(cols()));
    for (Index c = 0; c < cols(); ++c)
    {
        out[static_cast<std::size_t>(c)] = m_data[static_cast<std::size_t>(offset_2d(i, c))];
    }
    log_call("row:" + std::to_string(i));
    return std::make_unique<MockTensorBackend>(std::vector<Index>{1, cols()}, std::move(out));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::col(Index j) const
{
    ensure_shape(2);
    if (j >= cols()) throw std::out_of_range("col index out of range");
    std::vector<float> out(static_cast<std::size_t>(rows()));
    for (Index r = 0; r < rows(); ++r)
    {
        out[static_cast<std::size_t>(r)] = m_data[static_cast<std::size_t>(offset_2d(r, j))];
    }
    log_call("col:" + std::to_string(j));
    return std::make_unique<MockTensorBackend>(std::vector<Index>{rows(), 1}, std::move(out));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::leftCols(Index n) const
{
    ensure_shape(2);
    if (n > cols()) throw std::out_of_range("leftCols out of range");
    std::vector<float> out(static_cast<std::size_t>(rows() * n));
    for (Index r = 0; r < rows(); ++r)
    {
        for (Index c = 0; c < n; ++c)
        {
            out[static_cast<std::size_t>(r * n + c)] =
                m_data[static_cast<std::size_t>(offset_2d(r, c))];
        }
    }
    log_call("leftCols:" + std::to_string(n));
    return std::make_unique<MockTensorBackend>(std::vector<Index>{rows(), n}, std::move(out));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::topRows(Index n) const
{
    ensure_shape(2);
    if (n > rows()) throw std::out_of_range("topRows out of range");
    std::vector<float> out(static_cast<std::size_t>(n * cols()));
    for (Index r = 0; r < n; ++r)
    {
        for (Index c = 0; c < cols(); ++c)
        {
            out[static_cast<std::size_t>(r * cols() + c)] =
                m_data[static_cast<std::size_t>(offset_2d(r, c))];
        }
    }
    log_call("topRows:" + std::to_string(n));
    return std::make_unique<MockTensorBackend>(std::vector<Index>{n, cols()}, std::move(out));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::block(Index row, Index col, Index rows,
                                                         Index cols) const
{
    ensure_shape(2);
    if (row + rows > this->rows() || col + cols > this->cols())
    {
        throw std::out_of_range("block exceeds bounds");
    }
    std::vector<float> out(static_cast<std::size_t>(rows * cols));
    for (Index r = 0; r < rows; ++r)
    {
        for (Index c = 0; c < cols; ++c)
        {
            out[static_cast<std::size_t>(r * cols + c)] =
                m_data[static_cast<std::size_t>(offset_2d(row + r, col + c))];
        }
    }
    log_call("block:" + std::to_string(row) + "," + std::to_string(col));
    return std::make_unique<MockTensorBackend>(std::vector<Index>{rows, cols}, std::move(out));
}

void MockTensorBackend::setBlock(Index row, Index col, const ITensorBackend& block)
{
    ensure_shape(2);
    if (block.shape().size() != 2)
    {
        throw std::invalid_argument("setBlock expects 2D block");
    }
    if (row + block.shape()[0] > rows() || col + block.shape()[1] > cols())
    {
        throw std::invalid_argument("setBlock dimensions do not fit");
    }
    const auto* other = dynamic_cast<const MockTensorBackend*>(&block);
    if (!other)
    {
        throw std::invalid_argument("setBlock requires MockTensorBackend input");
    }
    for (Index r = 0; r < block.shape()[0]; ++r)
    {
        for (Index c = 0; c < block.shape()[1]; ++c)
        {
            m_data[static_cast<std::size_t>(offset_2d(row + r, col + c))] = other->at(r, c);
        }
    }
    log_call("setBlock:" + std::to_string(row) + "," + std::to_string(col));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::add(const ITensorBackend& other) const
{
    ensure_same_shape(other, "add");
    const auto* rhs = dynamic_cast<const MockTensorBackend*>(&other);
    if (!rhs) throw std::invalid_argument("add requires MockTensorBackend");
    std::vector<float> out(m_data.size());
    for (std::size_t i = 0; i < m_data.size(); ++i)
    {
        out[i] = m_data[i] + rhs->m_data[i];
    }
    log_call("add:" + shape_to_string());
    return std::make_unique<MockTensorBackend>(m_shape, std::move(out));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::multiply(const ITensorBackend& other) const
{
    ensure_same_shape(other, "multiply");
    const auto* rhs = dynamic_cast<const MockTensorBackend*>(&other);
    if (!rhs) throw std::invalid_argument("multiply requires MockTensorBackend");
    std::vector<float> out(m_data.size());
    for (std::size_t i = 0; i < m_data.size(); ++i)
    {
        out[i] = m_data[i] * rhs->m_data[i];
    }
    log_call("multiply:" + shape_to_string());
    return std::make_unique<MockTensorBackend>(m_shape, std::move(out));
}

void MockTensorBackend::add_scalar(float scalar)
{
    for (float& v : m_data) v += scalar;
    log_call("add_scalar");
}

void MockTensorBackend::multiply_scalar(float scalar)
{
    for (float& v : m_data) v *= scalar;
    log_call("multiply_scalar");
}

std::unique_ptr<ITensorBackend> MockTensorBackend::matmul(const ITensorBackend& other) const
{
    ensure_shape(2);
    const auto* rhs = dynamic_cast<const MockTensorBackend*>(&other);
    if (!rhs) throw std::invalid_argument("matmul requires MockTensorBackend");
    if (rhs->shape().size() != 2)
    {
        throw std::invalid_argument("matmul expects 2D rhs");
    }
    if (cols() != rhs->rows())
    {
        throw std::invalid_argument("matmul dimension mismatch");
    }

    const Index m = rows();
    const Index k = cols();
    const Index n = rhs->cols();
    std::vector<float> out(static_cast<std::size_t>(m * n), 0.0f);

    for (Index i = 0; i < m; ++i)
    {
        for (Index j = 0; j < n; ++j)
        {
            float acc = 0.0f;
            for (Index kk = 0; kk < k; ++kk)
            {
                acc += m_data[static_cast<std::size_t>(offset_2d(i, kk))] *
                       rhs->m_data[static_cast<std::size_t>(rhs->offset_2d(kk, j))];
            }
            out[static_cast<std::size_t>(i * n + j)] = acc;
        }
    }
    log_call("matmul:" + shape_to_string() + "x" + rhs->shape_to_string());
    return std::make_unique<MockTensorBackend>(std::vector<Index>{m, n}, std::move(out));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::transpose() const
{
    ensure_shape(2);
    std::vector<float> out(static_cast<std::size_t>(rows() * cols()));
    for (Index r = 0; r < rows(); ++r)
    {
        for (Index c = 0; c < cols(); ++c)
        {
            out[static_cast<std::size_t>(c * rows() + r)] =
                m_data[static_cast<std::size_t>(offset_2d(r, c))];
        }
    }
    log_call("transpose:" + shape_to_string());
    return std::make_unique<MockTensorBackend>(std::vector<Index>{cols(), rows()}, std::move(out));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::relu() const
{
    std::vector<float> out(m_data.size());
    for (std::size_t i = 0; i < m_data.size(); ++i)
    {
        out[i] = m_data[i] < 0.0f ? 0.0f : m_data[i];
    }
    log_call("relu:" + shape_to_string());
    return std::make_unique<MockTensorBackend>(m_shape, std::move(out));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::leaky_relu(float alpha) const
{
    std::vector<float> out(m_data.size());
    for (std::size_t i = 0; i < m_data.size(); ++i)
    {
        out[i] = m_data[i] < 0.0f ? m_data[i] * alpha : m_data[i];
    }
    log_call("leaky_relu:" + shape_to_string());
    return std::make_unique<MockTensorBackend>(m_shape, std::move(out));
}

float MockTensorBackend::mean_squared_error(const ITensorBackend& target) const
{
    ensure_same_shape(target, "mean_squared_error");
    const auto* rhs = dynamic_cast<const MockTensorBackend*>(&target);
    if (!rhs) throw std::invalid_argument("mean_squared_error requires MockTensorBackend");
    float acc = 0.0f;
    for (std::size_t i = 0; i < m_data.size(); ++i)
    {
        const float diff = m_data[i] - rhs->m_data[i];
        acc += diff * diff;
    }
    log_call("mse:" + shape_to_string());
    return m_data.empty() ? 0.0f : acc / static_cast<float>(m_data.size());
}

float MockTensorBackend::norm() const
{
    float acc = 0.0f;
    for (float v : m_data) acc += v * v;
    log_call("norm:" + shape_to_string());
    return std::sqrt(acc);
}

float MockTensorBackend::sum() const
{
    float acc = 0.0f;
    for (float v : m_data) acc += v;
    log_call("sum:" + shape_to_string());
    return acc;
}

std::unique_ptr<ITensorBackend> MockTensorBackend::sum_rows() const
{
    ensure_shape(2);
    const Index nrows = rows();
    const Index ncols = cols();
    std::vector<float> result(static_cast<std::size_t>(nrows), 0.0f);
    for (Index i = 0; i < nrows; ++i)
    {
        for (Index j = 0; j < ncols; ++j)
        {
            result[static_cast<std::size_t>(i)] +=
                m_data[static_cast<std::size_t>(offset_2d(i, j))];
        }
    }
    log_call("sum_rows:" + shape_to_string());
    return std::make_unique<MockTensorBackend>(std::vector<Index>{nrows, 1}, std::move(result));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::sum_cols() const
{
    ensure_shape(2);
    const Index nrows = rows();
    const Index ncols = cols();
    std::vector<float> result(static_cast<std::size_t>(ncols), 0.0f);
    for (Index i = 0; i < nrows; ++i)
    {
        for (Index j = 0; j < ncols; ++j)
        {
            result[static_cast<std::size_t>(j)] +=
                m_data[static_cast<std::size_t>(offset_2d(i, j))];
        }
    }
    log_call("sum_cols:" + shape_to_string());
    return std::make_unique<MockTensorBackend>(std::vector<Index>{1, ncols}, std::move(result));
}

void MockTensorBackend::zero_grad()
{
    if (!m_grad)
    {
        m_grad = std::make_unique<MockTensorBackend>(m_shape);
    }
    else
    {
        m_grad->construct(m_shape);
    }
    log_call("zero_grad:" + shape_to_string());
}

void MockTensorBackend::set_grad(const ITensorBackend& grad)
{
    const auto* rhs = dynamic_cast<const MockTensorBackend*>(&grad);
    if (!rhs) throw std::invalid_argument("set_grad requires MockTensorBackend");
    ensure_same_shape(grad, "set_grad");
    if (!m_grad)
    {
        m_grad = std::make_unique<MockTensorBackend>(m_shape, rhs->m_data);
    }
    else
    {
        m_grad->copy_from(grad);
    }
    log_call("set_grad:" + shape_to_string());
}

const ITensorBackend& MockTensorBackend::grad() const
{
    if (!m_grad)
    {
        const_cast<MockTensorBackend*>(this)->zero_grad();
    }
    log_call("grad-get:" + shape_to_string());
    return *m_grad;
}

ITensorBackend& MockTensorBackend::grad()
{
    if (!m_grad)
    {
        zero_grad();
    }
    log_call("grad-get:" + shape_to_string());
    return *m_grad;
}

const float* MockTensorBackend::data_ptr() const
{
    return m_data.data();
}

float* MockTensorBackend::mutable_data_ptr()
{
    return m_data.data();
}

Index MockTensorBackend::data_rows() const
{
    return rows();
}

Index MockTensorBackend::data_cols() const
{
    return cols();
}

// Element-wise math operations
std::unique_ptr<ITensorBackend> MockTensorBackend::sqrt() const
{
    std::vector<float> out(m_data.size());
    for (std::size_t i = 0; i < m_data.size(); ++i)
    {
        out[i] = std::sqrt(m_data[i]);
    }
    log_call("sqrt");
    return std::make_unique<MockTensorBackend>(m_shape, std::move(out));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::square() const
{
    std::vector<float> out(m_data.size());
    for (std::size_t i = 0; i < m_data.size(); ++i)
    {
        out[i] = m_data[i] * m_data[i];
    }
    log_call("square");
    return std::make_unique<MockTensorBackend>(m_shape, std::move(out));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::abs() const
{
    std::vector<float> out(m_data.size());
    for (std::size_t i = 0; i < m_data.size(); ++i)
    {
        out[i] = std::abs(m_data[i]);
    }
    log_call("abs");
    return std::make_unique<MockTensorBackend>(m_shape, std::move(out));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::divide(const ITensorBackend& other) const
{
    ensure_same_shape(other, "divide");
    const auto* rhs = dynamic_cast<const MockTensorBackend*>(&other);
    if (!rhs) throw std::invalid_argument("divide requires MockTensorBackend");
    std::vector<float> out(m_data.size());
    for (std::size_t i = 0; i < m_data.size(); ++i)
    {
        if (std::abs(rhs->m_data[i]) < kEpsilon)
        {
            throw std::invalid_argument("division by zero");
        }
        out[i] = m_data[i] / rhs->m_data[i];
    }
    log_call("divide");
    return std::make_unique<MockTensorBackend>(m_shape, std::move(out));
}

std::unique_ptr<ITensorBackend> MockTensorBackend::divide_scalar(float scalar) const
{
    if (std::abs(scalar) < kEpsilon)
    {
        throw std::invalid_argument("division by zero");
    }
    std::vector<float> out(m_data.size());
    for (std::size_t i = 0; i < m_data.size(); ++i)
    {
        out[i] = m_data[i] / scalar;
    }
    log_call("divide_scalar");
    return std::make_unique<MockTensorBackend>(m_shape, std::move(out));
}

// Initialization
void MockTensorBackend::fill(float value)
{
    std::fill(m_data.begin(), m_data.end(), value);
    log_call("fill:" + std::to_string(value));
}

void MockTensorBackend::set_zero()
{
    std::fill(m_data.begin(), m_data.end(), 0.0f);
    log_call("set_zero");
}

void MockTensorBackend::set_ones()
{
    std::fill(m_data.begin(), m_data.end(), 1.0f);
    log_call("set_ones");
}

std::unique_ptr<ITensorBackend> MockTensorBackend::clone() const
{
    auto cloned = std::make_unique<MockTensorBackend>(m_shape, m_data);
    if (m_grad)
    {
        cloned->m_grad = std::unique_ptr<MockTensorBackend>(
            static_cast<MockTensorBackend*>(m_grad->clone().release()));
    }
    cloned->m_calls = m_calls;
    log_call("clone:" + shape_to_string());
    return cloned;
}

void MockTensorBackend::copy_from(const ITensorBackend& other)
{
    const auto* rhs = dynamic_cast<const MockTensorBackend*>(&other);
    if (!rhs) throw std::invalid_argument("copy_from requires MockTensorBackend");
    m_shape = rhs->m_shape;
    m_data = rhs->m_data;
    if (rhs->m_grad)
    {
        m_grad = std::unique_ptr<MockTensorBackend>(
            static_cast<MockTensorBackend*>(rhs->m_grad->clone().release()));
    }
    else
    {
        m_grad.reset();
    }
    log_call("copy_from:" + shape_to_string());
}

std::unique_ptr<ITensorBackend> MockTensorBackend::slice(std::span<const int> indices) const
{
    ensure_shape(2);
    const Index new_rows = static_cast<Index>(indices.size());
    const Index num_cols = cols();
    std::vector<float> out(static_cast<std::size_t>(new_rows * num_cols));
    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        const int src = indices[i];
        if (src < 0 || src >= static_cast<int>(rows()))
        {
            throw std::out_of_range("slice index out of range");
        }
        for (Index c = 0; c < num_cols; ++c)
        {
            out[i * static_cast<std::size_t>(num_cols) + c] =
                m_data[static_cast<std::size_t>(offset_2d(static_cast<Index>(src), c))];
        }
    }
    log_call("slice:" + std::to_string(new_rows));
    return std::make_unique<MockTensorBackend>(std::vector<Index>{new_rows, num_cols},
                                               std::move(out));
}

Index MockTensorBackend::offset_2d(Index row, Index col) const
{
    if (row >= rows() || col >= cols())
    {
        throw std::out_of_range("2D index out of range");
    }
    return row * cols() + col;
}

Index MockTensorBackend::offset_nd(const std::vector<Index>& indices) const
{
    if (indices.size() != m_shape.size())
    {
        throw std::invalid_argument("index dimensionality mismatch");
    }
    Index flat = 0;
    Index stride = 1;
    for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i)
    {
        if (indices[static_cast<std::size_t>(i)] >= m_shape[static_cast<std::size_t>(i)])
        {
            throw std::out_of_range("nd index out of range");
        }
        flat += indices[static_cast<std::size_t>(i)] * stride;
        stride *= m_shape[static_cast<std::size_t>(i)];
    }
    return flat;
}

std::string MockTensorBackend::shape_to_string() const
{
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < m_shape.size(); ++i)
    {
        oss << m_shape[i];
        if (i + 1 < m_shape.size()) oss << "x";
    }
    oss << "]";
    return oss.str();
}

void MockTensorBackend::ensure_shape(std::size_t dims) const
{
    if (m_shape.size() != dims)
    {
        throw std::invalid_argument("Unexpected tensor rank");
    }
}

void MockTensorBackend::ensure_same_shape(const ITensorBackend& other, const char* op) const
{
    if (m_shape != other.shape())
    {
        throw std::invalid_argument(std::string(op) + ": shape mismatch");
    }
}

void MockTensorBackend::log_call(const std::string& name) const
{
    m_calls.push_back(name);
}

bool MockTensorBackend::hasNaN() const
{
    for (float val : m_data)
    {
        if (std::isnan(val))
        {
            return true;
        }
    }
    return false;
}

bool MockTensorBackend::equals(const ITensorBackend& other) const
{
    if (m_shape != other.shape())
    {
        return false;
    }
    const auto* mock_other = dynamic_cast<const MockTensorBackend*>(&other);
    if (!mock_other)
    {
        return false;
    }
    constexpr float epsilon = 1e-6f;
    for (size_t i = 0; i < m_data.size(); ++i)
    {
        if (std::abs(m_data[i] - mock_other->m_data[i]) > epsilon)
        {
            return false;
        }
    }
    return true;
}

} // namespace nn
