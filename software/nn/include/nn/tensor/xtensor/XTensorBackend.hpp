#pragma once
// XTensorBackend — xtensor-backed N-D tensor storage.
// Replaces XTensorBackend. Uses xt::xarray<float> for true N-D support.
// All shapes are logical (not flattened): a (B,T,D) tensor stores B*T*D elements
// and is indexed as m_data(b, t, d) — no manual stride arithmetic.

#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xeval.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xmanipulation.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <cblas.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "nn/logging/Logger.hpp"

namespace nn
{

#include <xsimd/xsimd.hpp>
static_assert(xsimd::batch<float>::size > 1, "SIMD not active");

using Index = std::size_t;

class XTensorBackend
{
   public:
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    XTensorBackend() = default;

    explicit XTensorBackend(Index rows, Index cols)
        : m_data(xt::zeros<float>({rows, cols}))
    {}

    explicit XTensorBackend(Index d1, Index d2, Index d3)
        : m_data(xt::zeros<float>({d1, d2, d3}))
    {}

    explicit XTensorBackend(Index d1, Index d2, Index d3, Index d4)
        : m_data(xt::zeros<float>({d1, d2, d3, d4}))
    {}

    explicit XTensorBackend(const std::vector<Index>& shape)
    {
        xt::dynamic_shape<Index> xshape(shape.begin(), shape.end());
        m_data = xt::zeros<float>(xshape);
    }

    explicit XTensorBackend(xt::xarray<float> data)
        : m_data(std::move(data))
    {}

    XTensorBackend(const XTensorBackend& other)
        : m_data(other.m_data)
    {
        if (other.m_grad)
            m_grad = other.m_grad;
    }

    XTensorBackend(XTensorBackend&& other) noexcept = default;

    XTensorBackend& operator=(const XTensorBackend& other)
    {
        if (this != &other)
        {
            m_data  = other.m_data;
            m_grad = other.m_grad;
        }
        return *this;
    }

    XTensorBackend& operator=(XTensorBackend&& other) noexcept = default;


    // ------------------------------------------------------------------
    // Static factories
    // ------------------------------------------------------------------

    static XTensorBackend zeros(Index rows, Index cols)
    {
        return XTensorBackend(xt::zeros<float>({rows, cols}));
    }

    static XTensorBackend zeros(Index d1, Index d2, Index d3)
    {
        return XTensorBackend(xt::zeros<float>({d1, d2, d3}));
    }

    static XTensorBackend ones(Index rows, Index cols)
    {
        return XTensorBackend(xt::ones<float>({rows, cols}));
    }

    static XTensorBackend ones(Index d1, Index d2, Index d3)
    {
        return XTensorBackend(xt::ones<float>({d1, d2, d3}));
    }

    static XTensorBackend random(Index rows, Index cols)
    {
        std::mt19937 rng(std::random_device{}());
        return random(rows, cols, rng);
    }

    static XTensorBackend random(Index rows, Index cols, std::mt19937& rng)
    {
        XTensorBackend t(rows, cols);
        fill_uniform_random_simd(t.m_data.data(), t.m_data.size(), rng);
        return t;
    }

    static XTensorBackend random(Index d1, Index d2, Index d3)
    {
        std::mt19937 rng(std::random_device{}());
        return random(d1, d2, d3, rng);
    }

    static XTensorBackend random(Index d1, Index d2, Index d3, std::mt19937& rng)
    {
        XTensorBackend t(d1, d2, d3);
        fill_uniform_random_simd(t.m_data.data(), t.m_data.size(), rng);
        return t;
    }

    // ------------------------------------------------------------------
    // Shape
    // ------------------------------------------------------------------

    std::vector<Index> shape() const
    {
        auto s = m_data.shape();
        return std::vector<Index>(s.begin(), s.end());
    }

    void reshape(const std::vector<Index>& new_shape)
    {
        const Index new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                                 Index{1}, std::multiplies<Index>{});
        if (m_data.size() != new_size)
            throw std::invalid_argument("Reshape total size mismatch");

        xt::dynamic_shape<Index> xshape(new_shape.begin(), new_shape.end());
        m_data.reshape(xshape);
    }

    XTensorBackend reshape(const std::vector<Index>& new_shape) const
    {
        const Index new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                                 Index{1}, std::multiplies<Index>{});
        if (m_data.size() != new_size)
            throw std::invalid_argument("Reshape total size mismatch");

        xt::dynamic_shape<Index> xshape(new_shape.begin(), new_shape.end());
        xt::xarray<float> res = m_data;
        res.reshape(xshape);
        return XTensorBackend(std::move(res));
    }

    Index rows() const { return m_data.shape().empty() ? 0 : m_data.shape(0); }
    Index cols() const { return m_data.shape().size() < 2 ? 1 : m_data.shape(1); }
    Index size() const { return m_data.size(); }


    // ------------------------------------------------------------------
    // Element access
    // ------------------------------------------------------------------

    // Unchecked flat access — caller guarantees i < size().
    float& at_unsafe(Index i) noexcept { return *(m_data.data() + i); }
    const float& at_unsafe(Index i) const noexcept { return *(m_data.data() + i); }

    // Unchecked 2D access.
    float& at_unsafe(Index r, Index c) noexcept { return m_data(r, c); }
    const float& at_unsafe(Index r, Index c) const noexcept { return m_data(r, c); }

    // Unchecked vector access.
    float& at_unsafe(const std::vector<Index>& indices)
    {
        Index flat = 0, stride = 1;
        for (int i = static_cast<int>(m_data.shape().size()) - 1; i >= 0; --i)
        {
            flat += indices[i] * stride;
            stride *= m_data.shape(i);
        }
        return *(m_data.data() + flat);
    }
    const float& at_unsafe(const std::vector<Index>& indices) const
    {
        Index flat = 0, stride = 1;
        for (int i = static_cast<int>(m_data.shape().size()) - 1; i >= 0; --i)
        {
            flat += indices[i] * stride;
            stride *= m_data.shape(i);
        }
        return *(m_data.data() + flat);
    }

    float& at(Index i)
    {
        if (i >= size()) throw std::out_of_range("Index out of range");
        return *(m_data.data() + i);
    }
    const float& at(Index i) const
    {
        if (i >= size()) throw std::out_of_range("Index out of range");
        return *(m_data.data() + i);
    }

    float& at(Index row, Index col)
    {
        if (m_data.shape().size() != 2) {
            std::cerr << "at(row, col) failed: shape size is " << m_data.shape().size() << "\n";
            throw std::invalid_argument("Tensor must be 2D");
        }
        if (row >= m_data.shape(0) || col >= m_data.shape(1))
            throw std::out_of_range("Index out of range");
        return m_data(row, col);
    }
    const float& at(Index row, Index col) const
    {
        if (m_data.shape().size() != 2) throw std::invalid_argument("Tensor must be 2D");
        if (row >= m_data.shape(0) || col >= m_data.shape(1))
            throw std::out_of_range("Index out of range");
        return m_data(row, col);
    }

    float& at(Index d1, Index d2, Index d3)
    {
        if (m_data.shape().size() != 3) throw std::invalid_argument("Tensor must be 3D");
        if (d1 >= m_data.shape(0) || d2 >= m_data.shape(1) || d3 >= m_data.shape(2))
            throw std::out_of_range("Index out of range");
        return m_data(d1, d2, d3);
    }
    const float& at(Index d1, Index d2, Index d3) const
    {
        if (m_data.shape().size() != 3) throw std::invalid_argument("Tensor must be 3D");
        if (d1 >= m_data.shape(0) || d2 >= m_data.shape(1) || d3 >= m_data.shape(2))
            throw std::out_of_range("Index out of range");
        return m_data(d1, d2, d3);
    }

    float& at(Index d1, Index d2, Index d3, Index d4)
    {
        if (m_data.shape().size() != 4) throw std::invalid_argument("Tensor must be 4D");
        if (d1 >= m_data.shape(0) || d2 >= m_data.shape(1) || d3 >= m_data.shape(2) || d4 >= m_data.shape(3))
            throw std::out_of_range("Index out of range");
        return m_data(d1, d2, d3, d4);
    }
    const float& at(Index d1, Index d2, Index d3, Index d4) const
    {
        if (m_data.shape().size() != 4) throw std::invalid_argument("Tensor must be 4D");
        if (d1 >= m_data.shape(0) || d2 >= m_data.shape(1) || d3 >= m_data.shape(2) || d4 >= m_data.shape(3))
            throw std::out_of_range("Index out of range");
        return m_data(d1, d2, d3, d4);
    }

    float& at(const std::vector<Index>& indices)
    {
        if (indices.size() != m_data.shape().size()) throw std::invalid_argument("Index dimension mismatch");
        if (indices.size() == 1) return at(indices[0]);
        if (indices.size() == 2) return at(indices[0], indices[1]);
        if (indices.size() == 3) return at(indices[0], indices[1], indices[2]);
        if (indices.size() == 4) return at(indices[0], indices[1], indices[2], indices[3]);
        Index flat = 0, stride = 1;
        for (int i = static_cast<int>(m_data.shape().size()) - 1; i >= 0; --i)
        {
            if (indices[i] >= m_data.shape(i)) throw std::out_of_range("Index out of range");
            flat += indices[i] * stride;
            stride *= m_data.shape(i);
        }
        return *(m_data.data() + flat);
    }
    const float& at(const std::vector<Index>& indices) const
    {
        if (indices.size() != m_data.shape().size()) throw std::invalid_argument("Index dimension mismatch");
        if (indices.size() == 1) return at(indices[0]);
        if (indices.size() == 2) return at(indices[0], indices[1]);
        if (indices.size() == 3) return at(indices[0], indices[1], indices[2]);
        if (indices.size() == 4) return at(indices[0], indices[1], indices[2], indices[3]);
        Index flat = 0, stride = 1;
        for (int i = static_cast<int>(m_data.shape().size()) - 1; i >= 0; --i)
        {
            if (indices[i] >= m_data.shape(i)) throw std::out_of_range("Index out of range");
            flat += indices[i] * stride;
            stride *= m_data.shape(i);
        }
        return *(m_data.data() + flat);
    }

    // ------------------------------------------------------------------
    // In-place arithmetic
    // ------------------------------------------------------------------

    void add_inplace(const XTensorBackend& other)       { m_data += other.m_data; }
    void subtract_inplace(const XTensorBackend& other)  { m_data -= other.m_data; }
    void multiply_inplace(const XTensorBackend& other)  { m_data *= other.m_data; }
    void divide_inplace(const XTensorBackend& other)    { m_data /= other.m_data; }
    void add_scalar_inplace(float val)                  { m_data += val; }
    void multiply_scalar_inplace(float val)             { m_data *= val; }
    void divide_scalar_inplace(float val)               { m_data /= val; }
    void sqrt_inplace()                                 { xt::noalias(m_data) = xt::sqrt(m_data); }
    void square_inplace()                               { xt::noalias(m_data) = xt::square(m_data); }

    // ------------------------------------------------------------------
    // Functional arithmetic
    // ------------------------------------------------------------------

    XTensorBackend exp() const
    {
        xt::xarray<float> r = xt::exp(m_data);
        return XTensorBackend(std::move(r));
    }

    XTensorBackend add(const XTensorBackend& other) const
    {
        if (!same_shape(other)) throw std::invalid_argument("Shape mismatch for add");
        xt::xarray<float> r = m_data + other.m_data;
        return XTensorBackend(std::move(r));
    }

    void add_col_vector_to_rows_inplace(const XTensorBackend& col_vector)
    {
        xt::xarray<float> bias_row = xt::reshape_view(
            xt::view(col_vector.m_data, xt::all(), 0),
            std::vector<Index>{1, col_vector.m_data.shape(0)});
        m_data += bias_row;
    }

    // Broadcast-add a (1,C) row tensor to every row of this (N,C) tensor.
    // Returns a new tensor; does not modify this.
    XTensorBackend add_row_broadcast(const XTensorBackend& row) const
    {
        xt::xarray<float> r = m_data + row.m_data;
        return XTensorBackend(std::move(r));
    }

    // In-place version: this (N,C) += row (1,C), broadcasting over N rows.
    void add_row_broadcast_inplace(const XTensorBackend& row)
    {
        m_data += row.m_data;
    }

    XTensorBackend subtract(const XTensorBackend& other) const
    {
        if (!same_shape(other)) throw std::invalid_argument("Shape mismatch for subtract");
        xt::xarray<float> r = m_data - other.m_data;
        return XTensorBackend(std::move(r));
    }

    XTensorBackend multiply(const XTensorBackend& other) const
    {
        if (!same_shape(other)) throw std::invalid_argument("Shape mismatch for multiply");
        xt::xarray<float> r = m_data * other.m_data;
        return XTensorBackend(std::move(r));
    }

    XTensorBackend matmul_transposed(const XTensorBackend& other) const
    {
        if (shape().size() != 2 || other.shape().size() != 2)
            throw std::invalid_argument("Tensors must be 2D");
        if (cols() != other.cols())
            throw std::invalid_argument("Dimension mismatch for matmul_transposed");

        XTensorBackend result(rows(), other.rows());
        cblas_sgemm(CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            static_cast<int>(rows()),
            static_cast<int>(other.rows()),
            static_cast<int>(cols()),
            1.0f,
            m_data.data(),
            static_cast<int>(cols()),
            other.m_data.data(),
            static_cast<int>(other.cols()),
            0.0f,
            result.m_data.data(),
            static_cast<int>(other.rows()));
        return result;
    }

    XTensorBackend matmul(const XTensorBackend& other) const
    {
        if (shape().size() != 2 || other.shape().size() != 2) {
            std::cerr << "matmul() failed: left shape size " << shape().size() 
                      << ", right shape size " << other.shape().size() << "\n";
            throw std::invalid_argument("Tensors must be 2D");
        }
        if (cols() != other.rows())
            throw std::invalid_argument("Dimension mismatch for matmul");
        xt::xarray<float> r = xt::linalg::dot(m_data, other.m_data);
        return XTensorBackend(std::move(r));
    }

    XTensorBackend transpose() const
    {
        if (m_data.shape().size() != 2) {
            std::cerr << "transpose() failed: shape size is " << m_data.shape().size() << "\n";
            throw std::invalid_argument("Tensor must be 2D");
        }
        xt::xarray<float> r = xt::transpose(m_data);
        return XTensorBackend(std::move(r));
    }

    XTensorBackend add_scalar(float val) const
    {
        xt::xarray<float> r = m_data + val;
        return XTensorBackend(std::move(r));
    }
    XTensorBackend multiply_scalar(float val) const
    {
        xt::xarray<float> r = m_data * val;
        return XTensorBackend(std::move(r));
    }
    XTensorBackend divide_scalar(float val) const
    {
        xt::xarray<float> r = m_data / val;
        return XTensorBackend(std::move(r));
    }

    XTensorBackend divide(const XTensorBackend& other) const
    {
        xt::xarray<float> r = m_data / other.m_data;
        return XTensorBackend(std::move(r));
    }

    // ------------------------------------------------------------------
    // Comparisons (return float 0/1 tensor)
    // ------------------------------------------------------------------

    // Comparisons: xtensor broadcasts automatically (e.g. {1,3} vs {2,3} → {2,3})
    XTensorBackend compare_lt(const XTensorBackend& other) const
    {
        xt::xarray<float> r = xt::cast<float>(m_data < other.m_data);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend compare_gt(const XTensorBackend& other) const
    {
        xt::xarray<float> r = xt::cast<float>(m_data > other.m_data);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend compare_le(const XTensorBackend& other) const
    {
        xt::xarray<float> r = xt::cast<float>(m_data <= other.m_data);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend compare_ge(const XTensorBackend& other) const
    {
        xt::xarray<float> r = xt::cast<float>(m_data >= other.m_data);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend compare_eq(const XTensorBackend& other) const
    {
        xt::xarray<float> r = xt::cast<float>(xt::equal(m_data, other.m_data));
        return XTensorBackend(std::move(r));
    }
    XTensorBackend compare_lt_scalar(float v) const
    {
        xt::xarray<float> r = xt::cast<float>(m_data < v);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend compare_gt_scalar(float v) const
    {
        xt::xarray<float> r = xt::cast<float>(m_data > v);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend compare_le_scalar(float v) const
    {
        xt::xarray<float> r = xt::cast<float>(m_data <= v);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend compare_ge_scalar(float v) const
    {
        xt::xarray<float> r = xt::cast<float>(m_data >= v);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend compare_eq_scalar(float v) const
    {
        xt::xarray<float> r = xt::cast<float>(xt::equal(m_data, v));
        return XTensorBackend(std::move(r));
    }

    // ------------------------------------------------------------------
    // Math
    // ------------------------------------------------------------------

    XTensorBackend sqrt() const
    {
        xt::xarray<float> r = xt::sqrt(m_data);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend square() const
    {
        xt::xarray<float> r = xt::square(m_data);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend abs() const
    {
        xt::xarray<float> r = xt::abs(m_data);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend relu() const
    {
        xt::xarray<float> r = xt::maximum(m_data, 0.0f);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend leaky_relu(float alpha) const
    {
        xt::xarray<float> r = xt::where(m_data > 0.0f, m_data, alpha * m_data);
        return XTensorBackend(std::move(r));
    }
    XTensorBackend clamp(float min_val, float max_val) const
    {
        xt::xarray<float> r = xt::clip(m_data, min_val, max_val);
        return XTensorBackend(std::move(r));
    }
    void clamp_inplace(float min_val, float max_val)
    {
        xt::noalias(m_data) = xt::clip(m_data, min_val, max_val);
    }

    // ------------------------------------------------------------------
    // Reductions
    // ------------------------------------------------------------------

    float mean_squared_error(const XTensorBackend& target) const
    {
        if (m_data.size() != target.m_data.size())
        {
            std::ostringstream oss;
            oss << "Shape mismatch in mean_squared_error: "
                << rows() << "x" << cols() << " vs "
                << target.rows() << "x" << target.cols();
            NN_LOG_ERROR(oss.str());
            throw std::invalid_argument("Shape mismatch for mean_squared_error");
        }
        const float* a = m_data.data();
        const float* b = target.m_data.data();
        const std::size_t n = m_data.size();
        float sq = 0.0f;
        for (std::size_t i = 0; i < n; ++i) { float d = a[i] - b[i]; sq += d * d; }
        return sq / static_cast<float>(n);
    }

    float norm() const
    {
        const float* ptr = m_data.data();
        const std::size_t n = m_data.size();
        float sq = 0.0f;
        for (std::size_t i = 0; i < n; ++i) sq += ptr[i] * ptr[i];
        return std::sqrt(sq);
    }

    float sum() const
    {
        const float* ptr = m_data.data();
        const std::size_t n = m_data.size();
        float s = 0.0f;
        for (std::size_t i = 0; i < n; ++i) s += ptr[i];
        return s;
    }
    float mean() const
    {
        if (m_data.size() == 0) return 0.0f;
        return sum() / static_cast<float>(m_data.size());
    }

    XTensorBackend sum_rows() const
    {
        // Reduce axis 1 → (R,1). Allocate target first to avoid xt::eval rank deduction
        // issues with xarray-backed reducers (xt::eval may infer wrong static rank).
        xt::xarray<float> s = xt::zeros<float>({m_data.shape(0), std::size_t{1}});
        xt::view(s, xt::all(), 0) = xt::sum(m_data, {std::size_t{1}});
        return XTensorBackend(std::move(s));
    }

    XTensorBackend sum_cols() const
    {
        // Reduce axis 0 → (1,C). Same rationale as sum_rows.
        xt::xarray<float> s = xt::zeros<float>({std::size_t{1}, m_data.shape(1)});
        xt::view(s, 0, xt::all()) = xt::sum(m_data, {std::size_t{0}});
        return XTensorBackend(std::move(s));
    }

    XTensorBackend rowwise_sum() const { return sum_rows(); }

    bool hasNaN() const
    {
        const float* ptr = m_data.data();
        for (std::size_t i = 0; i < m_data.size(); ++i)
            if (std::isnan(ptr[i])) return true;
        return false;
    }

    bool operator==(const XTensorBackend& other) const
    {
        if (shape() != other.shape()) return false;
        return static_cast<bool>(xt::all(xt::abs(m_data - other.m_data) <= 1e-5f));
    }
    bool operator!=(const XTensorBackend& other) const { return !(*this == other); }

    // ------------------------------------------------------------------
    // Slicing and views
    // ------------------------------------------------------------------

    XTensorBackend row(Index i) const
    {
        if (i >= rows()) throw std::out_of_range("Index out of range");
        if (m_data.shape().size() == 1)
        {
            std::vector<Index> shape = {1, 1};
            xt::xarray<float> r = xt::zeros<float>(shape);
            r(0, 0) = m_data(i);
            return XTensorBackend(std::move(r));
        }
        xt::xarray<float> r = xt::eval(
            xt::reshape_view(xt::view(m_data, i, xt::all()),
                             std::vector<Index>{std::size_t{1}, m_data.shape(1)}));
        return XTensorBackend(std::move(r));
    }

    XTensorBackend col(Index j) const
    {
        if (m_data.shape().size() == 1)
        {
            if (j != 0) throw std::out_of_range("Index out of range");
            std::vector<Index> shape = {m_data.shape(0), 1};
            xt::xarray<float> c = xt::zeros<float>(shape);
            for (Index i = 0; i < m_data.shape(0); ++i) c(i, 0) = m_data(i);
            return XTensorBackend(std::move(c));
        }
        if (j >= cols()) throw std::out_of_range("Index out of range");
        xt::xarray<float> c = xt::view(m_data, xt::all(), j);
        c.reshape({m_data.shape(0), std::size_t{1}});
        return XTensorBackend(std::move(c));
    }

    XTensorBackend leftCols(Index n) const
    {
        xt::xarray<float> r = xt::view(m_data, xt::all(), xt::range(Index{0}, n));
        return XTensorBackend(std::move(r));
    }

    XTensorBackend topRows(Index n) const
    {
        xt::xarray<float> r = xt::view(m_data, xt::range(Index{0}, n), xt::all());
        return XTensorBackend(std::move(r));
    }

    XTensorBackend block(Index r, Index c, Index block_rows, Index block_cols) const
    {
        if (m_data.shape().size() == 1)
        {
            if (c != 0 || block_cols != 1) throw std::invalid_argument("Block must be (r, 0, rows, 1) for 1D tensor");
            if (r + block_rows > m_data.shape(0)) throw std::out_of_range("Block indices out of range");
            
            std::vector<Index> shape = {block_rows, 1};
            xt::xarray<float> res = xt::zeros<float>(shape);
            for (Index i = 0; i < block_rows; ++i) res(i, 0) = m_data(r + i);
            return XTensorBackend(std::move(res));
        }
        if (r + block_rows > m_data.shape(0) || c + block_cols > m_data.shape(1))
            throw std::out_of_range("Block indices out of range");
        xt::xarray<float> res = xt::view(m_data,
                                           xt::range(r, r + block_rows),
                                           xt::range(c, c + block_cols));
        return XTensorBackend(std::move(res));
    }


    // Extract 2D slice [b, :, :] from a 3D (B, T, D) tensor → (T, D).
    XTensorBackend slice_batch(Index b) const
    {
        if (m_data.shape().size() != 3)
            throw std::invalid_argument("slice_batch: tensor must be 3D");
        if (b >= m_data.shape(0))
            throw std::out_of_range("slice_batch: index out of range");
        xt::xarray<float> r = xt::eval(xt::view(m_data, b, xt::all(), xt::all()));
        return XTensorBackend(std::move(r));
    }

    // Write 2D (T, D) tensor into slice [b, :, :] of a 3D (B, T, D) tensor.
    void set_batch_slice(Index b, const XTensorBackend& val)
    {
        if (m_data.shape().size() != 3)
            throw std::invalid_argument("set_batch_slice: tensor must be 3D");
        if (b >= m_data.shape(0))
            throw std::out_of_range("set_batch_slice: index out of range");
        xt::view(m_data, b, xt::all(), xt::all()) = val.m_data;
    }

    // Extract (B, D) slice at time t from a 3D (B, T, D) tensor.
    XTensorBackend slice_time(Index t) const
    {
        if (m_data.shape().size() != 3)
            throw std::invalid_argument("slice_time: tensor must be 3D");
        if (t >= m_data.shape(1))
            throw std::out_of_range("slice_time: index out of range");
        xt::xarray<float> r = xt::eval(xt::view(m_data, xt::all(), t, xt::all()));
        return XTensorBackend(std::move(r));
    }

    // Write (B, D) tensor into [:, t, :] of a 3D (B, T, D) tensor.
    void set_time_slice(Index t, const XTensorBackend& val)
    {
        if (m_data.shape().size() != 3)
            throw std::invalid_argument("set_time_slice: tensor must be 3D");
        if (t >= m_data.shape(1))
            throw std::out_of_range("set_time_slice: index out of range");
        xt::view(m_data, xt::all(), t, xt::all()) = val.m_data;
    }

    void setBlock(Index r, Index c, const XTensorBackend& other)
    {
        if (r + other.rows() > m_data.shape(0) || c + other.cols() > m_data.shape(1))
            throw std::invalid_argument("Block indices out of range");
        xt::view(m_data,
                 xt::range(r, r + other.rows()),
                 xt::range(c, c + other.cols())) = other.m_data;
    }

    XTensorBackend slice(std::span<const int> indices) const
    {
        const Index n = indices.size();
        xt::xarray<float> result = xt::zeros<float>({n, m_data.shape(1)});
        for (Index i = 0; i < n; ++i)
        {
            if (indices[i] < 0 || static_cast<Index>(indices[i]) >= rows())
                throw std::out_of_range("Index out of range");
            xt::view(result, i, xt::all()) =
                xt::view(m_data, static_cast<Index>(indices[i]), xt::all());
        }
        return XTensorBackend(std::move(result));
    }

    // ------------------------------------------------------------------
    // Mutators
    // ------------------------------------------------------------------

    void fill(float v)  { fill_constant_simd(m_data.data(), m_data.size(), v); }
    void set_zero()     { fill_constant_simd(m_data.data(), m_data.size(), 0.0f); }
    void set_ones()     { fill_constant_simd(m_data.data(), m_data.size(), 1.0f); }

    const float* data_ptr() const noexcept { return m_data.data(); }
    float* mutable_data_ptr() noexcept     { return m_data.data(); }

    // ------------------------------------------------------------------
    // Gradient
    // ------------------------------------------------------------------

    XTensorBackend get_grad() const
    {
        if (m_grad) return XTensorBackend(*m_grad);
        return XTensorBackend(shape());
    }

    void set_grad(const XTensorBackend& other)
    {
        if (!m_grad) m_grad.emplace(xt::zeros<float>(m_data.shape()));
        *m_grad = other.m_data;
    }

    void zero_grad()
    {
        if (m_grad) m_grad->fill(0.0f);
    }

    XTensorBackend& grad_ref()
    {
        if (!m_grad_wrapper)
        {
            std::vector<Index> shape_vec(m_data.shape().begin(), m_data.shape().end());
            m_grad_wrapper = std::make_unique<XTensorBackend>(shape_vec);
        }
        if (m_grad) m_grad_wrapper->m_data = *m_grad;
        return *m_grad_wrapper;
    }

    private:
    bool same_shape(const XTensorBackend& other) const noexcept
    {
        const auto& a = m_data.shape();
        const auto& b = other.m_data.shape();
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i]) return false;
        return true;
    }

    static void fill_uniform_random_simd(float* data, Index count, std::mt19937& rng)
    {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        constexpr Index kWidth = xsimd::batch<float>::size;
        std::array<float, kWidth> lane{};

        Index i = 0;
        for (; i + kWidth <= count; i += kWidth)
        {
            for (Index j = 0; j < kWidth; ++j)
                lane[j] = dist(rng);
            const auto batch = xsimd::batch<float>::load_unaligned(lane.data());
            batch.store_unaligned(data + i);
        }
        for (; i < count; ++i)
            data[i] = dist(rng);
    }

    static void fill_constant_simd(float* data, Index count, float value)
    {
        constexpr Index kWidth = xsimd::batch<float>::size;
        const auto batch = xsimd::batch<float>(value);

        Index i = 0;
        for (; i + kWidth <= count; i += kWidth)
            batch.store_unaligned(data + i);
        for (; i < count; ++i)
            data[i] = value;
    }

    xt::xarray<float>  m_data;
    mutable std::optional<xt::xarray<float>> m_grad;
    mutable std::unique_ptr<XTensorBackend> m_grad_wrapper;


    XTensorBackend make_like(xt::xarray<float> data) const
    {
        return XTensorBackend(std::move(data));
    }
};

} // namespace nn
