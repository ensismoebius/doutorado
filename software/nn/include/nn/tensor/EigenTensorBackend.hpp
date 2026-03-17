#ifndef EIGEN_TENSOR_BACKEND_HPP
#define EIGEN_TENSOR_BACKEND_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace nn
{

using Index = std::size_t;

/**
 * EigenTensorBackend — host-backed `Eigen::MatrixXf` storage.
 * Maps logical N-D shapes onto contiguous Eigen storage and preserves
 * lazy gradient semantics (deep-copy on copy, lazy alloc on grad_ref).
 */
class EigenTensorBackend
{
   public:
    // -----------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------
    // Lightweight construction; avoid heavy allocations (do not allocate GPU memory here).
    // Use explicit allocation helpers (e.g., `allocate_on_device()`) for device backends.
    EigenTensorBackend() = default;

    explicit EigenTensorBackend(Index rows, Index cols)
        : m_data(Eigen::MatrixXf::Zero(
              static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols))),
          m_shape({rows, cols})
    {
    }

    explicit EigenTensorBackend(Index d1, Index d2, Index d3, Index d4)
        : m_data(Eigen::MatrixXf::Zero(
              static_cast<Eigen::Index>(d1), static_cast<Eigen::Index>(d2 * d3 * d4))),
          m_shape({d1, d2, d3, d4})
    {
    }

    explicit EigenTensorBackend(const std::vector<Index>& shape) : m_shape(shape)
    {
        if (shape.size() == 2)
        {
            m_data.resize(static_cast<Eigen::Index>(shape[0]), static_cast<Eigen::Index>(shape[1]));
        }
        else if (shape.size() == 4)
        {
            m_data.resize(static_cast<Eigen::Index>(shape[0]),
                static_cast<Eigen::Index>(shape[1] * shape[2] * shape[3]));
        }
        else
        {
            // Flat fallback
            Eigen::Index total = 1;
            for (auto s : shape) total *= static_cast<Eigen::Index>(s);
            m_data.resize(total, 1);
        }
        m_data.setZero();
    }

    explicit EigenTensorBackend(const Eigen::MatrixXf& data)
        : m_data(data), m_shape({static_cast<Index>(data.rows()), static_cast<Index>(data.cols())})
    {
    }

    explicit EigenTensorBackend(Eigen::MatrixXf&& data)
        : m_data(std::move(data)),
          m_shape({static_cast<Index>(m_data.rows()), static_cast<Index>(m_data.cols())})
    {
    }

    // Copy ctor: deep-copy data and gradient backend to preserve autograd state.
    EigenTensorBackend(const EigenTensorBackend& other)
        : m_data(other.m_data), m_shape(other.m_shape)
    {
        if (other.m_grad_backend)
        {
            m_grad_backend = std::make_unique<EigenTensorBackend>(*other.m_grad_backend);
        }
    }

    // Move ctor: efficiently transfer ownership of storage and grad; leaves source
    // valid-but-unspecified.
    EigenTensorBackend(EigenTensorBackend&& other) noexcept = default;

    // Copy Assignment
    EigenTensorBackend& operator=(const EigenTensorBackend& other)
    {
        if (this != &other)
        {
            m_data = other.m_data;
            m_shape = other.m_shape;
            if (other.m_grad_backend)
                m_grad_backend = std::make_unique<EigenTensorBackend>(*other.m_grad_backend);
            else
                m_grad_backend.reset();
        }
        return *this;
    }

    // Move Assignment
    EigenTensorBackend& operator=(EigenTensorBackend&& other) noexcept = default;

    // -----------------------------------------------------------------
    // Static Factories
    // -----------------------------------------------------------------
    // Static factories: convenience constructors that initialize storage (zeros/ones).
    static EigenTensorBackend zeros(Index rows, Index cols)
    {
        EigenTensorBackend t(rows, cols);
        t.m_data.setZero();
        return t;
    }
    static EigenTensorBackend ones(Index rows, Index cols)
    {
        EigenTensorBackend t(rows, cols);
        t.m_data.setOnes();
        return t;
    }
    // Random uniform [0,1) initializer. Uses std::random_device to seed.
    static EigenTensorBackend random(Index rows, Index cols)
    {
        EigenTensorBackend t(rows, cols);
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        t.m_data = Eigen::MatrixXf::NullaryExpr(static_cast<Eigen::Index>(rows),
            static_cast<Eigen::Index>(cols),
            [&]() { return dist(rng); });
        return t;
    }
    // Random uniform [0,1) initializer using an external RNG for reproducibility.
    static EigenTensorBackend random(Index rows, Index cols, std::mt19937& rng)
    {
        EigenTensorBackend t(rows, cols);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        t.m_data = Eigen::MatrixXf::NullaryExpr(static_cast<Eigen::Index>(rows),
            static_cast<Eigen::Index>(cols),
            [&]() { return dist(rng); });
        return t;
    }

    // -----------------------------------------------------------------
    // Shape
    // -----------------------------------------------------------------
    // shape(): returns logical dims (e.g., {d1,d2,...}); stable until reshape().
    // Use rows()/cols() for common 2D access.
    const std::vector<Index>& shape() const
    {
        return m_shape;
    }

    // reshape: total elements must match; preserve linear (row-major) storage order.
    // Throws std::invalid_argument if total size differs.
    void reshape(const std::vector<Index>& new_shape)
    {
        // Reshape is conservative:
        // - Total element count must match.
        // - If the underlying matrix dimensions change, we allocate new storage and
        //   copy elements in the backend's linear order (Eigen's `.data()` order).
        Eigen::Index current_size = m_data.size();
        Eigen::Index new_size = 1;
        for (auto s : new_shape) new_size *= static_cast<Eigen::Index>(s);

        if (current_size != new_size) throw std::invalid_argument("Reshape total size mismatch");

        // Determine new dimensions for Eigen storage
        Eigen::Index new_rows, new_cols;
        if (new_shape.size() == 2)
        {
            new_rows = static_cast<Eigen::Index>(new_shape[0]);
            new_cols = static_cast<Eigen::Index>(new_shape[1]);
        }
        else if (new_shape.size() == 4)
        {
            new_rows = static_cast<Eigen::Index>(new_shape[0]);
            new_cols = static_cast<Eigen::Index>(new_shape[1] * new_shape[2] * new_shape[3]);
        }
        else
        {
            // Flat fallback
            new_rows = new_size;
            new_cols = 1;
        }

        if (new_rows != m_data.rows() || new_cols != m_data.cols())
        {
            Eigen::MatrixXf new_data(new_rows, new_cols);
            // Linear copy to preserve storage order
            if (current_size > 0)
            {
                std::copy(m_data.data(), m_data.data() + current_size, new_data.data());
            }
            m_data = std::move(new_data);
        }
        m_shape = new_shape;
    }

    // Return logical dimension 0 (d1). For 4D tensors this is d1.
    Index rows() const
    {
        return m_shape.empty() ? 0 : m_shape[0];
    }
    // Return logical dimension 1 (d2). For tensors with fewer than 2 dims returns 1.
    Index cols() const
    {
        return m_shape.size() < 2 ? 1 : m_shape[1];
    }
    // Total number of stored elements (contiguous storage size).
    Index size() const
    {
        return static_cast<Index>(m_data.size());
    }

    // -----------------------------------------------------------------
    // Access
    // -----------------------------------------------------------------

    // 1D access (contiguous, row-major). Throws std::out_of_range on bad index.
    // For device backends, document whether this requires a host copy.
    float& at(Index i)
    {
        if (i >= size()) throw std::out_of_range("Index out of range");
        return m_data(static_cast<Eigen::Index>(i));
    }
    const float& at(Index i) const
    {
        if (i >= size()) throw std::out_of_range("Index out of range");
        return m_data(static_cast<Eigen::Index>(i));
    }

    // 2D access (row,col). Valid for 2D shapes; throws invalid_argument/out_of_range on error.
    // Indices are logical coordinates matching shape().
    float& at(Index row, Index col)
    {
        if (m_shape.size() != 2)
            throw std::invalid_argument("at(row, col) is only valid for 2D tensors");
        if (row >= m_shape[0] || col >= m_shape[1]) throw std::out_of_range("Index out of range");
        return m_data(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
    }
    const float& at(Index row, Index col) const
    {
        if (m_shape.size() != 2)
            throw std::invalid_argument("at(row, col) is only valid for 2D tensors");
        if (row >= m_shape[0] || col >= m_shape[1]) throw std::out_of_range("Index out of range");
        return m_data(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
    }

    // 4D access: maps d2,d3,d4 → column index: col_idx = d2*(d3*d4) + d3*(d4) + d4.
    // This mapping is part of the public contract; changing it requires documenting compatibility.
    float& at(Index d1, Index d2, Index d3, Index d4)
    {
        if (m_shape.size() != 4)
            throw std::invalid_argument("at(d1, d2, d3, d4) is only valid for 4D tensors");
        if (d1 >= m_shape[0] || d2 >= m_shape[1] || d3 >= m_shape[2] || d4 >= m_shape[3])
            throw std::out_of_range("Index out of range");

        Index height = m_shape[2];
        Index width = m_shape[3];
        Index col_idx = (d2 * (height * width)) + (d3 * width) + d4;
        return m_data(static_cast<Eigen::Index>(d1), static_cast<Eigen::Index>(col_idx));
    }
    const float& at(Index d1, Index d2, Index d3, Index d4) const
    {
        if (m_shape.size() != 4)
            throw std::invalid_argument("at(d1, d2, d3, d4) is only valid for 4D tensors");
        if (d1 >= m_shape[0] || d2 >= m_shape[1] || d3 >= m_shape[2] || d4 >= m_shape[3])
            throw std::out_of_range("Index out of range");

        Index height = m_shape[2];
        Index width = m_shape[3];
        Index col_idx = (d2 * (height * width)) + (d3 * width) + d4;
        return m_data(static_cast<Eigen::Index>(d1), static_cast<Eigen::Index>(col_idx));
    }

    // N-D access: delegate to 1/2/4 specializations; otherwise linearize indices.
    float& at(const std::vector<Index>& indices)
    {
        if (indices.size() != m_shape.size())
            throw std::invalid_argument("Indices dimension mismatch");

        if (indices.size() == 2) return at(indices[0], indices[1]);
        if (indices.size() == 4) return at(indices[0], indices[1], indices[2], indices[3]);
        if (indices.size() == 1) return at(indices[0]);

        // General linear access for other dimensions (e.g. 3D flattened fallback)
        Eigen::Index flat_idx = 0;
        Eigen::Index current_stride = 1;
        for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i)
        {
            if (indices[i] >= m_shape[i]) throw std::out_of_range("Index out of range");
            flat_idx += static_cast<Eigen::Index>(indices[i]) * current_stride;
            current_stride *= static_cast<Eigen::Index>(m_shape[i]);
        }
        return m_data(flat_idx);
    }
    const float& at(const std::vector<Index>& indices) const
    {
        if (indices.size() != m_shape.size())
            throw std::invalid_argument("Indices dimension mismatch");

        if (indices.size() == 2) return at(indices[0], indices[1]);
        if (indices.size() == 4) return at(indices[0], indices[1], indices[2], indices[3]);
        if (indices.size() == 1) return at(indices[0]);

        Eigen::Index flat_idx = 0;
        Eigen::Index current_stride = 1;
        for (int i = static_cast<int>(m_shape.size()) - 1; i >= 0; --i)
        {
            if (indices[i] >= m_shape[i]) throw std::out_of_range("Index out of range");
            flat_idx += static_cast<Eigen::Index>(indices[i]) * current_stride;
            current_stride *= static_cast<Eigen::Index>(m_shape[i]);
        }
        return m_data(flat_idx);
    }

    // Arithmetic ops: validate shape and throw std::invalid_argument on mismatch.

    // In-place mutating operations
    void add_inplace(const EigenTensorBackend& other)
    {
        if (m_shape != other.m_shape) throw std::invalid_argument("Shape mismatch for add_inplace");
        m_data += other.m_data;
    }
    void subtract_inplace(const EigenTensorBackend& other)
    {
        if (m_shape != other.m_shape)
            throw std::invalid_argument("Shape mismatch for subtract_inplace");
        m_data -= other.m_data;
    }
    void multiply_inplace(const EigenTensorBackend& other)
    {
        if (m_shape != other.m_shape)
            throw std::invalid_argument("Shape mismatch for multiply_inplace");
        m_data.array() *= other.m_data.array();
    }
    void divide_inplace(const EigenTensorBackend& other)
    {
        if (m_shape != other.m_shape)
            throw std::invalid_argument("Shape mismatch for divide_inplace");
        m_data.array() /= other.m_data.array();
    }
    void add_scalar_inplace(float val)
    {
        m_data.array() += val;
    }
    void multiply_scalar_inplace(float val)
    {
        m_data *= val;
    }
    void divide_scalar_inplace(float val)
    {
        m_data /= val;
    }
    void sqrt_inplace()
    {
        m_data = m_data.array().sqrt();
    }
    void square_inplace()
    {
        m_data = m_data.array().square();
    }

    EigenTensorBackend exp() const
    {
        return EigenTensorBackend(m_data.array().exp());
    }
    EigenTensorBackend rowwise_sum() const
    {
        EigenTensorBackend res(m_data.rows(), 1);
        res.m_data = m_data.rowwise().sum();
        return res;
    }
    EigenTensorBackend matmul_transposed(const EigenTensorBackend& other) const
    {
        if (m_shape.size() != 2 || other.m_shape.size() != 2)
            throw std::invalid_argument("matmul valid only for 2D tensors");
        if (cols() != other.cols())
            throw std::invalid_argument("Dimension mismatch for matmul_transposed");

        return EigenTensorBackend(m_data * other.m_data.transpose());
    }

    EigenTensorBackend add(const EigenTensorBackend& other) const
    {
        if (m_shape != other.m_shape) throw std::invalid_argument("Shape mismatch for add");
        return EigenTensorBackend(m_data + other.m_data);
    }

    EigenTensorBackend subtract(const EigenTensorBackend& other) const
    {
        if (m_shape != other.m_shape) throw std::invalid_argument("Shape mismatch for subtract");
        return EigenTensorBackend(m_data - other.m_data);
    }

    EigenTensorBackend multiply(const EigenTensorBackend& other) const
    {
        if (m_shape != other.m_shape) throw std::invalid_argument("Shape mismatch for multiply");
        return EigenTensorBackend(m_data.cwiseProduct(other.m_data));
    }

    // Matrix multiplication for 2D tensors only. Backends should delegate to
    // optimized BLAS/GEMM when available to maximize performance.
    EigenTensorBackend matmul(const EigenTensorBackend& other) const
    {
        if (m_shape.size() != 2 || other.m_shape.size() != 2)
            throw std::invalid_argument("matmul valid only for 2D tensors");

        if (cols() != other.rows()) throw std::invalid_argument("Dimension mismatch for matmul");

        return EigenTensorBackend(m_data * other.m_data);
    }

    // Transpose returns a new tensor with swapped dims for 2D only.
    EigenTensorBackend transpose() const
    {
        if (m_shape.size() != 2) throw std::invalid_argument("transpose valid only for 2D tensors");
        return EigenTensorBackend(m_data.transpose());
    }

    EigenTensorBackend add_scalar(float val) const
    {
        return EigenTensorBackend(m_data.array() + val);
    }
    EigenTensorBackend multiply_scalar(float val) const
    {
        return EigenTensorBackend(m_data * val);
    }
    EigenTensorBackend divide_scalar(float val) const
    {
        return EigenTensorBackend(m_data / val);
    }

    EigenTensorBackend divide(const EigenTensorBackend& other) const
    {
        return EigenTensorBackend(m_data.array() / other.m_data.array());
    }

    // Elementwise comparisons: return tensor of 0.0/1.0 floats where condition holds.
    // Supports simple broadcasting along the first dimension when one operand has rows()==1.
    EigenTensorBackend compare_lt(const EigenTensorBackend& other) const
    {
        Index rows_a = rows();
        Index cols_a = cols();
        Index rows_b = other.rows();
        Index cols_b = other.cols();

        if (cols_a != cols_b) throw std::invalid_argument("Shape mismatch for compare_lt");

        Index out_rows = std::max(rows_a, rows_b);

        // Use Eigen array expressions + replicate when needed to leverage SIMD
        Eigen::ArrayXXf a = m_data.array();
        Eigen::ArrayXXf b = other.m_data.array();

        if (rows_a == rows_b)
        {
            return EigenTensorBackend((a < b).cast<float>().matrix());
        }
        else if (rows_a == 1)
        {
            return EigenTensorBackend(
                (a.replicate(static_cast<int>(out_rows), 1) < b).cast<float>().matrix());
        }
        else // rows_b == 1
        {
            return EigenTensorBackend(
                (a < b.replicate(static_cast<int>(out_rows), 1)).cast<float>().matrix());
        }
    }

    EigenTensorBackend compare_gt(const EigenTensorBackend& other) const
    {
        // Vectorized: flip operands and reuse compare_lt semantics
        return other.compare_lt(*this);
    }

    EigenTensorBackend compare_le(const EigenTensorBackend& other) const
    {
        Index rows_a = rows();
        Index cols_a = cols();
        Index rows_b = other.rows();
        Index cols_b = other.cols();

        if (cols_a != cols_b) throw std::invalid_argument("Shape mismatch for compare_le");

        Index out_rows = std::max(rows_a, rows_b);
        Eigen::ArrayXXf a = m_data.array();
        Eigen::ArrayXXf b = other.m_data.array();

        if (rows_a == rows_b)
        {
            return EigenTensorBackend((a <= b).cast<float>().matrix());
        }
        else if (rows_a == 1)
        {
            return EigenTensorBackend(
                (a.replicate(static_cast<int>(out_rows), 1) <= b).cast<float>().matrix());
        }
        else
        {
            return EigenTensorBackend(
                (a <= b.replicate(static_cast<int>(out_rows), 1)).cast<float>().matrix());
        }
    }

    EigenTensorBackend compare_ge(const EigenTensorBackend& other) const
    {
        // a >= b  <=>  b <= a
        return other.compare_le(*this);
    }

    EigenTensorBackend compare_eq(const EigenTensorBackend& other) const
    {
        Index rows_a = rows();
        Index cols_a = cols();
        Index rows_b = other.rows();
        Index cols_b = other.cols();

        if (cols_a != cols_b) throw std::invalid_argument("Shape mismatch for compare_eq");

        Index out_rows = std::max(rows_a, rows_b);
        Eigen::ArrayXXf a = m_data.array();
        Eigen::ArrayXXf b = other.m_data.array();

        if (rows_a == rows_b)
        {
            return EigenTensorBackend((a == b).cast<float>().matrix());
        }
        else if (rows_a == 1)
        {
            return EigenTensorBackend(
                (a.replicate(static_cast<int>(out_rows), 1) == b).cast<float>().matrix());
        }
        else
        {
            return EigenTensorBackend(
                (a == b.replicate(static_cast<int>(out_rows), 1)).cast<float>().matrix());
        }
    }

    // Scalar comparisons: compare every element to a scalar value.
    EigenTensorBackend compare_lt_scalar(float value) const
    {
        Eigen::MatrixXf out = (m_data.array() < value).cast<float>().matrix();
        return EigenTensorBackend(std::move(out));
    }
    EigenTensorBackend compare_gt_scalar(float value) const
    {
        Eigen::MatrixXf out = (m_data.array() > value).cast<float>().matrix();
        return EigenTensorBackend(std::move(out));
    }
    EigenTensorBackend compare_le_scalar(float value) const
    {
        Eigen::MatrixXf out = (m_data.array() <= value).cast<float>().matrix();
        return EigenTensorBackend(std::move(out));
    }
    EigenTensorBackend compare_ge_scalar(float value) const
    {
        Eigen::MatrixXf out = (m_data.array() >= value).cast<float>().matrix();
        return EigenTensorBackend(std::move(out));
    }
    EigenTensorBackend compare_eq_scalar(float value) const
    {
        Eigen::MatrixXf out = (m_data.array() == value).cast<float>().matrix();
        return EigenTensorBackend(std::move(out));
    }

    EigenTensorBackend sqrt() const
    {
        return EigenTensorBackend(m_data.array().sqrt());
    }
    EigenTensorBackend square() const
    {
        return EigenTensorBackend(m_data.array().square());
    }
    EigenTensorBackend abs() const
    {
        return EigenTensorBackend(m_data.array().abs());
    }

    EigenTensorBackend relu() const
    {
        return EigenTensorBackend(m_data.cwiseMax(0.0f));
    }

    EigenTensorBackend leaky_relu(float alpha) const
    {
        return EigenTensorBackend((m_data.array() > 0).select(m_data, alpha * m_data));
    }

    // -----------------------------------------------------------------
    // Clamp helpers
    // -----------------------------------------------------------------
    EigenTensorBackend clamp(float min_val, float max_val) const
    {
        EigenTensorBackend out(*this);
        out.m_data = out.m_data.cwiseMax(min_val).cwiseMin(max_val);
        return out;
    }

    void clamp_inplace(float min_val, float max_val)
    {
        m_data = m_data.cwiseMax(min_val).cwiseMin(max_val);
    }

    // -----------------------------------------------------------------
    // Reductions
    // - Reduction APIs should be implemented using numerically-stable kernels
    //   and avoid temporary allocations when possible.
    // -----------------------------------------------------------------
    float mean_squared_error(const EigenTensorBackend& target) const
    {
        if (m_shape != target.m_shape)
            throw std::invalid_argument("Shape mismatch for mean_squared_error");
        return (m_data - target.m_data).squaredNorm() / static_cast<float>(m_data.size());
    }

    // Euclidean norm of the underlying storage.
    float norm() const
    {
        return m_data.norm();
    }
    // Sum of all elements.
    float sum() const
    {
        return m_data.sum();
    }

    // Mean of all elements.
    float mean() const
    {
        if (m_data.size() == 0) return 0.0f;
        return static_cast<float>(m_data.mean());
    }

    // Sum over rows returning a column vector (rowwise sum).
    EigenTensorBackend sum_rows() const
    {
        return EigenTensorBackend(m_data.rowwise().sum());
    }

    // Sum over columns returning a row vector (colwise sum).
    EigenTensorBackend sum_cols() const
    {
        return EigenTensorBackend(m_data.colwise().sum());
    }

    // NaN detection helper.
    bool hasNaN() const
    {
        return m_data.hasNaN();
    }

    // Approximate equality using Eigen's isApprox; exact equality is rarely
    // useful for floating point tensors.
    bool operator==(const EigenTensorBackend& other) const
    {
        return m_data.isApprox(other.m_data);
    }
    bool operator!=(const EigenTensorBackend& other) const
    {
        return !(*this == other);
    }

    // -----------------------------------------------------------------
    // Slicing & block operations
    // - These helpers operate on the logical 2D interpretation of storage.
    // - For 4D tensors, users should reshape before using block/setBlock.
    // -----------------------------------------------------------------
    // Return a copy of row i as a new backend.
    EigenTensorBackend row(Index i) const
    {
        if (i >= rows()) throw std::out_of_range("Index out of range");
        return EigenTensorBackend(m_data.row(static_cast<Eigen::Index>(i)));
    }
    // Return a copy of column j.
    EigenTensorBackend col(Index j) const
    {
        if (j >= cols()) throw std::out_of_range("Index out of range");
        return EigenTensorBackend(m_data.col(static_cast<Eigen::Index>(j)));
    }
    // Return left-most n columns (copy).
    EigenTensorBackend leftCols(Index n) const
    {
        return EigenTensorBackend(m_data.leftCols(static_cast<Eigen::Index>(n)));
    }
    // Return top n rows (copy).
    EigenTensorBackend topRows(Index n) const
    {
        return EigenTensorBackend(m_data.topRows(static_cast<Eigen::Index>(n)));
    }
    // Return a rectangular block; valid only for 2D logical tensors.
    EigenTensorBackend block(Index r, Index c, Index rows, Index cols) const
    {
        if (m_shape.size() != 2) throw std::invalid_argument("block valid only for 2D");
        if (r + rows > m_shape[0] || c + cols > m_shape[1])
            throw std::out_of_range("Block indices out of range");

        return EigenTensorBackend(m_data.block(static_cast<Eigen::Index>(r),
            static_cast<Eigen::Index>(c),
            static_cast<Eigen::Index>(rows),
            static_cast<Eigen::Index>(cols)));
    }
    // Overwrite a block; both tensors must be 2D and size-compatible.
    void setBlock(Index r, Index c, const EigenTensorBackend& other)
    {
        if (m_shape.size() != 2 || other.m_shape.size() != 2)
            throw std::invalid_argument("setBlock valid only for 2D");
        if (r + other.rows() > m_shape[0] || c + other.cols() > m_shape[1])
            throw std::invalid_argument("Block indices out of range");

        m_data.block(static_cast<Eigen::Index>(r),
            static_cast<Eigen::Index>(c),
            static_cast<Eigen::Index>(other.rows()),
            static_cast<Eigen::Index>(other.cols())) = other.m_data;
    }

    // Slice returns a new backend with selected rows in the same column layout.
    EigenTensorBackend slice(std::span<const int> indices) const
    {
        Eigen::MatrixXf result(indices.size(), m_data.cols());
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] < 0 || static_cast<Index>(indices[i]) >= rows())
                throw std::out_of_range("Index out of range");
            result.row(static_cast<Eigen::Index>(i)) =
                m_data.row(static_cast<Eigen::Index>(indices[i]));
        }
        return EigenTensorBackend(std::move(result));
    }

    // -----------------------------------------------------------------
    // Mutators
    // - Mutating operations modify the underlying contiguous storage.
    // - For device-backed backends, ensure coherency between host/device views
    //   when exposing these mutators.
    // -----------------------------------------------------------------
    // Set all elements to value v.
    void fill(float v)
    {
        m_data.setConstant(v);
    }
    // Zero all elements.
    void set_zero()
    {
        m_data.setZero();
    }
    // Set all elements to one.
    void set_ones()
    {
        m_data.setOnes();
    }

    // data_ptr(): pointer to contiguous host memory (row-major).
    // GPU backends should either return a host mirror pointer or nullptr and provide explicit
    // transfer APIs.
    const float* data_ptr() const
    {
        return m_data.data();
    }
    // mutable_data_ptr(): host pointer; concurrent writes must be synchronized.
    float* mutable_data_ptr()
    {
        return m_data.data();
    }

    // Gradient: get_grad returns a value copy; grad_ref returns a mutable ref (allocates lazily);
    // set_grad copies values; zero_grad zeros storage if present.
    EigenTensorBackend get_grad() const
    {
        // Returns a *value* (copy). If no grad is allocated, returns a zeros tensor
        // with the same logical rows/cols. For device-backed tensors, consider
        // whether get_grad() implies a host copy or a device pointer; document
        // that behaviour clearly in your backend.
        if (m_grad_backend) return *m_grad_backend;
        return EigenTensorBackend::zeros(rows(), cols());
    }

    // Copy provided gradient into internal storage, allocating lazily if needed.
    void set_grad(const EigenTensorBackend& other)
    {
        // Copies the provided gradient values into this backend's grad buffer.
        // Precondition: `other` should be shape-compatible with this tensor.
        if (!m_grad_backend) m_grad_backend = std::make_unique<EigenTensorBackend>(rows(), cols());
        m_grad_backend->m_data = other.m_data;
    }

    // Zero out existing gradient storage (no-op if not allocated).
    void zero_grad()
    {
        // If gradient storage exists, overwrite with zeros.
        // If it doesn't exist yet, we keep it unallocated (lazy) until someone needs it.
        if (m_grad_backend) m_grad_backend->m_data.setZero();
    }

    // grad_ref(): returns mutable reference; allocates lazily. Callers must synchronize concurrent
    // access.
    EigenTensorBackend& grad_ref()
    {
        if (!m_grad_backend) m_grad_backend = std::make_unique<EigenTensorBackend>(rows(), cols());
        return *m_grad_backend;
    }

   private:
    // Underlying contiguous storage (row-major Eigen matrix). Implementations
    // of other backends should provide an equivalent contiguous or clearly
    // documented memory model and ensure `data_ptr()` / `mutable_data_ptr()`
    // semantics match the expectations in this class.
    Eigen::MatrixXf m_data;

    // Logical shape of the tensor (e.g. {d1, d2, d3, d4} for 4D). Keep this
    // consistent with indexing helpers (at(...)). Different storage layouts
    // are acceptable, but the logical indexing behaviour must be preserved.
    std::vector<Index> m_shape;

    // Lazy gradient storage. The gradient backend must be the same backend
    // type (or at least be convertible in a clearly documented way). Marked
    // mutable so const methods can return a copy of the gradient without
    // violating const-correctness. Backends with device-memory should
    // document host-copy behaviour and synchronization requirements here.
    mutable std::unique_ptr<EigenTensorBackend> m_grad_backend;
};

} // namespace nn

#endif // EIGEN_TENSOR_BACKEND_HPP
