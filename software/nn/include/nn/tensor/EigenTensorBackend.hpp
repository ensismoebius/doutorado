#ifndef EIGEN_TENSOR_BACKEND_HPP
#define EIGEN_TENSOR_BACKEND_HPP

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace nn
{

using Index = std::size_t;

class EigenTensorBackend
{
   public:
    // -----------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------
    EigenTensorBackend() = default;

    explicit EigenTensorBackend(Index rows, Index cols)
        : m_data(Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(rows),
                                       static_cast<Eigen::Index>(cols))),
          m_shape({rows, cols})
    {
    }

    explicit EigenTensorBackend(Index d1, Index d2, Index d3, Index d4)
        : m_data(Eigen::MatrixXf::Zero(static_cast<Eigen::Index>(d1),
                                       static_cast<Eigen::Index>(d2 * d3 * d4))),
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

    // Copy Constructor (Deep Copy for grad)
    EigenTensorBackend(const EigenTensorBackend& other)
        : m_data(other.m_data), m_shape(other.m_shape)
    {
        if (other.m_grad_backend)
        {
            m_grad_backend = std::make_unique<EigenTensorBackend>(*other.m_grad_backend);
        }
    }

    // Move Constructor
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

    // -----------------------------------------------------------------
    // Shape
    // -----------------------------------------------------------------
    const std::vector<Index>& shape() const
    {
        return m_shape;
    }

    void reshape(const std::vector<Index>& new_shape)
    {
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

    Index rows() const
    {
        return m_shape.empty() ? 0 : m_shape[0];
    }
    Index cols() const
    {
        return m_shape.size() < 2 ? 1 : m_shape[1];
    }
    Index size() const
    {
        return static_cast<Index>(m_data.size());
    }

    // -----------------------------------------------------------------
    // Access
    // -----------------------------------------------------------------

    // 1D
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

    // 2D
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

    // 4D
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

    // -----------------------------------------------------------------
    // arithmetic (Value return)
    // -----------------------------------------------------------------
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

    EigenTensorBackend matmul(const EigenTensorBackend& other) const
    {
        if (m_shape.size() != 2 || other.m_shape.size() != 2)
            throw std::invalid_argument("matmul valid only for 2D tensors");

        if (cols() != other.rows()) throw std::invalid_argument("Dimension mismatch for matmul");

        return EigenTensorBackend(m_data * other.m_data);
    }

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
    // Reductions
    // -----------------------------------------------------------------
    float mean_squared_error(const EigenTensorBackend& target) const
    {
        if (m_shape != target.m_shape)
            throw std::invalid_argument("Shape mismatch for mean_squared_error");
        return (m_data - target.m_data).squaredNorm() / static_cast<float>(m_data.size());
    }

    float norm() const
    {
        return m_data.norm();
    }
    float sum() const
    {
        return m_data.sum();
    }

    EigenTensorBackend sum_rows() const
    {
        return EigenTensorBackend(m_data.rowwise().sum());
    }

    EigenTensorBackend sum_cols() const
    {
        return EigenTensorBackend(m_data.colwise().sum());
    }

    bool hasNaN() const
    {
        return m_data.hasNaN();
    }

    bool operator==(const EigenTensorBackend& other) const
    {
        return m_data.isApprox(other.m_data);
    }
    bool operator!=(const EigenTensorBackend& other) const
    {
        return !(*this == other);
    }

    // -----------------------------------------------------------------
    // Slicing
    // -----------------------------------------------------------------
    EigenTensorBackend row(Index i) const
    {
        if (i >= rows()) throw std::out_of_range("Index out of range");
        return EigenTensorBackend(m_data.row(static_cast<Eigen::Index>(i)));
    }
    EigenTensorBackend col(Index j) const
    {
        if (j >= cols()) throw std::out_of_range("Index out of range");
        return EigenTensorBackend(m_data.col(static_cast<Eigen::Index>(j)));
    }
    EigenTensorBackend leftCols(Index n) const
    {
        return EigenTensorBackend(m_data.leftCols(static_cast<Eigen::Index>(n)));
    }
    EigenTensorBackend topRows(Index n) const
    {
        return EigenTensorBackend(m_data.topRows(static_cast<Eigen::Index>(n)));
    }
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
    // -----------------------------------------------------------------
    void fill(float v)
    {
        m_data.setConstant(v);
    }
    void set_zero()
    {
        m_data.setZero();
    }
    void set_ones()
    {
        m_data.setOnes();
    }

    const float* data_ptr() const
    {
        return m_data.data();
    }
    float* mutable_data_ptr()
    {
        return m_data.data();
    }

    // -----------------------------------------------------------------
    // Gradient
    // -----------------------------------------------------------------
    EigenTensorBackend get_grad() const
    {
        if (m_grad_backend) return *m_grad_backend;
        return EigenTensorBackend::zeros(rows(), cols());
    }

    void set_grad(const EigenTensorBackend& other)
    {
        if (!m_grad_backend) m_grad_backend = std::make_unique<EigenTensorBackend>(rows(), cols());
        m_grad_backend->m_data = other.m_data;
    }

    void zero_grad()
    {
        if (m_grad_backend) m_grad_backend->m_data.setZero();
    }

    EigenTensorBackend& grad_ref()
    {
        if (!m_grad_backend) m_grad_backend = std::make_unique<EigenTensorBackend>(rows(), cols());
        return *m_grad_backend;
    }

   private:
    Eigen::MatrixXf m_data;
    std::vector<Index> m_shape;
    mutable std::unique_ptr<EigenTensorBackend> m_grad_backend;
};

} // namespace nn

#endif // EIGEN_TENSOR_BACKEND_HPP
