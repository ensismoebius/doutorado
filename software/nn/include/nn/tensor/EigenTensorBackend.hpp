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

/**
 * @file EigenTensorBackend.hpp
 * @brief Default tensor backend using Eigen::MatrixXf storage.
 *
 * Storage model:
 * - Data is stored as an Eigen::MatrixXf (`m_data`) with a separate logical `m_shape`.
 * - For 2D tensors, `m_shape == {rows, cols}` and `m_data` has the same shape.
 * - For 4D tensors, `m_shape == {d1, d2, d3, d4}` but `m_data` is stored as:
 *     rows = d1
 *     cols = d2*d3*d4
 *   and `at(d1,d2,d3,d4)` maps the last three dims into a single column index.
 *
 * Important gotcha for callers:
 * - `rows()` and `cols()` report logical dims (d1 and d2 for 4D), not the flattened storage cols.
 *   If you need the true contiguous storage shape, use `size()` + `data_ptr()`.
 *
 * Gradients:
 * - Grad is stored lazily via `m_grad_backend` (unique_ptr). If absent, `get_grad()` returns zeros.
 * - `set_grad()` copies data into the grad buffer (allocating if needed).
 * - Copying a backend deep-copies its grad backend to preserve autograd state.
 *
 * -----------------------------------------------------------------
 * Backend Implementation Guide (for future implementers)
 * -----------------------------------------------------------------
 * This file defines the canonical behaviour and surface area that the rest
 * of the library expects from a tensor backend. If you want to implement an
 * alternative backend (e.g., CUDA, ROCm, MKL, or a memory-mapped backend),
 * keep the following contract in mind:
 *
 * - API surface: The backend must provide the same logical methods with the
 *   same semantics. Important methods include:
 *     - shape()/reshape()
 *     - rows(), cols(), size()
 *     - at(...) overloads for 1D/2D/4D/direct index access
 *     - arithmetic ops (add, subtract, multiply, divide, scalar ops)
 *     - matmul(), transpose()
 *     - reductions (sum(), norm(), mean_squared_error())
 *     - slicing and block operations (row(), col(), block(), setBlock(), slice())
 *     - data accessors: data_ptr(), mutable_data_ptr()
 *     - gradient operations: get_grad(), set_grad(), zero_grad(), grad_ref()
 *
 * - Error policy: Use `std::invalid_argument` for shape mismatches and
 *   `std::out_of_range` for index bounds, matching the behaviour in this
 *   implementation. Tests in the codebase rely on these exception types.
 *
 * - Gradient semantics:
 *     - get_grad() returns a copy (value) of the gradient if present, or a
 *       zeros-valued tensor if absent.
 *     - set_grad() copies values into the backend's gradient storage,
 *       allocating lazily if required.
 *     - zero_grad() zeros the gradient storage if present but does not
 *       necessarily allocate it.
 *     - grad_ref() returns a mutable reference to the internal gradient
 *       storage, allocating it if needed. This is an internal hook and
 *       callers are expected to synchronize access when used across threads.
 *
 * - Copy/Move semantics: Copy constructors should deep-copy the gradient
 *   storage to preserve autograd state; move operations should transfer
 *   ownership efficiently without unnecessary copies.
 *
 * - Memory layout: Keep contiguous storage semantics. `data_ptr()` must
 *   return a pointer to a contiguous memory region (row-major Eigen order
 *   is used here). If your backend uses non-host memory (GPU), document
 *   how callers should obtain/access data (e.g., via explicit host-transfer
 *   methods or a null `data_ptr()` and separate read/write API).
 *
 * - Performance notes: Implement efficient kernels for heavy ops (matmul,
 *   reductions) by delegating to optimized libraries if available. Avoid
 *   hidden copies during `reshape()` or transposes when possible.
 *
 * - Thread-safety: Backends are not required to be thread-safe by default.
 *   If the backend offers concurrent access guarantees, document them.
 *   Pay special attention to `grad_ref()` which returns a mutable reference
 *   and therefore must be protected by the caller if used concurrently.
 *
 * - 4D mapping: This implementation flattens d2,d3,d4 into columns (rows=d1,
 *   cols=d2*d3*d4). If you change this mapping for a new backend, ensure
 *   compatibility with high-level tensor indexing used across the repo.
 *
 * - Testing: Add unit-tests covering copy/move, reshape correctness,
 *   arithmetic ops, gradient allocation/copying, NaN detection, and
 *   correctness of 4D indexing. See existing tests for `EigenTensorBackend`
 *   behaviour as a reference.
 */
class EigenTensorBackend
{
   public:
    // -----------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------
    // Default constructor
    // - Do not allocate storage here. Backend implementations that use device
    //   memory (GPU) should keep construction lightweight and expose explicit
    //   allocation helpers if needed.
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
    // Logical shape accessor. Keep the returned vector stable until reshape
    // is called to avoid surprising callers.
    const std::vector<Index>& shape() const
    {
        return m_shape;
    }

    // reshape enforces total element count equality and may reallocate storage.
    // Implementations should preserve linear storage order when copying elements
    // to new storage to avoid subtle re-ordering bugs.
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

    // 1D linear access into contiguous storage.
    // - Throws std::out_of_range on bounds violation.
    // - For device-backed backends, document whether this performs a host copy.
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

    // 2D indexed access (row, col)
    // - Valid only when the logical shape has exactly 2 dimensions.
    // - Throws std::invalid_argument for wrong dimensionality and std::out_of_range
    //   for bounds violations.
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

    // 4D logical access mapping d2,d3,d4 into a single column index.
    // - This mapping is part of the public contract; alternative backends must
    //   preserve or clearly document different mapping semantics.
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

    // Generic N-D indexed access. For common dims (1/2/4) it delegates to
    // specialized overloads. For other dims it computes a linearized index in
    // row-major order; throw std::invalid_argument for dimension mismatches.
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
    // - All binary ops validate shape compatibility and throw std::invalid_argument
    //   on mismatches. Implementers should avoid hidden copies where possible.
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

    // Return a pointer to contiguous host memory. For device backends, document
    // whether this is a host mirror, a device pointer, or an invalid operation.
    const float* data_ptr() const
    {
        return m_data.data();
    }
    // Mutable pointer to contiguous host memory. Use with care; concurrent
    // modifications must be synchronized by the caller.
    float* mutable_data_ptr()
    {
        return m_data.data();
    }

    // -----------------------------------------------------------------
    // Gradient
    // -----------------------------------------------------------------
    // Gradient storage semantics and expectations for other backends:
    // - get_grad() returns a *value copy* of the gradient. If no grad exists,
    //   a zeros-valued tensor with matching logical rows/cols must be returned.
    // - set_grad() copies the provided gradient into the backend's gradient
    //   storage, allocating it lazily if needed. Implementations should ensure
    //   shape compatibility and avoid implicit device transfers unless clearly
    //   documented (GPU backends should provide an explicit host-copy path).
    // - zero_grad() zeros existing gradient storage but does not force allocation.
    // - grad_ref() returns a mutable reference to the internal gradient
    //   storage, allocating it if needed. This method is intended for
    //   internal use; callers must ensure synchronization when used across
    //   threads.
    // Retrieve a copy of the gradient tensor. Returns zeros if unallocated.
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

    // Mutable reference to internal gradient; intended for internal use only.
    EigenTensorBackend& grad_ref()
    {
        // Return a mutable reference to the internal gradient storage. This will
        // allocate lazily if not present. Because the returned reference is
        // mutable and can be used to change internal state, callers must ensure
        // proper synchronization when used concurrently. For GPU-based backends
        // that don't expose host-mutable storage, either implement an
        // equivalent behaviour (e.g., host mirror) or document the different
        // behaviour clearly and provide alternate accessors.
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
