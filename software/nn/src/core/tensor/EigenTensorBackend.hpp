#ifndef EIGEN_TENSOR_BACKEND_HPP
#define EIGEN_TENSOR_BACKEND_HPP

#include <Eigen/Dense>

#include "nn/tensor/ITensorBackend.hpp"

// Lightweight Eigen-backed Tensor backend implementation.
// Provides a thin wrapper around Eigen::MatrixXf to satisfy the
// ITensorBackend interface used throughout the codebase.
//
// Notes:
// - This class stores the primary data in `m_data` and lazily
//   allocates a gradient backend (`m_grad_backend`) when gradient
//   operations are requested.
// - Shape information is stored in `m_shape` for multi-dimensional
//   compatibility. The underlying storage is always 2D (rows x cols).
// - Keep this header header-only friendly for small inline helpers,
//   while most heavy logic lives in the corresponding .cpp file.
namespace nn
{

class EigenTensorBackend : public ITensorBackend
{
   public:
    // -- Construction -------------------------------------------------
    /// Default constructs an empty backend (0x0 matrix).
    EigenTensorBackend();

    /// Construct from an existing Eigen matrix (copy semantics).
    explicit EigenTensorBackend(const Eigen::MatrixXf& data);

    /// Construct by moving an Eigen matrix into the backend.
    explicit EigenTensorBackend(Eigen::MatrixXf&& data);

    // -- Shape / Allocation -------------------------------------------
    /// Allocate a 2D matrix with given rows and cols.
    void construct(Index rows, Index cols) override;

    /// Allocate a 4D-shaped tensor (stored flat in the matrix).
    void construct(Index d1, Index d2, Index d3, Index d4) override;

    /// Allocate using a generic shape view (std::span). Non-owning.
    void construct(std::span<const Index> shape) override;

    // -- Element access (flattened or multi-index) --------------------
    float& at(Index i) override;
    const float& at(Index i) const override;
    float& at(Index row, Index col) override;
    const float& at(Index row, Index col) const override;
    float& at(Index d1, Index d2, Index d3, Index d4) override;
    const float& at(Index d1, Index d2, Index d3, Index d4) const override;
    float& at(std::span<const Index> indices) override;
    const float& at(std::span<const Index> indices) const override;

    // -- Shape queries and reshaping ---------------------------------
    const std::vector<Index>& shape() const override;
    void reshape(std::span<const Index> new_shape) override;
    Index rows() const override;
    Index cols() const override;
    Index size() const override;

    // -- Sub-views and blocks ----------------------------------------
    std::unique_ptr<ITensorBackend> row(Index i) const override;
    std::unique_ptr<ITensorBackend> col(Index j) const override;
    std::unique_ptr<ITensorBackend> leftCols(Index n) const override;
    std::unique_ptr<ITensorBackend> topRows(Index n) const override;

    std::unique_ptr<ITensorBackend> block(Index row, Index col, Index rows,
                                          Index cols) const override;
    void setBlock(Index row, Index col, const ITensorBackend& block) override;

    // -- Arithmetic operations (elementwise) -------------------------
    std::unique_ptr<ITensorBackend> add(const ITensorBackend& other) const override;
    std::unique_ptr<ITensorBackend> multiply(const ITensorBackend& other) const override;
    void add_scalar(float scalar) override;
    void multiply_scalar(float scalar) override;

    // -- Linear algebra ----------------------------------------------
    std::unique_ptr<ITensorBackend> matmul(const ITensorBackend& other) const override;
    std::unique_ptr<ITensorBackend> transpose() const override;

    // -- Activation helpers ------------------------------------------
    std::unique_ptr<ITensorBackend> relu() const override;
    std::unique_ptr<ITensorBackend> leaky_relu(float alpha) const override;

    // -- Losses, norms and reductions -------------------------------
    float mean_squared_error(const ITensorBackend& target) const override;
    float norm() const override;
    float sum() const override;
    std::unique_ptr<ITensorBackend> sum_rows() const override;
    std::unique_ptr<ITensorBackend> sum_cols() const override;

    // -- Gradient handling ------------------------------------------
    /// Zero the gradient (allocates grad backend if needed).
    void zero_grad() override;

    /// Set the gradient from another backend (copy semantics).
    void set_grad(const ITensorBackend& grad) override;

    /// Accessors to gradient (const and mutable).
    const ITensorBackend& grad() const override;
    ITensorBackend& grad() override;

    // -- Element-wise math operations -------------------------------
    std::unique_ptr<ITensorBackend> sqrt() const override;
    std::unique_ptr<ITensorBackend> square() const override;
    std::unique_ptr<ITensorBackend> abs() const override;
    std::unique_ptr<ITensorBackend> divide(const ITensorBackend& other) const override;
    std::unique_ptr<ITensorBackend> divide_scalar(float scalar) const override;

    // -- Validation and comparison ---------------------------------
    bool hasNaN() const override;
    bool equals(const ITensorBackend& other) const override;

    // -- Initialization helpers ------------------------------------
    void fill(float value) override;
    void set_zero() override;
    void set_ones() override;

    // -- Raw data access (backwards compatibility) -----------------
    const float* data_ptr() const override;
    Index data_rows() const override;
    Index data_cols() const override;
    float* mutable_data_ptr() override;

    // -- Direct Eigen accessors (convenience) ---------------------
    // Returns a const reference to the underlying Eigen matrix.
    const Eigen::MatrixXf& get_data() const
    {
        return m_data;
    }

    // Mutable access to the underlying Eigen matrix.
    Eigen::MatrixXf& get_data()
    {
        return m_data;
    }

    // Returns a reference to the gradient matrix, allocating a zeroed
    // gradient matrix lazily if necessary. The gradient backend is itself
    // an EigenTensorBackend so we return the underlying matrix reference.
    const Eigen::MatrixXf& get_grad() const
    {
        if (!m_grad_backend)
        {
            if (m_data.rows() == 0 || m_data.cols() == 0)
            {
                // If m_data is empty, initialize m_grad_backend with an empty matrix
                m_grad_backend = std::make_unique<EigenTensorBackend>(Eigen::MatrixXf(0, 0));
            }
            else
            {
                // Otherwise, initialize with zeros of the correct size
                m_grad_backend = std::make_unique<EigenTensorBackend>(
                    Eigen::MatrixXf::Zero(m_data.rows(), m_data.cols()));
            }
        }
        return m_grad_backend->m_data;
    }

    Eigen::MatrixXf& get_grad()
    {
        if (!m_grad_backend)
        {
            if (m_data.rows() == 0 || m_data.cols() == 0)
            {
                m_grad_backend = std::make_unique<EigenTensorBackend>(Eigen::MatrixXf(0, 0));
            }
            else
            {
                m_grad_backend = std::make_unique<EigenTensorBackend>(
                    Eigen::MatrixXf::Zero(m_data.rows(), m_data.cols()));
            }
        }
        return m_grad_backend->m_data;
    }

    // -- Utilities -------------------------------------------------
    std::unique_ptr<ITensorBackend> clone() const override;
    void copy_from(const ITensorBackend& other) override;
    std::unique_ptr<ITensorBackend> slice(std::span<const int> indices) const override;

    // Primary storage for tensor values (rows x cols layout).
    Eigen::MatrixXf m_data;

    // Lazily-allocated gradient backend (mutable so const methods can initialize it).
    mutable std::unique_ptr<EigenTensorBackend> m_grad_backend;

    // Logical shape (N-dimensions) exposed to the rest of the system.
    std::vector<Index> m_shape;

    // Helper to calculate total size from shape
    static Index calculate_total_size(std::span<const Index> shape);
};

} // namespace nn

#endif // EIGEN_TENSOR_BACKEND_HPP