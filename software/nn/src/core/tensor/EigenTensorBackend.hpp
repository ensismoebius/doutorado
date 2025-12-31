#ifndef EIGEN_TENSOR_BACKEND_HPP
#define EIGEN_TENSOR_BACKEND_HPP

#include <Eigen/Dense>

#include "ITensorBackend.hpp"

namespace nn
{

class EigenTensorBackend : public ITensorBackend
{
   public:
    EigenTensorBackend();
    explicit EigenTensorBackend(const Eigen::MatrixXf& data);
    explicit EigenTensorBackend(Eigen::MatrixXf&& data);

    // Implement all ITensorBackend methods using Eigen
    void construct(Index rows, Index cols) override;
    void construct(Index d1, Index d2, Index d3, Index d4) override;
    void construct(const std::vector<Index>& shape) override;

    float& at(Index row, Index col) override;
    const float& at(Index row, Index col) const override;
    float& at(Index d1, Index d2, Index d3, Index d4) override;
    const float& at(Index d1, Index d2, Index d3, Index d4) const override;
    float& at(const std::vector<Index>& indices) override;
    const float& at(const std::vector<Index>& indices) const override;

    const std::vector<Index>& shape() const override;
    Index rows() const override;
    Index cols() const override;
    Index size() const override;

    std::unique_ptr<ITensorBackend> row(Index i) const override;
    std::unique_ptr<ITensorBackend> col(Index j) const override;
    std::unique_ptr<ITensorBackend> leftCols(Index n) const override;
    std::unique_ptr<ITensorBackend> topRows(Index n) const override;

    std::unique_ptr<ITensorBackend> block(Index row, Index col, Index rows,
                                          Index cols) const override;
    void setBlock(Index row, Index col, const ITensorBackend& block) override;

    std::unique_ptr<ITensorBackend> add(const ITensorBackend& other) const override;
    std::unique_ptr<ITensorBackend> multiply(const ITensorBackend& other) const override;
    void add_scalar(float scalar) override;
    void multiply_scalar(float scalar) override;

    std::unique_ptr<ITensorBackend> matmul(const ITensorBackend& other) const override;
    std::unique_ptr<ITensorBackend> transpose() const override;

    std::unique_ptr<ITensorBackend> relu() const override;
    std::unique_ptr<ITensorBackend> leaky_relu(float alpha) const override;

    float mean_squared_error(const ITensorBackend& target) const override;
    float norm() const override;

    void zero_grad() override;
    void set_grad(const ITensorBackend& grad) override;
    const ITensorBackend& grad() const override;
    ITensorBackend& grad() override;

    // Data access for backward compatibility
    const float* data_ptr() const override;
    Index data_rows() const override;
    Index data_cols() const override;

    // Direct access to Eigen matrices (for backward compatibility)
    const Eigen::MatrixXf& get_data() const
    {
        return m_data;
    }
    Eigen::MatrixXf& get_data()
    {
        return m_data;
    }
    const Eigen::MatrixXf& get_grad() const
    {
        if (!m_grad_backend)
        {
            // Create a temporary backend for const access
            m_grad_backend = std::make_unique<EigenTensorBackend>(
                Eigen::MatrixXf::Zero(m_data.rows(), m_data.cols()));
        }
        return m_grad_backend->m_data;
    }
    Eigen::MatrixXf& get_grad()
    {
        if (!m_grad_backend)
        {
            m_grad_backend = std::make_unique<EigenTensorBackend>(
                Eigen::MatrixXf::Zero(m_data.rows(), m_data.cols()));
        }
        return m_grad_backend->m_data;
    }

    // Utility
    std::unique_ptr<ITensorBackend> clone() const override;
    void copy_from(const ITensorBackend& other) override;
    Eigen::MatrixXf m_data;
    mutable std::unique_ptr<EigenTensorBackend> m_grad_backend;
    std::vector<Index> m_shape;

    // Helper to calculate total size from shape
    static Index calculate_total_size(const std::vector<Index>& shape);
};

} // namespace nn

#endif // EIGEN_TENSOR_BACKEND_HPP