#ifndef ITENSOR_BACKEND_HPP
#define ITENSOR_BACKEND_HPP

#include <memory>
#include <span>
#include <vector>

namespace nn
{

using Index = size_t; // Generic index type, not tied to Eigen

class ITensorBackend
{
   public:
    virtual ~ITensorBackend() = default;

    // Construction
    virtual void construct(Index rows, Index cols) = 0;
    virtual void construct(Index d1, Index d2, Index d3, Index d4) = 0;
    virtual void construct(const std::vector<Index>& shape) = 0;

    // Data access
    virtual float& at(Index row, Index col) = 0;
    virtual const float& at(Index row, Index col) const = 0;
    virtual float& at(Index d1, Index d2, Index d3, Index d4) = 0;
    virtual const float& at(Index d1, Index d2, Index d3, Index d4) const = 0;
    virtual float& at(const std::vector<Index>& indices) = 0;
    virtual const float& at(const std::vector<Index>& indices) const = 0;

    // Shape and size
    virtual const std::vector<Index>& shape() const = 0;
    virtual Index rows() const = 0;
    virtual Index cols() const = 0;
    virtual Index size() const = 0;

    // Row/column operations
    virtual std::unique_ptr<ITensorBackend> row(Index i) const = 0;
    virtual std::unique_ptr<ITensorBackend> col(Index j) const = 0;
    virtual std::unique_ptr<ITensorBackend> leftCols(Index n) const = 0;
    virtual std::unique_ptr<ITensorBackend> topRows(Index n) const = 0;

    // Block operations
    virtual std::unique_ptr<ITensorBackend> block(Index row, Index col, Index rows,
                                                  Index cols) const = 0;
    virtual void setBlock(Index row, Index col, const ITensorBackend& block) = 0;

    // Element-wise operations
    virtual std::unique_ptr<ITensorBackend> add(const ITensorBackend& other) const = 0;
    virtual std::unique_ptr<ITensorBackend> multiply(const ITensorBackend& other) const = 0;
    virtual void add_scalar(float scalar) = 0;
    virtual void multiply_scalar(float scalar) = 0;

    // Matrix operations
    virtual std::unique_ptr<ITensorBackend> matmul(const ITensorBackend& other) const = 0;
    virtual std::unique_ptr<ITensorBackend> transpose() const = 0;

    // Activation functions
    virtual std::unique_ptr<ITensorBackend> relu() const = 0;
    virtual std::unique_ptr<ITensorBackend> leaky_relu(float alpha) const = 0;

    // Loss functions
    virtual float mean_squared_error(const ITensorBackend& target) const = 0;
    virtual float norm() const = 0;

    // Gradient operations
    virtual void zero_grad() = 0;
    virtual void set_grad(const ITensorBackend& grad) = 0;
    virtual const ITensorBackend& grad() const = 0;
    virtual ITensorBackend& grad() = 0;

    // Data access for backward compatibility
    virtual const float* data_ptr() const = 0;
    virtual Index data_rows() const = 0;
    virtual Index data_cols() const = 0;

    // Utility
    virtual std::unique_ptr<ITensorBackend> clone() const = 0;
    virtual void copy_from(const ITensorBackend& other) = 0;
    virtual std::unique_ptr<ITensorBackend> slice(std::span<const int> indices) const = 0;
};

} // namespace nn

#endif // ITENSOR_BACKEND_HPP