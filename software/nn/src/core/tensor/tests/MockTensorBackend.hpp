#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "../ITensorBackend.hpp"

namespace nn
{

class MockTensorBackend : public ITensorBackend
{
   public:
    MockTensorBackend();
    explicit MockTensorBackend(const std::vector<Index>& shape);
    MockTensorBackend(const std::vector<Index>& shape, std::vector<float> data);

    // Introspection helpers for tests
    [[nodiscard]] const std::vector<std::string>& log() const noexcept
    {
        return m_calls;
    }
    void clear_log() noexcept
    {
        m_calls.clear();
    }

    // ITensorBackend interface
    void construct(Index rows, Index cols) override;
    void construct(Index d1, Index d2, Index d3, Index d4) override;
    void construct(const std::vector<Index>& shape) override;

    float& at(Index i) override;
    const float& at(Index i) const override;
    float& at(Index row, Index col) override;
    const float& at(Index row, Index col) const override;
    float& at(Index d1, Index d2, Index d3, Index d4) override;
    const float& at(Index d1, Index d2, Index d3, Index d4) const override;
    float& at(const std::vector<Index>& indices) override;
    const float& at(const std::vector<Index>& indices) const override;

    const std::vector<Index>& shape() const override;
    void reshape(const std::vector<Index>& new_shape) override;
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
    float sum() const override;
    std::unique_ptr<ITensorBackend> sum_rows() const override;
    std::unique_ptr<ITensorBackend> sum_cols() const override;

    void zero_grad() override;
    void set_grad(const ITensorBackend& grad) override;
    const ITensorBackend& grad() const override;
    ITensorBackend& grad() override;

    // Element-wise math operations
    std::unique_ptr<ITensorBackend> sqrt() const override;
    std::unique_ptr<ITensorBackend> square() const override;
    std::unique_ptr<ITensorBackend> abs() const override;
    std::unique_ptr<ITensorBackend> divide(const ITensorBackend& other) const override;
    std::unique_ptr<ITensorBackend> divide_scalar(float scalar) const override;

    // Initialization
    void fill(float value) override;
    void set_zero() override;
    void set_ones() override;

    const float* data_ptr() const override;
    Index data_rows() const override;
    Index data_cols() const override;
    float* mutable_data_ptr() override;

    bool hasNaN() const override;
    bool equals(const ITensorBackend& other) const override;

    std::unique_ptr<ITensorBackend> clone() const override;
    void copy_from(const ITensorBackend& other) override;
    std::unique_ptr<ITensorBackend> slice(std::span<const int> indices) const override;

   private:
    [[nodiscard]] Index offset_2d(Index row, Index col) const;
    [[nodiscard]] Index offset_nd(const std::vector<Index>& indices) const;
    [[nodiscard]] std::string shape_to_string() const;
    void ensure_shape(std::size_t dims) const;
    void ensure_same_shape(const ITensorBackend& other, const char* op) const;
    void log_call(const std::string& name) const;

    std::vector<float> m_data;
    std::vector<Index> m_shape;
    mutable std::unique_ptr<MockTensorBackend> m_grad;
    mutable std::vector<std::string> m_calls;
};

} // namespace nn
