#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <memory>
#include <span>
#include <vector>

#include "ITensorBackend.hpp"

namespace nn
{

class Tensor
{
   public:
    // Constructors
    Tensor();
    Tensor(std::unique_ptr<ITensorBackend> backend);
    Tensor(Index rows, Index cols);
    Tensor(Index dim1, Index d2, Index d3, Index d4);
    Tensor(const std::vector<Index>& shape);

    // Static factory methods for creating initialized tensors
    static auto constant(Index rows, Index cols, float value) -> Tensor;
    static auto zeros(Index rows, Index cols) -> Tensor;
    static auto ones(Index rows, Index cols) -> Tensor;

    // Default copy/move semantics
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) = default;
    Tensor& operator=(Tensor&& other) = default;
    ~Tensor() = default;

    // Shape and size information
    auto get_shape() const -> const std::vector<Index>&;
    [[nodiscard]] auto rows() const noexcept -> Index;
    [[nodiscard]] auto cols() const noexcept -> Index;
    [[nodiscard]] auto size() const noexcept -> Index;

    // Element access for 1D, 2D and 4D tensors
    auto at(Index i) -> float&;
    [[nodiscard]] auto at(Index i) const -> const float&;
    auto at(Index row, Index col) -> float&;
    [[nodiscard]] auto at(Index row, Index col) const -> const float&;
    auto at(Index d1, Index d2, Index d3, Index d4) -> float&;
    [[nodiscard]] auto at(Index d1, Index d2, Index d3, Index d4) const -> const float&;
    // General N-D access
    auto at(const std::vector<Index>& indices) -> float&;
    [[nodiscard]] auto at(const std::vector<Index>& indices) const -> const float&;

    // Row and column access
    auto row(Index i) const -> Tensor;
    auto col(Index j) const -> Tensor;
    auto leftCols(Index n) const -> Tensor;
    auto topRows(Index n) const -> Tensor;

    // Block operations
    auto block(Index row, Index col, Index rows, Index cols) const -> Tensor;
    void setBlock(Index row, Index col, const Tensor& block);

    // Element-wise operations
    auto add(const Tensor& other) const -> Tensor;
    auto multiply(const Tensor& other) const -> Tensor;
    auto add_scalar(float scalar) const -> Tensor;
    auto multiply_scalar(float scalar) const -> Tensor;

    // Matrix operations
    auto matmul(const Tensor& other) const -> Tensor;
    auto transpose() const -> Tensor;

    // Activation functions
    auto relu() const -> Tensor;
    auto leaky_relu(float alpha = 0.01f) const -> Tensor;

    // Loss functions
    auto mean_squared_error(const Tensor& target) const -> float;
    auto norm() const -> float;
    auto sum() const -> float;
    auto sum_rows() const -> Tensor; // Sum across columns, return column vector (rows, 1)
    auto sum_cols() const -> Tensor; // Sum across rows, return row vector (1, cols)

    // Element-wise math operations
    auto sqrt() const -> Tensor;
    auto square() const -> Tensor;
    auto abs() const -> Tensor;
    auto divide(const Tensor& other) const -> Tensor;
    auto divide_scalar(float scalar) const -> Tensor;

    // Array-like interface for chaining (returns *this for method chaining)
    auto array() const -> const Tensor&
    {
        return *this;
    }

    // Initialization
    void fill(float value);
    void set_zero();
    void set_ones();
    // Compatibility aliases
    void setZero()
    {
        set_zero();
    }
    void setOnes()
    {
        set_ones();
    }
    void setConstant(float value)
    {
        fill(value);
    }

    // Data access (legacy helpers)
    const float* data() const
    {
        return data_ptr();
    }
    float* mutable_data()
    {
        return mutable_data_ptr();
    }
    // Operator() convenience
    float& operator()(Index i, Index j)
    {
        return at(i, j);
    }
    const float& operator()(Index i, Index j) const
    {
        return at(i, j);
    }

    // Data access (for backward compatibility with existing code)
    const float* data_ptr() const;
    float* mutable_data_ptr();

    // Gradient access
    auto grad() const -> Tensor;
    auto grad() -> Tensor&;
    void set_grad(const Tensor& new_grad);

    // Conversion to std::vector
    template <typename vector_type>
    [[nodiscard]] auto toVector() const -> std::vector<vector_type>;

    // Slice operation (non-owning view over indices)
    [[nodiscard]] auto slice(std::span<const int> indices) const -> Tensor;

    // Zero out the gradient
    void zero_grad();

    // Operator overloads for convenience (member functions to avoid ambiguity)
    auto operator+(const Tensor& other) const -> Tensor
    {
        return add(other);
    }
    auto operator-(const Tensor& other) const -> Tensor;
    auto operator*(const Tensor& other) const -> Tensor
    {
        return multiply(other);
    }
    auto operator*(float scalar) const -> Tensor;
    auto operator+(float scalar) const -> Tensor;
    auto operator-(float scalar) const -> Tensor;
    auto operator/(float scalar) const -> Tensor;

    auto get_backend() const -> const ITensorBackend*
    {
        return m_backend.get();
    }

   private:
    std::unique_ptr<ITensorBackend> m_backend;
};

// Template implementations must be available in the header, outside the class but inside the
// namespace.
template <typename vector_type>
auto Tensor::toVector() const -> std::vector<vector_type>
{
    // This needs to be implemented using the backend
    // For now, return empty vector - will be implemented when we have backend access
    return std::vector<vector_type>();
}

} // namespace nn

#endif // TENSOR_HPP