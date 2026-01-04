#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <Eigen/Dense>
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

    // Element access for 2D and 4D tensors
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
    auto add_scalar(float scalar) -> Tensor&;
    auto multiply_scalar(float scalar) -> Tensor&;

    // Matrix operations
    auto matmul(const Tensor& other) const -> Tensor;
    auto transpose() const -> Tensor;

    // Activation functions
    auto relu() const -> Tensor;
    auto leaky_relu(float alpha = 0.01f) const -> Tensor;

    // Loss functions
    auto mean_squared_error(const Tensor& target) const -> float;
    auto norm() const -> float;

    // Conversion to std::vector
    template <typename vector_type>
    [[nodiscard]] auto toVector() const -> std::vector<vector_type>;

    // Slice operation (non-owning view over indices)
    [[nodiscard]] auto slice(std::span<const int> indices) const -> Tensor;

    // Zero out the gradient
    void zero_grad();

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