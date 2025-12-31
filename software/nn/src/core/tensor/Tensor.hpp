#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <Eigen/Dense>
#include <functional>
#include <numeric>
#include <span>
#include <vector>

namespace nn
{
class Tensor
{
   public:
    // Constructors
    Tensor() = default;
    explicit Tensor(const Eigen::MatrixXf& data);
    explicit Tensor(Eigen::MatrixXf&& data);
    Tensor(Eigen::Index rows, Eigen::Index cols);
    Tensor(Eigen::Index dim1, Eigen::Index dim2, Eigen::Index dim3,
           Eigen::Index dim4);                               // New 4D constructor
    explicit Tensor(const std::vector<Eigen::Index>& shape); // New general N-D constructor

    // Getters for data and gradient
    [[nodiscard]] auto get_data_ref() const -> const Eigen::MatrixXf&;
    [[nodiscard]] auto get_grad_ref() const -> const Eigen::MatrixXf&;
    auto get_data_ref() -> Eigen::MatrixXf&;
    auto get_grad_ref() -> Eigen::MatrixXf&;

    // Setter for gradient
    void set_grad(const Eigen::MatrixXf& grad);
    void set_grad(Eigen::MatrixXf&& grad);

    // Setter for data
    void set_data(const Eigen::MatrixXf& data);

    // Shape and size information
    [[nodiscard]] auto get_shape() const -> const std::vector<Eigen::Index>&;
    [[nodiscard]] auto rows() const -> Eigen::Index;
    [[nodiscard]] auto cols() const -> Eigen::Index;
    [[nodiscard]] auto size() const -> Eigen::Index;

    // Element access for 2D and 4D tensors (new)
    auto at(Eigen::Index row, Eigen::Index col) -> float&;
    [[nodiscard]] auto at(Eigen::Index row, Eigen::Index col) const -> const float&;
    auto at(Eigen::Index d1, Eigen::Index d2, Eigen::Index d3, Eigen::Index d4) -> float&;
    [[nodiscard]] auto at(Eigen::Index d1, Eigen::Index d2, Eigen::Index d3, Eigen::Index d4) const
        -> const float&;
    // General N-D access
    auto at(const std::vector<Eigen::Index>& indices) -> float&;
    [[nodiscard]] auto at(const std::vector<Eigen::Index>& indices) const -> const float&;

    // Conversion to std::vector
    template <typename vector_type>
    [[nodiscard]] auto toVector() const -> std::vector<vector_type>;

    // Slice operation (non-owning view over indices)
    [[nodiscard]] auto slice(std::span<const int> indices) const -> Tensor;

    // Zero out the gradient
    void zero_grad();

   private:
    static Eigen::Index calculate_total_size(const std::vector<Eigen::Index>& shape)
    {
        return std::accumulate(
            shape.begin(), shape.end(), Eigen::Index(1), std::multiplies<Eigen::Index>());
    }

    Eigen::MatrixXf m_data;
    Eigen::MatrixXf m_grad;
    std::vector<Eigen::Index> m_shape; // New member to store N-dimensional shape
}; // End of class Tensor

// Template implementations must be available in the header, outside the class but inside the
// namespace.
template <typename vector_type>
auto Tensor::toVector() const -> std::vector<vector_type>
{
    std::vector<vector_type> vec;
    vec.reserve(static_cast<size_t>(m_data.size()));
    const auto* data_ptr = m_data.data();
    for (Eigen::Index i = 0; i < m_data.size(); ++i)
    {
        vec.push_back(static_cast<vector_type>(data_ptr[i]));
    }
    return vec;
}

} // namespace nn

#endif // TENSOR_HPP