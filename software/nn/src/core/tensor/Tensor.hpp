#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <Eigen/Dense>

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
    [[nodiscard]] auto get_shape() const -> std::vector<Eigen::Index>;
    [[nodiscard]] auto rows() const -> Eigen::Index;
    [[nodiscard]] auto cols() const -> Eigen::Index;
    [[nodiscard]] auto size() const -> Eigen::Index;

    // Conversion to std::vector<float>
    [[nodiscard]] auto toVector() const -> std::vector<float>;

    // Slice operation
    [[nodiscard]] auto slice(const std::vector<int>& indices) const -> Tensor;

    // Zero out the gradient
    void zero_grad();

   private:
    Eigen::MatrixXf m_data;
    Eigen::MatrixXf m_grad;
};
} // namespace nn

#endif // TENSOR_HPP