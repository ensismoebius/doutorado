#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <Eigen/Dense>

class Tensor
{
  public:
    // Constructors
    Tensor() = default;
    explicit Tensor(const Eigen::MatrixXf& data);
    explicit Tensor(Eigen::MatrixXf&& data);
    Tensor(Eigen::Index rows, Eigen::Index cols);

    // Getters for data and gradient
    auto get_data_ref() -> Eigen::MatrixXf&;
    auto get_data_ref() const -> const Eigen::MatrixXf&;
    auto get_grad_ref() -> Eigen::MatrixXf&;
    auto get_grad_ref() const -> const Eigen::MatrixXf&;

    // Setter for gradient
    void set_grad(const Eigen::MatrixXf& grad);
    void set_grad(Eigen::MatrixXf&& grad);

    // Setter for data
    void set_data(const Eigen::MatrixXf& data);

    // Shape and size information
    auto get_shape() const -> std::vector<Eigen::Index>;
    auto rows() const -> Eigen::Index;
    auto cols() const -> Eigen::Index;
    auto size() const -> Eigen::Index;

    // Slice operation
    auto slice(const std::vector<int>& indices) const -> Tensor;

    // Zero out the gradient
    void zero_grad();

  private:
    Eigen::MatrixXf m_data;
    Eigen::MatrixXf m_grad;
};

#endif // TENSOR_HPP