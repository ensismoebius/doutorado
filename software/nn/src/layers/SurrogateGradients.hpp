#ifndef SURROGATE_GRADIENTS_HPP
#define SURROGATE_GRADIENTS_HPP

#include <Eigen/Dense>

namespace SurrogateGradients {

// Exponential / SuperSpike
inline auto exponential(const Eigen::MatrixXf &v_mem_pre_spike, float voltage_threshold,
                        float sharpness = 1.0F) -> Eigen::MatrixXf {
  const Eigen::MatrixXf diff_abs = (v_mem_pre_spike.array() - voltage_threshold).abs();
  return (1.0F / sharpness) * ((-diff_abs / sharpness).array().exp());
}

// Boxcar
inline auto boxcar(const Eigen::MatrixXf &v_mem_pre_spike, float voltage_threshold,
                   float surrogate_window = 0.5F) -> Eigen::MatrixXf {
  const Eigen::MatrixXf diff_abs = (v_mem_pre_spike.array() - voltage_threshold).abs();
  return (diff_abs.array() < (surrogate_window / 2.0F)).cast<float>();
}

} // namespace SurrogateGradients

#endif // SURROGATE_GRADIENTS_HPP
