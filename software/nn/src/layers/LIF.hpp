#ifndef LIF_HPP
#define LIF_HPP

#include "../tensor/Tensor.hpp"
#include "layers/Module.hpp"
#include <Eigen/Dense>

/**
 * @brief Leaky Integrate-and-Fire (LIF) neuron activation function.
 * Simulates the membrane potential update and spike generation.
 */
struct LIF : public Module {
  // Parameters for LIF neuron
  float dt = 1.0F;
  float R = 5.0F;
  float C = 1.0F;
  float V_thresh = 2.0F;
  bool reset_zero = true;

  // State cache for backward (not used in this simple version)
  Eigen::MatrixXf V_mem_cache;

  LIF(float dt_ = 1.0f, float R_ = 5.0f, float C_ = 1.0f, float V_thresh_ = 2.0f, bool reset_zero_ = true) : dt(dt_), R(R_), C(C_), V_thresh(V_thresh_), reset_zero(reset_zero_) {}

  auto forward(const Tensor &input) -> Tensor override {
    // input.data: [batch_size x features], interpreted as input current I_in
    Eigen::MatrixXf V_mem = Eigen::MatrixXf::Zero(input.data.rows(), input.data.cols());
    Eigen::MatrixXf output = V_mem;

    float const tau = R * C;

    for (int i = 0; i < input.data.rows(); ++i) {
      for (int j = 0; j < input.data.cols(); ++j) {
        // Update membrane potential
        V_mem(i, j) = V_mem(i, j) + (dt / tau) * (-V_mem(i, j) + input.data(i, j) * R);

        // Spike condition
        if (V_mem(i, j) > V_thresh) {
          output(i, j) = 1.0F; // spike
          if (reset_zero) {
            V_mem(i, j) = 0.0F;
          } else {
            V_mem(i, j) = V_mem(i, j) - V_thresh;
          }
        } else {
          output(i, j) = 0.0F; // no spike
        }
      }
    }

    V_mem_cache = V_mem; // Cache for backward if needed
    return {output};
  }

  auto backward(const Tensor &grad_output) -> Tensor override {
    // LIF is non-differentiable at spike, so here we return zeros (straight-through estimator could be used)
    Eigen::MatrixXf grad_input = Eigen::MatrixXf::Zero(grad_output.data.rows(), grad_output.data.cols());
    return {grad_input};
  }
};

#endif // LIF_HPP