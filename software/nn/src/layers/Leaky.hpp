#ifndef LIF_HPP
#define LIF_HPP

#include "../tensor/Tensor.hpp"
#include "layers/Module.hpp"
#include <Eigen/Dense>

/**
 * @brief Leaky Integrate-and-Fire (LIF) neuron activation function.
 * Simulates the membrane potential update and spike generation.
 */
struct Leaky : public Module {
  // Parameters for LIF neuron
  float dt = 1.0F;
  float resistence = 5.0F;
  float capacitance = 1.0F;
  float voltage_threshold = 2.0F;
  bool reset_zero = true;

  // Persistent membrane potential (stateful, snnTorch-like)
  Eigen::MatrixXf v_mem_cache;
  Eigen::MatrixXf v_mem;

  Leaky(float dt_ = 1.0f, float R_ = 5.0f, float C_ = 1.0f, float V_thresh_ = 2.0f, bool reset_zero_ = true) : dt(dt_), resistence(R_), capacitance(C_), voltage_threshold(V_thresh_), reset_zero(reset_zero_), v_mem() {}

  auto forward(const Tensor &input) -> Tensor override {
    // snnTorch-like: persistent v_mem, decay, and reset on spike
    if (v_mem.size() == 0 || v_mem.rows() != input.data.rows() || v_mem.cols() != input.data.cols()) {
      v_mem = Eigen::MatrixXf::Zero(input.data.rows(), input.data.cols());
    }
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(input.data.rows(), input.data.cols());

    // Typical snnTorch decay: v_mem = v_mem * beta + input
    // Here, beta = exp(-dt/tau) for continuous LIF, but for simplicity, use beta = 1 - (dt/tau)
    float const tau = resistence * capacitance;
    float const beta = 1.0F - (dt / tau);

    // Update membrane potential with decay and input
    v_mem = v_mem * beta + input.data;

    // Spike condition and reset
    for (int i = 0; i < v_mem.rows(); ++i) {
      for (int j = 0; j < v_mem.cols(); ++j) {
        if (v_mem(i, j) > voltage_threshold) {
          output(i, j) = 1.0F; // spike
          if (reset_zero) {
            v_mem(i, j) = 0.0F;
          } else {
            v_mem(i, j) = v_mem(i, j) - voltage_threshold;
          }
        } else {
          output(i, j) = 0.0F; // no spike
        }
      }
    }

    v_mem_cache = v_mem; // Cache for backward if needed
    return {output};
  }

  auto backward(const Tensor &grad_output) -> Tensor override {
    // Surrogate Gradient Descent with Hard Tanh
    // d spike/d V_mem ≈ 1 if |V_mem - voltage_threshold| < 1, else 0
    Eigen::MatrixXf grad_input = Eigen::MatrixXf::Zero(grad_output.data.rows(), grad_output.data.cols());
    for (int i = 0; i < grad_output.data.rows(); ++i) {
      for (int j = 0; j < grad_output.data.cols(); ++j) {
        float const prev_v_men = v_mem_cache(i, j);
        float const diff = prev_v_men - voltage_threshold;
        float const surrogate_grad = (std::abs(diff) < 1.0F) ? 1.0F : 0.0F; // Hard Tanh window
        grad_input(i, j) = grad_output.data(i, j) * surrogate_grad;
      }
    }
    return {grad_input};
  }
};

#endif // LIF_HPP