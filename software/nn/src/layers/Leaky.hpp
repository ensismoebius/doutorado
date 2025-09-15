#ifndef LIF_HPP
#define LIF_HPP

#include "../tensor/Tensor.hpp"
#include "layers/Module.hpp"
#include <Eigen/Dense>

#ifdef DEBUG
#include "util/printTensor.hpp"
#endif

/**
 * @brief Leaky Integrate-and-Fire (LIF) neuron activation function.
 * Simulates the membrane potential update and spike generation.
 */
struct Leaky : public Module {
public:
  auto params() -> std::vector<Tensor *> {
    return {&resistance, &voltage_threshold};
  }
  // Parameters for LIF neuron
  float dt = 1.0F;
  Tensor resistance = Tensor(Eigen::MatrixXf::Constant(1, 1, 5.0F));
  float capacitance = 1.0F;
  Tensor voltage_threshold = Tensor(Eigen::MatrixXf::Constant(1, 1, 2.0F));
  bool reset_zero = true;
  float reset_potential = 0.0F;  // New: configurable reset value
  float surrogate_window = 1.0F; // New: configurable surrogate window width

  // Persistent membrane potential (stateful, snnTorch-like)
  Eigen::MatrixXf v_mem_cache;
  Eigen::MatrixXf v_mem;

  /**
   * @brief Construct a new Leaky object
   *
   * @param dt_ Time step
   * @param R_ Resistance
   * @param C_ Capacitance
   * @param V_thresh_ Voltage threshold
   * @param reset_zero_ Whether to reset membrane potential to zero after spike
   */
  Leaky(float dt_ = 1.0F,              // time step
        float R_ = 5.0F,               // resistance
        float C_ = 1.0F,               // capacitance
        float V_thresh_ = 1.0F,        // voltage threshold
        bool reset_zero_ = true,       // reset to zero or subtract threshold
        float reset_potential_ = 0.0F, // reset potential value
        float surrogate_window_ = 1.0F // surrogate gradient window
        )
      : dt(dt_),                                                       // time step
        resistance(Eigen::MatrixXf::Constant(1, 1, R_)),               // resistance
        capacitance(C_),                                               // capacitance
        voltage_threshold(Eigen::MatrixXf::Constant(1, 1, V_thresh_)), // voltage threshold
        reset_zero(reset_zero_),            // reset to zero or subtract threshold
        reset_potential(reset_potential_),  // reset potential value
        surrogate_window(surrogate_window_) // surrogate gradient window
  {}

  auto forward(const Tensor &input) -> Tensor override {

    // snnTorch-like: persistent v_mem, decay, and reset on spike
    if (v_mem.size() == 0 || v_mem.rows() != input.data.rows() ||
        v_mem.cols() != input.data.cols()) {
      v_mem = Eigen::MatrixXf::Zero(input.data.rows(), input.data.cols());
    }

    // Initialize output tensor
    Eigen::MatrixXf output = Eigen::MatrixXf::Zero(input.data.rows(), input.data.cols());

    // Use exponential decay: beta = exp(-dt/tau)
    float const tau = resistance.data(0, 0) * capacitance;
    float const beta = std::exp(-dt / tau);

    // Update membrane potential with decay and scaled input
    v_mem = v_mem * beta + resistance.data(0, 0) * input.data * dt;
#ifdef DEBUG
    printTensor(Tensor(v_mem), "Updated V_mem");
#endif

    // Spike condition and reset
    for (int i = 0; i < v_mem.rows(); ++i) {
      for (int j = 0; j < v_mem.cols(); ++j) {
        if (v_mem(i, j) > voltage_threshold.data(0, 0)) {
          output(i, j) = 1.0F; // spike
          if (reset_zero) {
            v_mem(i, j) = reset_potential;
          } else {
            v_mem(i, j) = v_mem(i, j) - voltage_threshold.data(0, 0);
          }
        } else {
          output(i, j) = 0.0F; // no spike
        }
      }
    }

    v_mem_cache = v_mem; // Cache for backward if needed
    return {output};
  }

  /**
   * @brief Backward pass for Leaky neuron.
   * Implements surrogate gradient descent with Hard Tanh.
   *
   * @param grad_output Gradient from the next layer
   * @return Tensor Gradient w.r.t. input
   */
  auto backward(const Tensor &grad_output) -> Tensor override {

    // Surrogate Gradient Descent with Hard Tanh
    // d spike/d V_mem ≈ 1 if |V_mem - voltage_threshold| < surrogate_window, else 0
    Eigen::MatrixXf grad_input =
        Eigen::MatrixXf::Zero(grad_output.data.rows(), grad_output.data.cols());

    for (int i = 0; i < grad_output.data.rows(); ++i) {
      for (int j = 0; j < grad_output.data.cols(); ++j) {
        float const prev_v_mem = v_mem_cache(i, j); // Typo fixed
        float const diff = prev_v_mem - voltage_threshold.data(0, 0);
        float const surrogate_grad =
            (std::abs(diff) < surrogate_window) ? 1.0F : 0.0F; // Configurable window
        grad_input(i, j) = grad_output.data(i, j) * surrogate_grad;
      }
    }
    return {grad_input};
  }
};

#endif // LIF_HPP