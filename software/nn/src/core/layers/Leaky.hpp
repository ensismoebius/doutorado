#ifndef LIF_HPP
#define LIF_HPP

#include <Eigen/Dense>
#include <memory>
#include <utility>

#include "../tensor/Tensor.hpp"
#include "SurrogateGradient.hpp"
#include "core/layers/Module.hpp"

#ifdef DEBUG
#include "core/utility/printTensor.hpp"
#endif

/**
 * @brief A layer of Leaky Integrate-and-Fire (LIF) neurons, a fundamental component for Spiking
 * Neural Networks (SNNs).
 *
 * This struct implements a layer of LIF neurons, which are a fundamental building
 * block for Spiking Neural Networks (SNNs). The key characteristics of this
 * implementation are:
 *
 * 1.  **Stateful**: The neuron's most important property, its membrane potential
 *     (`v_mem`), is preserved across multiple calls to the `forward` function.
 *     This allows it to integrate inputs over time.
 *
 * 2.  **Vectorized**: It uses the Eigen library to perform calculations on
 *     entire matrices of neurons at once, which is much more efficient than
 *     looping through each neuron individually.
 *
 * 3.  **Trainable**: It includes a `backward` pass that uses a surrogate
 *     gradient, making it possible to train the network using backpropagation.
 */
struct Leaky : public Module
{
   public:
    [[nodiscard]] auto params() -> std::vector<nn::Tensor*> override
    {
        return {&resistance, &voltage_threshold};
    }

    // --- Parameters for LIF neuron dynamics ---

    /// @brief The simulation time step (dt).
    float dt = 1.0F;

    /// @brief Membrane resistance (R). Used to calculate the membrane time constant.
    nn::Tensor resistance = nn::Tensor(Eigen::MatrixXf::Constant(1, 1, 1.0F));

    /// @brief Membrane capacitance (C). Used with R to calculate the membrane time constant.
    float capacitance = 1.0F;

    /// @brief If the membrane potential exceeds this value, the neuron fires a spike.
    nn::Tensor voltage_threshold = nn::Tensor(Eigen::MatrixXf::Constant(1, 1, 1.0F));

    /// @brief Controls the reset mechanism after a spike.
    bool reset_zero = true;

    /// @brief The potential to reset to if `reset_zero` is true.
    float reset_potential = 0.0F;

    // Persistent membrane potential (stateful, snnTorch-like)

    /// @brief Caches the membrane potential *before* spike/reset for the backward pass.
    Eigen::MatrixXf v_mem_pre_spike;

    /// @brief The core state of the neuron layer. Each element is one neuron's potential.
    Eigen::MatrixXf v_mem;

    /// @brief Caches the membrane potential from the previous time step, v(t-1), for backprop.
    Eigen::MatrixXf v_mem_t_minus_1;

    /// @brief The surrogate gradient strategy.
    std::shared_ptr<ISurrogateGradient> surrogate_gradient;

    /**
     * @brief Construct a new Leaky object
     *
     * @param dt_ Time step
     * @param R_ Resistance
     * @param C_ Capacitance
     * @param V_thresh_ Voltage threshold
     * @param reset_zero_ Whether to reset membrane potential to zero after spike
     * @param surrogate_grad The surrogate gradient implementation to use.
     */
    explicit Leaky(float dt_ = 1.0F,              // time step
                   float R_ = 1.0F,               // resistance
                   float C_ = 1.0F,               // capacitance
                   float V_thresh_ = 1.0F,        // voltage threshold
                   bool reset_zero_ = true,       // reset to zero or subtract threshold
                   float reset_potential_ = 0.0F, // reset potential value
                   std::shared_ptr<ISurrogateGradient> surrogate_grad =
                       std::make_shared<ExponentialSurrogate>())
        : dt(dt_),                                                     // time step
          resistance(nn::Tensor(Eigen::MatrixXf::Constant(1, 1, R_))), // resistance
          capacitance(C_),                                             // capacitance
          voltage_threshold(
              nn::Tensor(Eigen::MatrixXf::Constant(1, 1, V_thresh_))), // voltage threshold
          reset_zero(reset_zero_),           // reset to zero or subtract threshold
          reset_potential(reset_potential_), // reset potential value
          surrogate_gradient(std::move(surrogate_grad))
    {
    }

    /**
     * @brief Simulates one time step of the neuron's dynamics.
     *
     * This function performs the "Leaky", "Integrate", "Fire", and "Reset"
     * steps for an entire layer of neurons in a vectorized manner. It follows the
     * discrete-time LIF neuron equation.
     * @param input The input current for this time step.
     */
    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        // Ensure v_mem is correctly sized, initializing if necessary
        if (v_mem.rows() != input.rows() || v_mem.cols() != input.get_data_ref().cols())
            [[unlikely]]
        {
            v_mem.resize(input.get_data_ref().rows(), input.get_data_ref().cols());
            v_mem.setZero();
        }

        // Initialize output tensor
        Eigen::MatrixXf output;

        // The membrane time constant (tau = R * C) determines how quickly potential leaks.
        // Beta is the discrete-time decay factor derived from the continuous-time
        // decay equation, representing the "leaky" nature of the neuron.
        float const tau = resistance.get_data_ref()(0, 0) * capacitance;
        float const beta = std::exp(-dt / tau);

        // snnTorch-like: persistent v_mem, decay, and reset on spike
        if (v_mem.size() == 0 || v_mem.rows() != input.get_data_ref().rows() ||
            v_mem.cols() != input.get_data_ref().cols()) [[unlikely]]
        {
            v_mem = Eigen::MatrixXf::Zero(input.get_data_ref().rows(), input.get_data_ref().cols());
        }

        // Cache the membrane potential from the previous time step, v(t-1), for the backward pass.
        if (requires_grad)
        {
            v_mem_t_minus_1 = v_mem;
        }

        // 1. Decay (Leaky): The membrane potential from the previous time step (`v_mem`)
        // is decayed by a factor of `beta`. If there were no input, the potential
        // would exponentially decay toward its resting potential (0).
        v_mem = v_mem * beta;

        // 2. Integrate: The new input current (`input.data`) is added to the
        // decayed membrane potential. This is the "integrate" part of the neuron's name.
        v_mem = v_mem + input.get_data_ref();

        // 3. Cache: The potential is saved just before the spike check. This is for
        // the `backward` pass, as the surrogate gradient is calculated based on this
        // pre-spike potential.
        if (requires_grad)
        {
            v_mem_pre_spike = v_mem;
        }
#ifdef DEBUG
        {
            std::ostringstream oss;
            oss << "Updated V_mem - " << static_cast<const void*>(this);
            printTensor(nn::Tensor{v_mem}, oss.str());
        }
#endif
        // 4. Fire (Spike): Generate a spike (1.0) if potential exceeds the threshold.
        // This is a non-differentiable step function, which is why we need surrogate
        // gradients for training.
        output = (v_mem.array() > voltage_threshold.get_data_ref()(0, 0)).cast<float>();

        // 5. Reset: For every neuron that fired a spike, its membrane potential must be reset.
        if (reset_zero)
        {
            // Hard Reset: The potential is reset to a fixed value, `reset_potential`
            // (which is often 0).
            v_mem = (output.array() == 1.0F)
                        .select(                       //
                            Eigen::MatrixXf::Constant( //
                                v_mem.rows(),          //
                                v_mem.cols(),          //
                                reset_potential        //
                                ),
                            v_mem //
                        );
        }
        else
        {
            // Soft Reset: The threshold voltage is subtracted from the membrane
            // potential. This retains any "excess" potential that was accumulated
            // above the threshold.
            v_mem = v_mem.array() - output.array() * voltage_threshold.get_data_ref()(0, 0);
        }

        return nn::Tensor{output};
    }

    /**
     * @brief Backward pass for Leaky neuron.
     *
     * The Problem: The derivative of the spike function in the forward pass is a
     * step function (zero almost everywhere, and infinite at the threshold). This
     * prevents learning via backpropagation, a problem often called the "dead
     * neuron problem".
     *
     * The Solution: We use a **surrogate gradient**. Instead of the true (and
     * useless) derivative, we substitute a "fake" or "surrogate" derivative that
     * has a non-zero value in a small region around the threshold. This allows a
     * gradient to flow back through the neuron, enabling training.
     *
     * @param grad_output Gradient from the next layer
     * @return Tensor Gradient w.r.t. input
     */
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override
    {
        // --- Surrogate Gradient Calculation ---
        const Eigen::MatrixXf surrogate_grad =
            surrogate_gradient
                ->calculate(nn::Tensor(v_mem_pre_spike), voltage_threshold.get_data_ref()(0, 0))
                .get_data_ref();

        // Gradient of the loss with respect to the pre-spike membrane potential (dL/dv_pre)
        // This is the starting point for calculating other gradients via the chain rule.
        const Eigen::MatrixXf grad_v_pre =
            grad_output.get_data_ref().array() * surrogate_grad.array();

        // --- Gradient for voltage_threshold ---
        // dL/dV_th = dL/ds * ds/dV_th = dL/ds * (-ds/dv_pre) = - (dL/ds * ds/dv_pre) = -grad_v_pre
        // Since V_th is a scalar, we sum the gradients from all neurons.
        const float dL_dVth = -grad_v_pre.sum();
        voltage_threshold.set_grad(Eigen::MatrixXf::Constant(1, 1, dL_dVth));

        // --- Gradient for resistance ---
        // dL/dR = dL/dv_pre * dv_pre/dR, where dv_pre/dR = v(t-1) * d(beta)/dR
        const float R = resistance.get_data_ref()(0, 0);
        const float C = capacitance;
        const float tau = R * C;
        if (tau > 1e-6) [[likely]]
        { // Avoid division by zero if R or C are zero
            const float beta = std::exp(-dt / tau);
            const float d_beta_dR = (beta * dt) / (C * R * R);

            // dL/dbeta = dL/dv_pre * dv_pre/dbeta = grad_v_pre * v(t-1)
            const Eigen::MatrixXf dL_dbeta_matrix = grad_v_pre.array() * v_mem_t_minus_1.array();
            const float dL_dbeta = dL_dbeta_matrix.sum();
            const float dL_dR = dL_dbeta * d_beta_dR;
            resistance.set_grad(Eigen::MatrixXf::Constant(1, 1, dL_dR));
        }
        else
        {
            resistance.set_grad(Eigen::MatrixXf::Zero(1, 1));
        }

        // Apply the chain rule: the gradient flowing to the input (`grad_input`) is
        // the gradient from the subsequent layer (`grad_output`) multiplied by this
        // local surrogate gradient.
        // dL/dI = dL/dv_pre * dv_pre/dI = grad_v_pre * 1
        const auto& grad_input = grad_v_pre;

        return nn::Tensor{grad_input};
    }
};

#endif // LIF_HPP
