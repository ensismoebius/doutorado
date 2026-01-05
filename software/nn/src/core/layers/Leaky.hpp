#ifndef LIF_HPP
#define LIF_HPP

#include <cmath>
#include <memory>
#include <utility>

#include "nn/tensor/Tensor.hpp"
#include "SurrogateGradient.hpp"
#include "core/layers/Module.hpp"

#ifdef DEBUG
#include "nn/utility/printTensor.hpp"
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
 * 2.  **Vectorized**: It uses the Tensor API to perform calculations on
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
    nn::Tensor resistance = nn::Tensor::constant(1, 1, 1.0F);

    /// @brief Membrane capacitance (C). Used with R to calculate the membrane time constant.
    float capacitance = 1.0F;

    /// @brief If the membrane potential exceeds this value, the neuron fires a spike.
    nn::Tensor voltage_threshold = nn::Tensor::constant(1, 1, 1.0F);

    /// @brief Controls the reset mechanism after a spike.
    bool reset_zero = true;

    /// @brief The potential to reset to if `reset_zero` is true.
    float reset_potential = 0.0F;

    // Persistent membrane potential (stateful, snnTorch-like)

    /// @brief Caches the membrane potential *before* spike/reset for the backward pass.
    nn::Tensor v_mem_pre_spike;

    /// @brief The core state of the neuron layer. Each element is one neuron's potential.
    nn::Tensor v_mem;

    /// @brief Caches the membrane potential from the previous time step, v(t-1), for backprop.
    nn::Tensor v_mem_t_minus_1;

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
        : dt(dt_)
    {
        resistance = nn::Tensor(1, 1);
        resistance.at(0, 0) = R_;
        capacitance = C_;
        voltage_threshold = nn::Tensor(1, 1);
        voltage_threshold.at(0, 0) = V_thresh_;
        reset_zero = reset_zero_;
        reset_potential = reset_potential_;
        surrogate_gradient = std::move(surrogate_grad);
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
        if (v_mem.rows() != static_cast<int>(input.rows()) ||
            v_mem.cols() != static_cast<int>(input.cols())) [[unlikely]]
        {
            v_mem = nn::Tensor(input.rows(), input.cols());
            v_mem.setZero();
        }

        // The membrane time constant (tau = R * C) determines how quickly potential leaks.
        // Beta is the discrete-time decay factor derived from the continuous-time
        // decay equation, representing the "leaky" nature of the neuron.
        float const tau = resistance(0, 0) * capacitance;
        float const beta = std::exp(-dt / tau);

        // snnTorch-like: persistent v_mem, decay, and reset on spike
        if (v_mem.size() == 0 || v_mem.rows() != static_cast<int>(input.rows()) ||
            v_mem.cols() != static_cast<int>(input.cols())) [[unlikely]]
        {
            v_mem = nn::Tensor(input.rows(), input.cols());
            v_mem.setZero();
        }

        // Cache the membrane potential from the previous time step, v(t-1), for the backward
        // pass.
        if (requires_grad)
        {
            v_mem_t_minus_1 = v_mem;
        }

        // 1. Decay (Leaky): The membrane potential from the previous time step (`v_mem`)
        // is decayed by a factor of `beta`. If there were no input, the potential
        // would exponentially decay toward its resting potential (0).
        v_mem = v_mem.multiply_scalar(beta);

        // 2. Integrate: The new input current (`input.data`) is added to the
        // decayed membrane potential. This is the "integrate" part of the neuron's name.
        v_mem = v_mem.add(input);

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
            printTensor(v_mem, oss.str());
        }
#endif
        // 4. Fire (Spike): Generate a spike (1.0) if potential exceeds the threshold.
        // This is a non-differentiable step function, which is why we need surrogate
        // gradients for training.
        nn::Tensor output(input.rows(), input.cols());
        float threshold_val = voltage_threshold.at(0, 0);
        for (size_t i = 0; i < v_mem.rows(); ++i)
        {
            for (size_t j = 0; j < v_mem.cols(); ++j)
            {
                output.at(i, j) = (v_mem.at(i, j) > threshold_val) ? 1.0f : 0.0f;
            }
        }

        // 5. Reset: For every neuron that fired a spike, its membrane potential must be reset.
        if (reset_zero)
        {
            // Hard Reset: The potential is reset to a fixed value, `reset_potential`
            // (which is often 0).
            for (size_t i = 0; i < v_mem.rows(); ++i)
            {
                for (size_t j = 0; j < v_mem.cols(); ++j)
                {
                    if (output.at(i, j) == 1.0f)
                    {
                        v_mem.at(i, j) = reset_potential;
                    }
                }
            }
        }
        else
        {
            // Soft Reset: The threshold voltage is subtracted from the membrane
            // potential. This retains any "excess" potential that was accumulated
            // above the threshold.
            for (size_t i = 0; i < v_mem.rows(); ++i)
            {
                for (size_t j = 0; j < v_mem.cols(); ++j)
                {
                    v_mem.at(i, j) = v_mem.at(i, j) - output.at(i, j) * threshold_val;
                }
            }
        }

        return output;
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
        const auto surrogate_grad =
            surrogate_gradient->calculate(v_mem_pre_spike, voltage_threshold.at(0, 0));

        // Gradient of the loss with respect to the pre-spike membrane potential (dL/dv_pre)
        // This is the starting point for calculating other gradients via the chain rule.
        nn::Tensor grad_v_pre_mat(grad_output.rows(), grad_output.cols());
        for (size_t i = 0; i < grad_output.rows(); ++i)
        {
            for (size_t j = 0; j < grad_output.cols(); ++j)
            {
                grad_v_pre_mat.at(i, j) = grad_output.at(i, j) * surrogate_grad.at(i, j);
            }
        }

        // --- Gradient for voltage_threshold ---
        // dL/dV_th = dL/ds * ds/dV_th = dL/ds * (-ds/dv_pre) = - (dL/ds * ds/dv_pre) =
        // -grad_v_pre Since V_th is a scalar, we sum the gradients from all neurons.
        float dL_dVth = 0.0f;
        for (size_t i = 0; i < grad_v_pre_mat.rows(); ++i)
        {
            for (size_t j = 0; j < grad_v_pre_mat.cols(); ++j)
            {
                dL_dVth -= grad_v_pre_mat.at(i, j);
            }
        }
        nn::Tensor vth_grad(1, 1);
        vth_grad.at(0, 0) = dL_dVth;
        voltage_threshold.set_grad(vth_grad);

        // --- Gradient for resistance ---
        // dL/dR = dL/dv_pre * dv_pre/dR, where dv_pre/dR = v(t-1) * d(beta)/dR
        const float R = resistance.at(0, 0);
        const float C = capacitance;
        const float tau = R * C;
        if (tau > 1e-6) [[likely]]
        { // Avoid division by zero if R or C are zero
            const float beta = std::exp(-dt / tau);
            const float d_beta_dR = (beta * dt) / (C * R * R);

            // dL/dbeta = dL/dv_pre * dv_pre/dbeta = grad_v_pre * v(t-1)
            float dL_dbeta = 0.0f;
            for (size_t i = 0; i < grad_v_pre_mat.rows(); ++i)
            {
                for (size_t j = 0; j < grad_v_pre_mat.cols(); ++j)
                {
                    dL_dbeta += grad_v_pre_mat.at(i, j) * v_mem_t_minus_1.at(i, j);
                }
            }
            const float dL_dR = dL_dbeta * d_beta_dR;
            nn::Tensor r_grad(1, 1);
            r_grad.at(0, 0) = dL_dR;
            resistance.set_grad(r_grad);
        }
        else
        {
            nn::Tensor r_grad(1, 1);
            r_grad.set_zero();
            resistance.set_grad(r_grad);
        }

        // Apply the chain rule: the gradient flowing to the input (`grad_input`) is
        // the gradient from the subsequent layer (`grad_output`) multiplied by this
        // local surrogate gradient.
        // dL/dI = dL/dv_pre * dv_pre/dI = grad_v_pre * 1
        return grad_v_pre_mat;
    }
};

#endif // LIF_HPP
