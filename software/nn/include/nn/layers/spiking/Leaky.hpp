#ifndef NN_LAYERS_LEAKY_HPP
#define NN_LAYERS_LEAKY_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

#include "nn/layers/base/Module.hpp"
#include "nn/layers/spiking/ExponentialSurrogate.hpp"
#include "nn/tensor/Tensor.hpp"

#ifdef DEBUG
#include "nn/utility/printTensor.hpp"
#endif

/**
 * @file Leaky.hpp
 * @brief Leaky Integrate-and-Fire (LIF) layer.
 *
 * This is the simplest (single-step) spiking neuron layer in the project.
 * Conceptually it matches snnTorch's `snn.Leaky` behavior: it keeps a persistent
 * membrane potential `v_mem` across calls to `forward()`, emits spikes when
 * crossing a threshold, and then resets.
 *
 * Shape contract:
 * - Input is a 2D tensor (rows x cols). In non-temporal use, rows usually act as
 *   batch and cols as features.
 * - This layer is *stateful*; if you change the input shape between calls,
 *   `v_mem` is resized and reset to zeros.
 *
 * Training contract:
 * - `forward(requires_grad=true)` must be called before `backward()` so cached
 *   state (`v_mem_pre_spike`, `v_mem_t_minus_1`) is available.
 * - The spike function is non-differentiable; gradients are approximated via a
 *   configurable surrogate (`ISurrogateGradient`).
 */

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
 *     Trainable parameters: `resistance`, `voltage_threshold`, and `capacitance`.
 *     All three are exposed via `params()` and receive gradient updates.
 */
template <typename Backend>
struct LeakyImpl : public Module<Backend>
{
   public:
    /// Tensor type for the active compute backend.
    using Tensor = typename Module<Backend>::Tensor;

    // --- Parameters for LIF neuron dynamics ---

    /// @brief The simulation time step (time_step).
    float time_step = 1.0F;

    /// @brief Membrane resistance (R). Used to calculate the membrane time constant.
    Tensor resistance = Tensor::constant(1, 1, 1.0F);

    /// @brief Membrane capacitance (C). Used with R to calculate the membrane time constant.
    /// Stored as a 1×1 trainable tensor so the optimizer can update it via backprop.
    Tensor capacitance = Tensor::constant(1, 1, 1.0F);

    /// @brief If the membrane potential exceeds this value, the neuron fires a spike.
    Tensor voltage_threshold = Tensor::constant(1, 1, 1.0F);

    /// @brief Controls the reset mechanism after a spike.
    bool reset_zero = true;

    /// @brief The potential to reset to if `reset_zero` is true.
    float reset_potential = 0.0F;

    // Persistent membrane potential (stateful, snnTorch-like)

    /// @brief Caches the membrane potential *before* spike/reset for the backward pass.
    Tensor v_mem_pre_spike;

    /// @brief The core state of the neuron layer. Each element is one neuron's potential.
    Tensor v_mem;

    /// @brief Caches the membrane potential from the previous time step, v(t-1), for backprop.
    Tensor v_mem_t_minus_1;

    /// @brief The surrogate gradient strategy.
    std::shared_ptr<ISurrogateGradient> surrogate_gradient;
    // Persistent parameter pointer storage for returning spans.
    std::array<nn::Tensor*, 3> param_ptrs_{{&resistance, &voltage_threshold, &capacitance}};

    [[nodiscard]] auto params() -> std::span<nn::Tensor*> override
    {
        return std::span<nn::Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    auto state_dict() const -> std::map<std::string, nn::Tensor> override
    {
        std::map<std::string, nn::Tensor> d;
        d["resistance"] = resistance;
        d["capacitance"] = capacitance;
        d["voltage_threshold"] = voltage_threshold;
        return d;
    }

    void load_state_dict(const std::map<std::string, nn::Tensor>& sd) override
    {
        auto it = sd.find("resistance");
        if (it != sd.end()) resistance = it->second;
        it = sd.find("capacitance");
        if (it != sd.end()) capacitance = it->second;
        it = sd.find("voltage_threshold");
        if (it != sd.end()) voltage_threshold = it->second;
    }

    void reset_state() override
    {
        if (v_mem.size() > 0)
        {
            v_mem.setZero();
        }
        if (v_mem_pre_spike.size() > 0)
        {
            v_mem_pre_spike.setZero();
        }
        if (v_mem_t_minus_1.size() > 0)
        {
            v_mem_t_minus_1.setZero();
        }
    }

    /**
     * @brief Construct a new Leaky object
     *
     * @param time_step_ Time step
     * @param resistance_ Resistance
     * @param capacitance_ Capacitance
     * @param voltage_threshold_ Voltage threshold
     * @param reset_zero_ Whether to reset membrane potential to zero after spike
     * @param surrogate_grad The surrogate gradient implementation to use.
     */
    explicit LeakyImpl(float time_step_ = 1.0F, // time step
        float resistance_ = 1.0F,               // resistance
        float capacitance_ = 1.0F,              // capacitance
        float voltage_threshold_ = 1.0F,        // voltage threshold
        bool reset_zero_ = true,                // reset to zero or subtract threshold
        float reset_potential_ = 0.0F,          // reset potential value
        std::shared_ptr<ISurrogateGradient> surrogate_grad =
            std::make_shared<ExponentialSurrogate>())
        : time_step(time_step_),
          resistance(Tensor::constant(1, 1, resistance_)),
          capacitance(Tensor::constant(1, 1, capacitance_)),
          voltage_threshold(Tensor::constant(1, 1, voltage_threshold_)),
          reset_zero(reset_zero_),
          reset_potential(reset_potential_),
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
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        // Ensure v_mem is correctly sized, initializing if necessary
        if (v_mem.rows() != input.rows() || v_mem.cols() != input.cols()) [[unlikely]]
        {
            v_mem = Tensor(input.rows(), input.cols());
            v_mem.setZero();
        }

        // The membrane time constant (tau = R * C) determines how quickly potential leaks.
        // Beta is the discrete-time decay factor derived from the continuous-time
        // decay equation, representing the "leaky" nature of the neuron.
        // Clamp R and C to keep tau strictly positive and beta numerically stable.
        // Rationale: during training, optimizers can temporarily drive raw R/C <= 0;
        // this guard prevents invalid tau and unstable exp() behavior in forward pass.
        constexpr float kMinPositiveParam = 1e-6F;
        float const R = std::max(kMinPositiveParam, resistance.at(0, 0));
        float const C = std::max(kMinPositiveParam, capacitance.at(0, 0));
        float const tau = R * C;
        float const beta = std::exp(-time_step / tau);

        // snnTorch-like: persistent v_mem, decay, and reset on spike
        // NOTE: This check is redundant with the initialization above, but is
        // kept as-is for safety/clarity. If you refactor, ensure state semantics
        // remain identical.
        if (v_mem.size() == 0 || v_mem.rows() != input.rows() || v_mem.cols() != input.cols())
            [[unlikely]]
        {
            v_mem = Tensor(input.rows(), input.cols());
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
        // Implementation note: this uses explicit loops rather than a vectorized
        // compare operation; for large tensors this can become a hotspot.
        Tensor output(input.rows(), input.cols());
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
    auto backward(const Tensor& grad_output) -> Tensor override
    {
        // --- Surrogate Gradient Calculation ---
        const auto surrogate_grad =
            surrogate_gradient->calculate(v_mem_pre_spike, voltage_threshold.at(0, 0));

        // Gradient of the loss with respect to the pre-spike membrane potential (dL/dv_pre)
        // This is the starting point for calculating other gradients via the chain rule.
        Tensor grad_v_pre_mat = grad_output.multiply(surrogate_grad);

        // --- Gradient for voltage_threshold ---
        // dL/dV_th = dL/ds * ds/dV_th = dL/ds * (-ds/dv_pre) = - (dL/ds * ds/dv_pre) =
        // -grad_v_pre Since V_th is a scalar, we sum the gradients from all neurons.
        float dL_dVth = -grad_v_pre_mat.sum();
        Tensor vth_grad(1, 1);
        vth_grad.at(0, 0) = dL_dVth;
        voltage_threshold.set_grad(vth_grad);

        // --- Gradient for resistance ---
        // dL/dR = dL/dv_pre * dv_pre/dR, where dv_pre/dR = v(t-1) * d(beta)/dR
        constexpr float kMinPositiveParam = 1e-6F;
        const float raw_R = resistance.at(0, 0);
        const float raw_C = capacitance.at(0, 0);
        const float R = std::max(kMinPositiveParam, raw_R);
        const float C = std::max(kMinPositiveParam, raw_C);
        const float tau = R * C;
        if (tau > 1e-12F) [[likely]]
        { // Avoid division by zero if R or C are zero
            const float beta = std::exp(-time_step / tau);
            // Keep gradient consistent with clamp-at-use semantics: once raw_R is in the
            // clamped region, do not backprop through the clamped surrogate expression.
            const float d_beta_dR =
                (raw_R > kMinPositiveParam) ? (beta * time_step) / (C * R * R) : 0.0F;

            // dL/dbeta = dL/dv_pre * dv_pre/dbeta = grad_v_pre * v(t-1)
            float dL_dbeta = grad_v_pre_mat.multiply(v_mem_t_minus_1).sum();

            // --- Gradient for resistance ---
            const float dL_dR = dL_dbeta * d_beta_dR;
            Tensor r_grad(1, 1);
            r_grad.at(0, 0) = dL_dR;
            resistance.set_grad(r_grad);

            // --- Gradient for capacitance (symmetric to dL/dR) ---
            // dBeta/dC = beta * dt / (R * C^2)
            // Same clamp-boundary rule for C: avoid artificial gradient amplification when
            // the raw value is below the positive-stability floor.
            const float d_beta_dC =
                (raw_C > kMinPositiveParam) ? (beta * time_step) / (R * C * C) : 0.0F;
            Tensor c_grad(1, 1);
            c_grad.at(0, 0) = dL_dbeta * d_beta_dC;
            capacitance.set_grad(c_grad);
        }
        else
        {
            Tensor r_grad(1, 1);
            r_grad.set_zero();
            resistance.set_grad(r_grad);
            Tensor c_grad(1, 1);
            c_grad.set_zero();
            capacitance.set_grad(c_grad);
        }

        // Apply the chain rule: the gradient flowing to the input (`grad_input`) is
        // the gradient from the subsequent layer (`grad_output`) multiplied by this
        // local surrogate gradient.
        // dL/dI = dL/dv_pre * dv_pre/dI = grad_v_pre * 1
        return grad_v_pre_mat;
    }
};

#endif // NN_LAYERS_LEAKY_HPP
