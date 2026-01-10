#ifndef LEAKY_BPTT_HPP
#define LEAKY_BPTT_HPP

#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "nn/layers/Module.hpp"
#include "nn/layers/SurrogateGradient.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @file LeakyBPTT.hpp
 * @brief LIF/Leaky Integrate-and-Fire dynamics with a full (explicit) BPTT-style backward.
 *
 * Design intent (snnTorch mental model):
 * - This is similar to snnTorch's leaky neuron module that is unrolled over time.
 * - Input is provided as a *flattened* time-major matrix: rows are concatenated time slices.
 *
 * Shape contract:
 * - forward() expects `input` with shape (T*B, F)
 *   where T = time_steps, B = batch size, F = features.
 * - Rows are ordered as: t0 batch rows, then t1 batch rows, ..., t(T-1) batch rows.
 * - Invariants: `input.rows() % time_steps == 0` and `batch_size = rows / time_steps`.
 *
 * State semantics:
 * - `v_mem` is the persistent membrane state across calls (like keeping hidden state).
 * - `reset_state()` clears that persistent state so the next forward starts from zero.
 * - When `requires_grad==true`, we cache `v_mem_history` so backward can do BPTT.
 *
 * Notes on gradients:
 * - The backward() here is intentionally “manual BPTT”: it iterates time in reverse and
 *   propagates gradients through the recurrence `v[t] = beta * v[t-1] + input[t]`.
 * - Because spiking includes a hard threshold + reset, the gradient uses a surrogate
 *   derivative for the spike event and simplified/approximate handling of the reset.
 */
/**
 * @brief Leaky Integrate-and-Fire (LIF) layer with Full Backpropagation Through Time (BPTT).
 *
 * This module expects a flattened time-series input of shape (Time * Batch, Features).
 * It unrolls the simulation over `time_steps`, maintaining state across the sequence,
 * and computes accurate gradients through time during the backward pass.
 */
struct LeakyBPTT : public Module
{
   public:
    [[nodiscard]] auto params() -> std::vector<nn::Tensor*> override
    {
        // These are the trainable scalars exposed to optimizers.
        // Note: dt and capacitance are plain floats (not optimized here).
        return {&resistance, &voltage_threshold};
    }

    /// @brief The simulation time step (dt).
    float dt = 1.0F;

    /// @brief Membrane resistance (R).
    nn::Tensor resistance = nn::Tensor::constant(1, 1, 1.0F);

    /// @brief Membrane capacitance (C).
    float capacitance = 1.0F;

    /// @brief Voltage threshold.
    nn::Tensor voltage_threshold = nn::Tensor::constant(1, 1, 1.0F);

    // State management
    nn::Tensor v_mem;         ///< Persistent state after last processed time step (shape: B x F)
    nn::Tensor v_mem_history; ///< Cached pre-reset membrane values for BPTT (shape: (T*B) x F)
    nn::Tensor
        spike_history; ///< Placeholder for spike cache (currently unused in this implementation)

    // Configuration
    int time_steps; ///< Number of time steps in the input sequence
    bool reset_zero = true;
    bool readout_mode =
        false; ///< If true, outputs membrane potential instead of spikes (Regression)
    float reset_potential = 0.0F;

    std::shared_ptr<ISurrogateGradient> surrogate_gradient;

    explicit LeakyBPTT(int time_steps_, float dt_ = 1.0F, float R_ = 1.0F, float C_ = 1.0F,
                       float V_thresh_ = 1.0F, bool reset_zero_ = true,
                       float reset_potential_ = 0.0F, bool readout_mode_ = false,
                       std::shared_ptr<ISurrogateGradient> surrogate_grad =
                           std::make_shared<ExponentialSurrogate>())
        : dt(dt_), time_steps(time_steps_), readout_mode(readout_mode_)
    {
        resistance.at(0, 0) = R_;
        capacitance = C_;
        voltage_threshold.at(0, 0) = V_thresh_;
        reset_zero = reset_zero_;
        reset_potential = reset_potential_;
        surrogate_gradient = std::move(surrogate_grad);
    }

    void reset_state() override
    {
        // Clearing v_mem means the next forward() will re-initialize state to zeros.
        // This matches the common “reset hidden state between sequences” pattern.
        v_mem = nn::Tensor(); // Clear state
    }

    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        // Infer Batch Size
        int total_rows = input.rows();
        if (total_rows % time_steps != 0)
        {
            throw std::invalid_argument("LeakyBPTT: Input rows must be divisible by time_steps");
        }
        int batch_size = total_rows / time_steps;
        int features = input.cols();

        // Initialize state if needed
        if (v_mem.rows() != batch_size || v_mem.cols() != features)
        {
            v_mem = nn::Tensor(batch_size, features);
            v_mem.setZero();
        }

        // Prepare Output and History
        nn::Tensor output(total_rows, features);
        if (requires_grad)
        {
            // We cache v_pre (the membrane after decay+input, before spike/reset) for each time.
            // Backward() reads this cache as its “pre-activation” for surrogate gradients.
            v_mem_history = nn::Tensor(total_rows, features);
        }

        float const tau = resistance.at(0, 0) * capacitance;
        float const beta = std::exp(-dt / tau);
        float threshold_val = voltage_threshold.at(0, 0);

        // Time Loop
        for (int t = 0; t < time_steps; ++t)
        {
            // Extract input slice for this time step
            // Implementation detail:
            // - We iterate manually instead of using a block view to stay within the current
            //   Tensor interface and keep the logic explicit/teachable.

            // 1. Decay & Integrate
            // v[t] = v[t-1] * beta + input[t]

            // Optimized: v_mem = v_mem * beta + input_slice
            // We access input rows: [t*batch, (t+1)*batch)

            int offset = t * batch_size;

            // We can iterate batch items
            for (int b = 0; b < batch_size; ++b)
            {
                for (int f = 0; f < features; ++f)
                {
                    float vin = input.at(offset + b, f);
                    float v = v_mem.at(b, f);
                    v = v * beta + vin;

                    // Store Pre-Spike for history
                    if (requires_grad)
                    {
                        v_mem_history.at(offset + b, f) = v;
                    }

                    if (readout_mode)
                    {
                        // Readout Mode: Output is V_mem
                        output.at(offset + b, f) = v;
                        // Readout mode is used for regression/continuous outputs.
                        // There is no spike event, so no threshold/reset is applied.
                    }
                    else
                    {
                        // Spiking Mode
                        float s = (v > threshold_val) ? 1.0f : 0.0f;
                        output.at(offset + b, f) = s;

                        // Reset
                        if (s > 0.5f)
                        {
                            if (reset_zero)
                                v = reset_potential;
                            else
                                v -= threshold_val;
                        }
                    }

                    // Update State
                    v_mem.at(b, f) = v;
                }
            }
        }

        return output;
    }

    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override
    {
        // Grad Output: (Time*Batch, Feat)
        int total_rows = grad_output.rows();
        int batch_size = total_rows / time_steps;
        int features = grad_output.cols();

        nn::Tensor grad_input(total_rows, features);
        nn::Tensor grad_next_state(batch_size, features); // dL / dv[t+1] (recurrence accumulator)
        grad_next_state.setZero();

        float const tau = resistance.at(0, 0) * capacitance;
        float const beta = std::exp(-dt / tau);
        float threshold_val = voltage_threshold.at(0, 0);

        // Accumulators for params
        float dL_dVth_sum = 0.0f;
        float dL_dR_sum = 0.0f;
        float d_beta_dR =
            (tau > 1e-6) ? (beta * dt) / (capacitance * resistance.at(0, 0) * resistance.at(0, 0))
                         : 0.0f;
        // Note: d_beta_dR is derived from beta = exp(-dt/(R*C)).
        // This implementation uses a scalar R and C shared across all neurons.

        // BPTT Loop (Reverse Time)
        for (int t = time_steps - 1; t >= 0; --t)
        {
            int offset = t * batch_size;

            for (int b = 0; b < batch_size; ++b)
            {
                for (int f = 0; f < features; ++f)
                {
                    // Get history
                    float v_pre = v_mem_history.at(offset + b, f);
                    float grad_out = grad_output.at(offset + b, f);

                    // Get Gradient from Next Step (Recurrent)
                    float grad_from_next = grad_next_state.at(b, f);

                    // dL / dS[t] = grad_output[t]  (Direct loss)
                    // Term 1: From Output
                    float grad_v_pre = 0.0f;
                    float dvpost_dvpre = 1.0f;

                    if (readout_mode)
                    {
                        // dL/dV = grad_out
                        // No surrogate.
                        // d(next)/d(curr) = beta. (No reset).
                        grad_v_pre = grad_out + grad_from_next * beta;
                    }
                    else
                    {
                        // Surrogate dS/dv
                        float surr = surrogate_gradient->calculate_scalar(v_pre, threshold_val);

                        // Term 2: From Next State v[t+1] (Recurrent)
                        // ...

                        if (reset_zero)
                        {
                            // Approximation: hard reset is non-differentiable; treat as identity
                            // for the purpose of propagating gradients through the state.
                            dvpost_dvpre = 1.0f;
                        }
                        else
                        {
                            // Subtraction: v - S*th
                            // d = 1 - surr * th
                            dvpost_dvpre = 1.0f - surr * threshold_val;
                        }

                        grad_v_pre = grad_out * surr + grad_from_next * beta * dvpost_dvpre;
                    }

                    // Store dL/dI = grad_v_pre * 1
                    grad_input.at(offset + b, f) = grad_v_pre;

                    // Update grad_next_state for t-1
                    grad_next_state.at(b, f) = grad_v_pre;

                    // Params Gradients
                    dL_dVth_sum += -grad_v_pre; // Simplified
                    // Note: threshold gradient is simplified here. A more complete treatment
                    // would include how V_th influences spike/reset and therefore the state.

                    // R: dL/dR = dL/dbeta * dbeta/dR + ...
                    // approximation for resistance gradient:
                    float v_prev_post = 0.0f;
                    if (t > 0)
                    {
                        // We reconstruct an approximate post-reset previous state using cached
                        // v_pre (and re-applying the spike decision). This keeps backward
                        // self-contained without storing a full spike history.
                        float vp = v_mem_history.at((t - 1) * batch_size + b, f);
                        float s_p = (vp > threshold_val) ? 1.0f : 0.0f;
                        if (reset_zero && s_p > 0.5f)
                            v_prev_post = reset_potential;
                        else if (!reset_zero)
                            v_prev_post = vp - s_p * threshold_val;
                        else
                            v_prev_post = vp;
                    }
                    dL_dR_sum += grad_v_pre * v_prev_post * d_beta_dR;
                }
            }
        }

        // Set Params Grads
        nn::Tensor vth_grad(1, 1);
        vth_grad.at(0, 0) = dL_dVth_sum;
        voltage_threshold.set_grad(vth_grad);

        nn::Tensor r_grad(1, 1);
        r_grad.at(0, 0) = dL_dR_sum;
        resistance.set_grad(r_grad);

        return grad_input;
    }
};

#endif // LEAKY_BPTT_HPP
