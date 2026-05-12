#ifndef LEAKY_BPTT_HPP
#define LEAKY_BPTT_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

#include "layers/base/Module.hpp"
#include "layers/spiking/ExponentialSurrogate.hpp"
#include "tensor/Tensor.hpp"

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
 *   derivative for the spike event and simplified/approximate handling of the reset. *
 * Convergence fixes (summary for onboarding):
 * 1. R/C forward clamp — raw resistance/capacitance is clamped to kMinPositiveParam (1e-6)
 *    before computing tau=R*C and beta=exp(-dt/tau). Prevents NaN/Inf when optimizers
 *    drive parameters non-positive.
 * 2. R/C backward guard — d_beta_dR and d_beta_dC are set to 0 when the raw parameter
 *    is in the clamped region, keeping gradients consistent with the forward clamp.
 * 3. v_post_history — caches the exact post-reset state v[t] after each forward step.
 *    BPTT uses this (not the pre-reset v_pre) for dL/dR and dL/dC, eliminating
 *    systematic bias from reconstructing state at backward time.
 * 4. Full threshold gradient — dL/dVth accumulates both the direct spike term
 *    (-grad_out * surr) and the recurrent reset-path term (grad_from_next * beta *
 *    dvpost_dVth), covering both hard-reset (dvpost_dVth = surr*(v_pre - V_reset))
 *    and soft-reset (dvpost_dVth = -spike + Vth*surr) branches.
 * 5. Readout-mode isolation — when readout_mode=true the forward pass emits v_mem
 *    directly with no spike/reset, and backward treats dynamics as purely continuous
 *    (grad_v_pre = grad_out + grad_from_next * beta), preventing cross-contamination
 *    from the spiking branch. */
/**
 * @brief Leaky Integrate-and-Fire (LIF) layer with Full Backpropagation Through Time (BPTT).
 *
 * This module expects a flattened time-series input of shape (Time * Batch, Features).
 * It unrolls the simulation over `time_steps`, maintaining state across the sequence,
 * and computes accurate gradients through time during the backward pass.
 */
template <typename Backend>
struct LeakyBPTTImpl : public Module<Backend>
{
   public:
    /// Tensor type for the active compute backend.
    using Tensor = typename Module<Backend>::Tensor;

    /// @brief The simulation time step (time_step).
    float time_step = 1.0F;

    /// @brief Membrane resistance (R).
    Tensor resistance = Tensor::constant(1, 1, 1.0F);

    /// @brief Membrane capacitance (C).
    Tensor capacitance = Tensor::constant(1, 1, 1.0F);

    /// @brief Voltage threshold.
    Tensor voltage_threshold = Tensor::constant(1, 1, 1.0F);

    // State management
    Tensor v_mem;         ///< Persistent state after last processed time step (shape: B x F)
    Tensor v_mem_history; ///< Cached pre-reset membrane values for BPTT (shape: (T*B) x F)
    // Added to avoid backward-time reconstruction drift: gradients for beta (and thus R/C)
    // must use the exact previous post-reset state seen in forward.
    Tensor v_post_history; ///< Cached post-reset membrane values for exact recurrence derivatives.
    Tensor spike_history; ///< Placeholder for spike cache (currently unused in this implementation)
    Tensor adapt_a_bptt_; ///< Adaptation variable state (shape: B x F), persists across calls.

    // Configuration
    int time_steps; ///< Number of time steps in the input sequence
    bool reset_zero = true;
    bool readout_mode =
        false; ///< If true, outputs membrane potential instead of spikes (Regression)
    float reset_potential = 0.0F;

    // Spike-frequency adaptation (same semantics as LeakyImpl)
    // effective threshold = voltage_threshold + adapt_a[b,f]
    // adapt_a decays each step by adapt_decay, rises by adapt_coupling on each spike.
    float adapt_decay = 0.9F;
    float adapt_coupling = 0.0F; // 0 = disabled

    std::shared_ptr<ISurrogateGradient> surrogate_gradient;
    // Persistent parameter pointer storage for returning spans.
    std::array<Tensor*, 3> param_ptrs_{{&resistance, &voltage_threshold, &capacitance}};

    [[nodiscard]] auto params() -> std::span<Tensor*> override
    {
        return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    explicit LeakyBPTTImpl(int time_steps_,
        float time_step_ = 1.0F,
        float resistance_ = 1.0F,
        float capacitance_ = 1.0F,
        float voltage_threshold_ = 1.0F,
        bool reset_zero_ = true,
        float reset_potential_ = 0.0F,
        bool readout_mode_ = false,
        std::shared_ptr<ISurrogateGradient> surrogate_grad =
            std::make_shared<ExponentialSurrogate>(),
        float adapt_decay_ = 0.9F,
        float adapt_coupling_ = 0.0F)
        : time_step(time_step_), time_steps(time_steps_), readout_mode(readout_mode_)
    {
        resistance.at(0, 0) = resistance_;
        capacitance.at(0, 0) = capacitance_;
        voltage_threshold.at(0, 0) = voltage_threshold_;
        reset_zero = reset_zero_;
        reset_potential = reset_potential_;
        adapt_decay = adapt_decay_;
        adapt_coupling = adapt_coupling_;
        surrogate_gradient = std::move(surrogate_grad);
    }

    void reset_state() override
    {
        v_mem = Tensor();         // Clear state — next forward() re-initialises to zeros.
        adapt_a_bptt_ = Tensor(); // Clear adaptation state as well.
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        // Infer Batch Size
        int total_rows = input.rows();
        if (total_rows % time_steps != 0)
        {
            throw std::invalid_argument("LeakyBPTT: Input rows must be divisible by time_steps");
        }
        int batch_size = total_rows / time_steps;
        int features = input.cols();
        const nn::Index batch_size_idx = static_cast<nn::Index>(batch_size);
        const nn::Index features_idx = static_cast<nn::Index>(features);

        // Initialize state if needed
        if (v_mem.rows() != batch_size_idx || v_mem.cols() != features_idx)
        {
            v_mem = Tensor(batch_size, features);
            v_mem.setZero();
        }

        // Prepare Output and History
        Tensor output(total_rows, features);
        if (requires_grad)
        {
            // We cache v_pre (the membrane after decay+input, before spike/reset) for each time.
            // Backward() reads this cache as its "pre-activation" for surrogate gradients.
            v_mem_history = Tensor(total_rows, features);
            v_post_history = Tensor(total_rows, features);
        }

        constexpr float kMinPositiveParam = 1e-6F;
        float const R = std::max(kMinPositiveParam, resistance.at(0, 0));
        float const C = std::max(kMinPositiveParam, capacitance.at(0, 0));
        float const tau = R * C;
        float const beta = std::exp(-time_step / tau);
        float const base_threshold = voltage_threshold.at(0, 0);

        // Spike-frequency adaptation state: shape (B, F), lazy-init like v_mem
        const bool use_adaptation = (adapt_coupling > 0.0F);
        if (use_adaptation &&
            (adapt_a_bptt_.rows() != batch_size_idx || adapt_a_bptt_.cols() != features_idx))
        {
            adapt_a_bptt_ = Tensor(batch_size, features);
            adapt_a_bptt_.setZero();
        }

        // Time Loop
        for (int t = 0; t < time_steps; ++t)
        {
            int offset = t * batch_size;

            // Decay adaptation before this step (adapt_a[t] = decay * adapt_a[t-1])
            if (use_adaptation)
            {
                adapt_a_bptt_.multiply_scalar_inplace(adapt_decay);
            }

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
                        output.at(offset + b, f) = v;
                    }
                    else
                    {
                        // Effective threshold includes adaptation variable
                        float eff_thresh =
                            base_threshold + (use_adaptation ? adapt_a_bptt_.at(b, f) : 0.0F);
                        float s = (v > eff_thresh) ? 1.0f : 0.0f;
                        output.at(offset + b, f) = s;

                        // Reset + update adaptation on spike
                        if (s > 0.5f)
                        {
                            if (reset_zero)
                                v = reset_potential;
                            else
                                v -= base_threshold;

                            if (use_adaptation)
                            {
                                adapt_a_bptt_.at(b, f) += adapt_coupling;
                            }
                        }
                    }

                    v_mem.at(b, f) = v;
                    if (requires_grad)
                    {
                        v_post_history.at(offset + b, f) = v;
                    }
                }
            }
        }

        return output;
    } //

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        // Grad Output: (Time*Batch, Feat)
        int total_rows = grad_output.rows();
        int batch_size = total_rows / time_steps;
        int features = grad_output.cols();

        Tensor grad_input(total_rows, features);
        Tensor grad_next_state(batch_size, features); // dL / dv[t+1] (recurrence accumulator)
        grad_next_state.setZero();

        constexpr float kMinPositiveParam = 1e-6F;
        float const raw_R = resistance.at(0, 0);
        float const raw_C = capacitance.at(0, 0);
        float const R = std::max(kMinPositiveParam, raw_R);
        float const C = std::max(kMinPositiveParam, raw_C);
        float const tau = R * C;
        float const beta = std::exp(-time_step / tau);
        float threshold_val = voltage_threshold.at(0, 0);

        // Accumulators for params
        float dL_dVth_sum = 0.0f;
        float dL_dR_sum = 0.0f;
        float dL_dC_sum = 0.0f;
        float d_beta_dR =
            (tau > 1e-12F && raw_R > kMinPositiveParam) ? (beta * time_step) / (C * R * R) : 0.0f;
        float d_beta_dC =
            (tau > 1e-12F && raw_C > kMinPositiveParam) ? (beta * time_step) / (R * C * C) : 0.0f;
        // Note: d_beta_dR is derived from beta = exp(-time_step/(R*C)).
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

                    if (readout_mode)
                    {
                        // Readout-mode consistency fix: treat dynamics as purely continuous.
                        // No spike/reset path is allowed to leak into this branch.
                        // dL/dV = grad_out, and recurrence Jacobian is beta.
                        grad_v_pre = grad_out + grad_from_next * beta;
                    }
                    else
                    {
                        float dvpost_dvpre = 1.0f;
                        // Surrogate dS/dv
                        float surr = surrogate_gradient->calculate_scalar(v_pre, threshold_val);
                        float spike = (v_pre > threshold_val) ? 1.0F : 0.0F;
                        float dvpost_dVth = 0.0F;

                        // Term 2: From Next State v[t+1] (Recurrent)

                        if (reset_zero)
                        {
                            // Exact local derivative for v_post = v_pre + s*(V_reset - v_pre)
                            dvpost_dvpre = 1.0F - spike;
                            // ds/dVth = -surr
                            dvpost_dVth = surr * (v_pre - reset_potential);
                        }
                        else
                        {
                            // Soft reset: v_post = v_pre - s*Vth
                            dvpost_dvpre = 1.0f - surr * threshold_val;
                            // d(v_pre - s*Vth)/dVth = -s - Vth*ds/dVth
                            dvpost_dVth = -spike + threshold_val * surr;
                        }

                        grad_v_pre = grad_out * surr + grad_from_next * beta * dvpost_dvpre;

                        // dL/dVth includes both direct spike term and recurrent reset-path term.
                        // This replaced the previous simplified accumulator that biased Vth
                        // updates.
                        dL_dVth_sum += (-grad_out * surr) + (grad_from_next * beta * dvpost_dVth);
                    }

                    // Store dL/dI = grad_v_pre * 1
                    grad_input.at(offset + b, f) = grad_v_pre;

                    // Update grad_next_state for t-1
                    grad_next_state.at(b, f) = grad_v_pre;

                    // dL/dbeta = dL/dv_pre * dv_pre/dbeta = dL/dv_pre * v_post[t-1]
                    float v_prev_post = 0.0f;
                    if (t > 0)
                    {
                        v_prev_post = v_post_history.at((t - 1) * batch_size + b, f);
                    }
                    dL_dR_sum += grad_v_pre * v_prev_post * d_beta_dR;
                    dL_dC_sum += grad_v_pre * v_prev_post * d_beta_dC;
                }
            }
        }

        // Set Params Grads
        Tensor vth_grad(1, 1);
        vth_grad.at(0, 0) = dL_dVth_sum;
        voltage_threshold.set_grad(vth_grad);

        Tensor r_grad(1, 1);
        r_grad.at(0, 0) = dL_dR_sum;
        resistance.set_grad(r_grad);

        Tensor c_grad(1, 1);
        c_grad.at(0, 0) = dL_dC_sum;
        capacitance.set_grad(c_grad);

        return grad_input;
    }
};

#endif // LEAKY_BPTT_HPP
