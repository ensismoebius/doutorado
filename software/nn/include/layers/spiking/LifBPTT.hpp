#ifndef LIF_BPTT_HPP
#define LIF_BPTT_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

#include "layers/base/Module.hpp"
#include "layers/spiking/ExponentialSurrogate.hpp"
#include "tensor/Tensor.hpp"

/**
 * @file LifBPTT.hpp
 * @brief LIF/Lif Integrate-and-Fire dynamics with a full (explicit) BPTT-style backward.
 *
 * Design intent (snnTorch mental model):
 * - This is similar to snnTorch's leaky neuron module that is unrolled over time.
 * - Equivalence is EXACT for reset_zero=true (the default, and the only mode production
 *   uses); verified against snn.Leaky under a hard drive by micro_network_parity_gtest.
 *   For reset_zero=false (subtract) it is NOT exact: we apply the reset immediately, so it
 *   is decayed on the next step, whereas snnTorch subtracts it un-decayed one step later
 *   (our reset term ends up multiplied by beta). Measured disagreement: ~2-3% of spikes.
 *   Ours is the textbook soft-reset LIF; snnTorch's is snnTorch's convention — neither is
 *   wrong, but they are not the same neuron, so do not assume snn.Leaky parity in subtract.
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
 * @brief Lif Integrate-and-Fire (LIF) layer with Full Backpropagation Through Time (BPTT).
 *
 * This module expects a flattened time-series input of shape (Time * Batch, Features).
 * It unrolls the simulation over `time_steps`, maintaining state across the sequence,
 * and computes accurate gradients through time during the backward pass.
 */
template <typename Backend>
struct LifBPTTImpl : public Module<Backend>
{
   public:
    /// Tensor type for the active compute backend.
    using Tensor = typename Module<Backend>::Tensor;

    /// @brief Simulation step SIZE (delta-t): how long one step lasts.
    ///        Not to be confused with `time_steps` (how MANY steps).
    float delta_t = 1.0F;

    // IDENTIFIABILITY NOTE (audit m-2): dynamics depend only on the membrane time
    // constant tau = R * C (beta = exp(-dt/tau)). R and C are not separately
    // identifiable — only their product matters — so training both is a redundant
    // degree of freedom. Kept as two tensors for config/serialization backward
    // compatibility. Treat tau = R*C as the single effective time constant.

    /// @brief Membrane resistance (R). With C forms tau = R*C (the identifiable quantity).
    Tensor resistance = Tensor::constant(1, 1, 1.0F);

    /// @brief Membrane capacitance (C). With R forms tau = R*C; redundant alone (see note).
    Tensor capacitance = Tensor::constant(1, 1, 1.0F);

    /// @brief Voltage threshold.
    Tensor voltage_threshold = Tensor::constant(1, 1, 1.0F);

    // State management
    Tensor v_mem;         ///< Persistent state after last processed time step (shape: B x F)
    Tensor v_mem_history; ///< Cached pre-reset membrane values for BPTT (shape: (T*B) x F)
    // Added to avoid backward-time reconstruction drift: gradients for beta (and thus R/C)
    // must use the exact previous post-reset state seen in forward.
    Tensor v_post_history; ///< Cached post-reset membrane values for exact recurrence derivatives.
    Tensor adapt_a_bptt_;  ///< Adaptation variable state (shape: B x F), persists across calls.

    // Configuration
    int time_steps; ///< HOW MANY steps one sample spans. Splits a (T*B, F) tensor:
                    ///< batch_size = rows / time_steps. See .wiki/Concepts/Time-Steps.md
    bool reset_zero = true;
    bool readout_mode =
        false; ///< If true, outputs membrane potential instead of spikes (Regression)
    float reset_potential = 0.0F;

    // Spike-frequency adaptation (same semantics as LifImpl)
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

    auto state_dict() const -> std::map<std::string, Tensor> override
    {
        std::map<std::string, Tensor> d;
        d["resistance"] = resistance;
        d["capacitance"] = capacitance;
        d["voltage_threshold"] = voltage_threshold;
        return d;
    }

    void load_state_dict(const std::map<std::string, Tensor>& sd) override
    {
        auto it = sd.find("resistance");
        if (it != sd.end()) resistance = it->second;
        it = sd.find("capacitance");
        if (it != sd.end()) capacitance = it->second;
        it = sd.find("voltage_threshold");
        if (it != sd.end()) voltage_threshold = it->second;
    }

    explicit LifBPTTImpl(int time_steps_,
        float delta_t_ = 1.0F,
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
        : delta_t(delta_t_), time_steps(time_steps_), readout_mode(readout_mode_)
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
            throw std::invalid_argument("LifBPTT: Input rows must be divisible by time_steps");
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
        float const beta = std::exp(-delta_t / tau);
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

        const BackwardParams bp = compute_backward_params();

        // Accumulators for params
        float dL_dVth_sum = 0.0f;
        float dL_dR_sum = 0.0f;
        float dL_dC_sum = 0.0f;

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
                    const GradStep step = compute_grad_step(
                        v_pre, grad_out, grad_from_next, bp.beta, bp.threshold_val);
                    float grad_v_pre = step.grad_v_pre;
                    dL_dVth_sum += step.dVth_contrib;

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
                    dL_dR_sum += grad_v_pre * v_prev_post * bp.d_beta_dR;
                    dL_dC_sum += grad_v_pre * v_prev_post * bp.d_beta_dC;
                }
            }
        }

        store_param_grads(dL_dVth_sum, dL_dR_sum, dL_dC_sum);

        return grad_input;
    }

   private:
    /// @brief Per-call constants shared by every (t, b, f) cell of the BPTT loop.
    struct BackwardParams
    {
        float beta;
        float threshold_val;
        float d_beta_dR;
        float d_beta_dC;
    };

    // Computes beta = exp(-dt/tau) and its partial derivatives w.r.t. R and C, applying the
    // same kMinPositiveParam clamp as forward() so gradients stay consistent with the clamped
    // forward value (see "Convergence fixes" notes 1/2 in the file header).
    [[nodiscard]] auto compute_backward_params() const -> BackwardParams
    {
        constexpr float kMinPositiveParam = 1e-6F;
        float const raw_R = resistance.at(0, 0);
        float const raw_C = capacitance.at(0, 0);
        float const R = std::max(kMinPositiveParam, raw_R);
        float const C = std::max(kMinPositiveParam, raw_C);
        float const tau = R * C;
        float const beta = std::exp(-delta_t / tau);
        float const d_beta_dR =
            (tau > 1e-12F && raw_R > kMinPositiveParam) ? (beta * delta_t) / (C * R * R) : 0.0f;
        float const d_beta_dC =
            (tau > 1e-12F && raw_C > kMinPositiveParam) ? (beta * delta_t) / (R * C * C) : 0.0f;
        // Note: d_beta_dR is derived from beta = exp(-delta_t/(R*C)).
        // This implementation uses a scalar R and C shared across all neurons.
        return BackwardParams{beta, voltage_threshold.at(0, 0), d_beta_dR, d_beta_dC};
    }

    /// @brief Result of one (t, b, f) BPTT cell: the propagated pre-activation gradient plus
    ///        this cell's contribution to the threshold gradient accumulator.
    struct GradStep
    {
        float grad_v_pre;
        float dVth_contrib;
    };

    // Per-element gradient step for one (t, b, f) cell of the reverse-time loop. Isolates the
    // readout-mode / hard-reset / soft-reset branches (see "Convergence fixes" notes 4/5 in the
    // file header) so backward() itself stays a plain loop skeleton.
    [[nodiscard]] auto compute_grad_step(
        float v_pre, float grad_out, float grad_from_next, float beta, float threshold_val) const
        -> GradStep
    {
        if (readout_mode)
        {
            // Readout-mode consistency fix: treat dynamics as purely continuous.
            // No spike/reset path is allowed to leak into this branch.
            // dL/dV = grad_out, and recurrence Jacobian is beta.
            return GradStep{grad_out + grad_from_next * beta, 0.0F};
        }

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

        float grad_v_pre = grad_out * surr + grad_from_next * beta * dvpost_dvpre;

        // dL/dVth includes both direct spike term and recurrent reset-path term.
        // This replaced the previous simplified accumulator that biased Vth updates.
        float dVth_contrib = (-grad_out * surr) + (grad_from_next * beta * dvpost_dVth);
        return GradStep{grad_v_pre, dVth_contrib};
    }

    // Writes the accumulated scalar gradients back onto the (1x1) parameter tensors.
    void store_param_grads(float dL_dVth_sum, float dL_dR_sum, float dL_dC_sum)
    {
        Tensor vth_grad(1, 1);
        vth_grad.at(0, 0) = dL_dVth_sum;
        voltage_threshold.set_grad(vth_grad);

        Tensor r_grad(1, 1);
        r_grad.at(0, 0) = dL_dR_sum;
        resistance.set_grad(r_grad);

        Tensor c_grad(1, 1);
        c_grad.at(0, 0) = dL_dC_sum;
        capacitance.set_grad(c_grad);
    }
};

#endif // LIF_BPTT_HPP
