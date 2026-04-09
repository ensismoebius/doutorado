#ifndef LEAKY_INTEGRATOR_HPP
#define LEAKY_INTEGRATOR_HPP

#include <cmath>

#include "nn/layers/Leaky.hpp"

/**
 * @file LeakyIntegrator.hpp
 * @brief Continuous (non-spiking) leaky integrator readout.
 *
 * This class derives from `Leaky` but overrides the spiking behavior:
 * it integrates with decay and returns the membrane potential directly.
 *
 * Practical use:
 * - Often used as a readout layer after spiking layers when you want continuous
 *   values (e.g., reconstruction, regression) instead of spike events.
 * - In this codebase, it is conceptually similar to using `LeakyBPTT` in
 *   `readout_mode=true`, but without explicit time-unrolling.
 */

/**
 * @brief Leaky Integrator (Readout) Layer.
 *
 * A non-spiking neuron model that accumulates input current into membrane potential
 * with decay, but never fires or resets.
 *
 * Dynamics:
 *   V[t] = beta * V[t-1] + input[t]
 *
 * Output:
 *   y[t] = V[t] (Continuous membrane potential)
 *
 * Gradient behavior:
 *   Since there is no non-differentiable threshold function, the gradient passes
 *   through directly (dV/dInput = 1). The recurrent dependency dV[t]/dV[t-1] = beta
 *   is typically captured via BPTT (or truncated BPTT approximations).
 */
template <typename Backend>
struct LeakyIntegratorImpl : public LeakyImpl<Backend>
{
    /// Tensor type for the active compute backend.
    using Tensor = typename LeakyImpl<Backend>::Tensor;

    /**
     * @brief Construct a new Leaky Integrator
     * @param dt_ Time step
     * @param R_ Resistance
     * @param C_ Capacitance
     */
    LeakyIntegratorImpl(float dt_, float R_, float C_)
        : LeakyImpl<Backend>(dt_, R_, C_, 0.0f, false, 0.0f)
    {
    }

    /**
     * @brief Continuous forward pass (Integrate without Fire/Reset)
     */
    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        // 1. Resize/Init State (if batch size changed or first run)
        if (this->v_mem.rows() != input.rows() || this->v_mem.cols() != input.cols()) [[unlikely]]
        {
            this->v_mem = Tensor(input.rows(), input.cols());
            this->v_mem.setZero();
        }

        // 2. Calculate Decay Factor (beta)
        // beta = exp(-dt / RC)
        // Clamp capacitance to prevent tau = 0.
        float const C = std::max(1e-6F, this->capacitance.at(0, 0));
        float const tau = this->resistance.at(0, 0) * C;
        float const beta = std::exp(-this->time_step / tau);

        // Note: if tau is extremely small, beta can underflow; callers should
        // keep (R,C,dt) in a numerically sensible range.

        // 3. Cache state for gradient calculation (dL/dR depends on previous voltage)
        if (requires_grad)
        {
            this->v_mem_t_minus_1 = this->v_mem;
        }

        // 4. Update Dynamics: V[t] = beta * V[t-1] + Input[t]
        this->v_mem = this->v_mem.multiply_scalar(beta).add(input);

        // Output is the membrane potential itself (Continuous)
        return this->v_mem;
    }

    /**
     * @brief Backward pass for continuous integration.
     * @param grad_output Gradient of loss w.r.t the output (which is v_mem)
     */
    auto backward(const Tensor& grad_output) -> Tensor override
    {
        // dL/dOutput = grad_output
        // Output = V_mem
        // So this is effectively dL/dV_mem

        // 1. Pass-through Gradient w.r.t Input
        // V[t] = beta*V[t-1] + Input[t]
        // dV[t]/dInput[t] = 1
        // Thus, dL/dInput = dL/dV * 1
        Tensor grad_input = grad_output;

        // 2. Gradient w.r.t Resistance (Parameter Update)
        // dL/dR = dL/dV * dV/dbeta * dbeta/dR
        // dV/dbeta = V[t-1]
        const float R = this->resistance.at(0, 0);
        const float C = std::max(1e-6F, this->capacitance.at(0, 0));
        const float tau = R * C;

        if (tau > 1e-6) [[likely]]
        {
            const float beta = std::exp(-this->time_step / tau);
            const float d_beta_dR = (beta * this->time_step) / (C * R * R);

            // dL/dbeta = sum( dL/dV * V[t-1] )
            float dL_dbeta = grad_input.multiply(this->v_mem_t_minus_1).sum();

            // --- Gradient for resistance ---
            Tensor r_grad(1, 1);
            r_grad.at(0, 0) = dL_dbeta * d_beta_dR;
            this->resistance.set_grad(nn::Tensor(r_grad));

            // --- Gradient for capacitance (symmetric to dL/dR) ---
            // dBeta/dC = beta * dt / (R * C^2)
            const float d_beta_dC = (beta * this->time_step) / (R * C * C);
            Tensor c_grad(1, 1);
            c_grad.at(0, 0) = dL_dbeta * d_beta_dC;
            this->capacitance.set_grad(nn::Tensor(c_grad));
        }
        else
        {
            // Degenerate tau (R or C near zero): treat d(beta)/dR as 0 and keep
            // resistance gradient well-defined.
            Tensor r_grad(1, 1);
            r_grad.set_zero();
            this->resistance.set_grad(nn::Tensor(r_grad));
            Tensor c_grad(1, 1);
            c_grad.set_zero();
            this->capacitance.set_grad(nn::Tensor(c_grad));
        }

        return grad_input;
    }
};

#endif // LEAKY_INTEGRATOR_HPP
