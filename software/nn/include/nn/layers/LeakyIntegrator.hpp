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
struct LeakyIntegrator : public Leaky
{
    /**
     * @brief Construct a new Leaky Integrator
     * @param dt_ Time step
     * @param R_ Resistance
     * @param C_ Capacitance
     */
    LeakyIntegrator(float dt_, float R_, float C_) : Leaky(dt_, R_, C_, 0.0f, false, 0.0f) {}

    /**
     * @brief Continuous forward pass (Integrate without Fire/Reset)
     */
    auto forward(const nn::Tensor& input, bool requires_grad = true) -> nn::Tensor override
    {
        // 1. Resize/Init State (if batch size changed or first run)
        if (v_mem.rows() != input.rows() || v_mem.cols() != input.cols()) [[unlikely]]
        {
            v_mem = nn::Tensor(input.rows(), input.cols());
            v_mem.setZero();
        }

        // 2. Calculate Decay Factor (beta)
        // beta = exp(-dt / RC)
        float const tau = resistance.at(0, 0) * capacitance;
        float const beta = std::exp(-dt / tau);

        // Note: if tau is extremely small, beta can underflow; callers should
        // keep (R,C,dt) in a numerically sensible range.

        // 3. Cache state for gradient calculation (dL/dR depends on previous voltage)
        if (requires_grad)
        {
            v_mem_t_minus_1 = v_mem;
        }

        // 4. Update Dynamics: V[t] = beta * V[t-1] + Input[t]
        v_mem = v_mem.multiply_scalar(beta).add(input);

        // Output is the membrane potential itself (Continuous)
        return v_mem;
    }

    /**
     * @brief Backward pass for continuous integration.
     * @param grad_output Gradient of loss w.r.t the output (which is v_mem)
     */
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor override
    {
        // dL/dOutput = grad_output
        // Output = V_mem
        // So this is effectively dL/dV_mem

        // 1. Pass-through Gradient w.r.t Input
        // V[t] = beta*V[t-1] + Input[t]
        // dV[t]/dInput[t] = 1
        // Thus, dL/dInput = dL/dV * 1
        nn::Tensor grad_input = grad_output;

        // 2. Gradient w.r.t Resistance (Parameter Update)
        // dL/dR = dL/dV * dV/dbeta * dbeta/dR
        // dV/dbeta = V[t-1]
        const float R = resistance.at(0, 0);
        const float C = capacitance;
        const float tau = R * C;

        if (tau > 1e-6) [[likely]]
        {
            const float beta = std::exp(-dt / tau);
            const float d_beta_dR = (beta * dt) / (C * R * R);

            // dL/dbeta = sum( dL/dV * V[t-1] )
            float dL_dbeta = grad_input.multiply(v_mem_t_minus_1).sum();

            nn::Tensor r_grad(1, 1);
            r_grad.at(0, 0) = dL_dbeta * d_beta_dR;
            resistance.set_grad(r_grad);
        }
        else
        {
            // Degenerate tau (R or C near zero): treat d(beta)/dR as 0 and keep
            // resistance gradient well-defined.
            nn::Tensor r_grad(1, 1);
            r_grad.set_zero();
            resistance.set_grad(r_grad);
        }

        return grad_input;
    }
};

#endif // LEAKY_INTEGRATOR_HPP
