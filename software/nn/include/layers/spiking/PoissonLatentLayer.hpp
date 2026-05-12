#ifndef NN_LAYERS_SPIKING_POISSONLATENTLAYER_HPP
#define NN_LAYERS_SPIKING_POISSONLATENTLAYER_HPP

#include <algorithm>
#include <cmath>
#include <random>

#include "layers/activations/Sigmoid.hpp"
#include "layers/base/Module.hpp"
#include "tensor/Tensor.hpp"

/**
 * @file PoissonLatentLayer.hpp
 * @brief Poisson-reparameterizable latent layer for Spiking Variational Autoencoders (SVAE).
 *
 * This layer implements the reparameterizable Poisson latent space described
 * in ESVAE (arXiv:2310.14839, 2024). It replaces the standard Gaussian VAE
 * reparameterization trick with one appropriate for discrete spike counts.
 *
 * **Forward pass** (training):
 *   1. Encoder outputs logits z (B × F).
 *   2. Rate λ = softplus(z) so λ > 0 always.
 *   3. Spike count s ~ Poisson(λ * T) sampled per (b, f).
 *   4. Spike train output = s / T  (continuous relaxation, ∈ [0,∞) ).
 *
 * **Forward pass** (inference, requires_grad=false):
 *   Output = λ (mean of the Poisson distribution, no sampling).
 *
 * **Backward pass** (straight-through / REINFORCE estimator):
 *   Gradient flows through the continuous rate λ as if sampling were deterministic.
 *   This is the straight-through estimator used in ESVAE and FSVAE:
 *     dL/dz ≈ dL/d_output * d_output/dλ * dλ/dz
 *            = dL/d_output * (1/T) * sigmoid(z)
 *
 * **KL divergence term** (for VAE training):
 *   KL(Poisson(λ) || Poisson(λ₀)) ≈ λ - λ₀ - λ*log(λ/λ₀)
 *   where λ₀ is the prior rate (default 0.1). Access via kl_loss() after forward().
 *   Add kl_loss * beta_kl to your total loss (β-SVAE training).
 *
 * **Usage (β-SVAE training loop):**
 * @code
 * PoissonLatentLayerImpl<Backend> latent(features, T);
 * Tensor s = latent.forward(encoder_out, true);      // reparameterized sample
 * Tensor recon = decoder.forward(s, true);
 * float recon_loss = mse_loss(recon, target);
 * float total_loss = recon_loss + beta * latent.kl_loss();
 * @endcode
 *
 * References:
 * [29] K. Kamata et al., "Fully spiking variational autoencoder," AAAI 2022.
 * [30] C. Chen et al., "ESVAE," arXiv:2310.14839, 2024.
 */
template <typename Backend>
class PoissonLatentLayerImpl : public Module<Backend>
{
    using Tensor = typename Module<Backend>::Tensor;

   public:
    int time_steps = 1;      ///< T: number of spike time steps to simulate
    float prior_rate = 0.1F; ///< λ₀: prior Poisson rate for KL divergence
    float beta_kl = 1.0F;    ///< β weighting for KL term (β-SVAE; 0 = plain AE)

    explicit PoissonLatentLayerImpl(
        int T = 1, float prior_rate_val = 0.1F, float beta_kl_val = 1.0F)
        : time_steps(T), prior_rate(prior_rate_val), beta_kl(beta_kl_val)
    {
        rng_.seed(42U);
    }

    /// KL divergence accumulated in the last forward() call. Add to total loss.
    float kl_loss() const
    {
        return kl_loss_;
    }

    /// Mean rates λ from last forward (useful for logging spike sparsity).
    const Tensor& last_rates() const
    {
        return rate_cache_;
    }

    auto forward(const Tensor& input, bool requires_grad = true) -> Tensor override
    {
        const size_t B = input.rows();
        const size_t F = input.cols();

        // 1. Rate: λ = softplus(z) = log(1 + exp(z)), ensures λ > 0
        Tensor rate(B, F);
        for (size_t b = 0; b < B; ++b)
        {
            for (size_t f = 0; f < F; ++f)
            {
                float z = input.at(b, f);
                // Numerically stable softplus
                rate.at(b, f) = (z > 20.0f) ? z : std::log1p(std::exp(z));
            }
        }

        if (requires_grad)
        {
            input_cache_ = input;
            rate_cache_ = rate;
        }

        // 2. Compute KL: KL(Poisson(λ) || Poisson(λ₀))
        //    = λ - λ₀ - λ*log(λ/λ₀)  (summed over B×F, then normalised)
        kl_loss_ = 0.0f;
        if (beta_kl > 0.0f)
        {
            for (size_t b = 0; b < B; ++b)
            {
                for (size_t f = 0; f < F; ++f)
                {
                    float lam = rate.at(b, f);
                    float kl = prior_rate - lam + lam * std::log(lam / prior_rate + 1e-8f);

                    kl_loss_ += kl;
                }
            }
            kl_loss_ = beta_kl * kl_loss_ / static_cast<float>(B * F);
        }

        // 3. Sample or return mean
        if (!requires_grad)
        {
            // Inference: return the mean rate directly (no stochasticity)
            return rate;
        }

        // Training: sample s ~ Poisson(λ * T), then return s/T
        Tensor output(B, F);
        const float T_f = static_cast<float>(time_steps);
        for (size_t b = 0; b < B; ++b)
        {
            for (size_t f = 0; f < F; ++f)
            {
                float lam = rate.at(b, f) * T_f;
                std::poisson_distribution<int> poisson(static_cast<double>(lam));
                float s = static_cast<float>(poisson(rng_));
                output.at(b, f) = s / T_f;
            }
        }
        return output;
    }

    auto backward(const Tensor& grad_output) -> Tensor override
    {
        const size_t B = grad_output.rows();
        const size_t F = grad_output.cols();

        // Straight-through estimator:
        // dL/dz = dL/d_output * (1/T) * sigmoid(z)
        // where sigmoid(z) = dλ/dz from softplus.
        Tensor grad_input(B, F);
        const float inv_T = 1.0f / static_cast<float>(time_steps);

        for (size_t b = 0; b < B; ++b)
        {
            for (size_t f = 0; f < F; ++f)
            {
                float z = input_cache_.at(b, f);
                float sigmoid_z = nn::activation::sigmoid(z); // d(softplus)/dz
                grad_input.at(b, f) = grad_output.at(b, f) * inv_T * sigmoid_z;
            }
        }

        // KL gradient: dKL/dz = (1 - λ₀/λ) * sigmoid(z) * beta_kl / (B*F)
        if (beta_kl > 0.0f)
        {
            const float scale = beta_kl / static_cast<float>(B * F);
            for (size_t b = 0; b < B; ++b)
            {
                for (size_t f = 0; f < F; ++f)
                {
                    float lam = rate_cache_.at(b, f);
                    float z = input_cache_.at(b, f);
                    float sigmoid_z = nn::activation::sigmoid(z);
                    float d_kl = (1.0f - prior_rate_ / (lam + 1e-8f)) * sigmoid_z * scale;
                    grad_input.at(b, f) += d_kl;
                }
            }
        }

        return grad_input;
    }

   private:
    Tensor input_cache_;
    Tensor rate_cache_;
    float kl_loss_ = 0.0f;
    float prior_rate_ = 0.1F; // internal copy to avoid aliasing with public field
    mutable std::mt19937 rng_;
};

#endif // NN_LAYERS_SPIKING_POISSONLATENTLAYER_HPP
