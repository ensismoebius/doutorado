#ifndef GAUSSIAN_NOISE_HPP
#define GAUSSIAN_NOISE_HPP

#include <random>
#include <stdexcept>

#include "utility/ITransform.hpp"

namespace nn::transforms
{

// Additive white Gaussian noise, elementwise: out = x + N(0, std^2). The corruption step of
// a denoising autoencoder (Vincent et al. 2008, "Extracting and Composing Robust Features
// with Denoising Autoencoders"; 2010, "Stacked Denoising Autoencoders" -- both cited in
// .wiki/References.md) -- applied to the ENCODER's input while the loss target stays the
// clean, uncorrupted window. See meeting01::train_with_early_stopping_snn / train_ae for
// where this is applied (training input only, never the target, never validation/test).
//
// std == 0.0 is the "denoising disabled" state (Meeting01Config::Model::denoising_noise_std
// defaults to 0), not an error: operator() short-circuits to an exact pass-through rather
// than drawing from std::normal_distribution(0, 0), whose behaviour at stddev == 0 is a
// standard-library precondition violation (UB), not a guaranteed no-op.
//
// The instance owns its RNG and advances it on every call (never re-seeded per call) --
// same stateful-callable semantics as RandomCrop/RandomIndexCrop.
class GaussianNoise final : public ITransform
{
    float std_;
    mutable std::mt19937 rng_;

   public:
    GaussianNoise(float std, unsigned int seed) : std_(std), rng_(seed)
    {
        if (std_ < 0.0F) throw std::invalid_argument("GaussianNoise: std must be >= 0");
    }

    auto operator()(const nn::Tensor& x) const -> nn::Tensor override
    {
        if (std_ == 0.0F) return x;

        nn::Tensor out = x;
        std::normal_distribution<float> dist(0.0F, std_);
        for (nn::Index i = 0; i < out.size(); ++i) out.at(i) = out.at(i) + dist(rng_);
        return out;
    }
};

} // namespace nn::transforms

#endif // GAUSSIAN_NOISE_HPP
