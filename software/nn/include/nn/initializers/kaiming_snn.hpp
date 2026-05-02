/**
 * @file kaiming_snn.hpp
 * @brief Kaiming/He-style initializer used by several demos and small models.
 *
 * Note: the default implementation seeds a local RNG from `std::random_device`, so
 * it is non-deterministic unless you refactor to pass a seeded generator.
 */

#ifndef NN_INITIALIZERS_KAIMING_SNN_HPP
#define NN_INITIALIZERS_KAIMING_SNN_HPP
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>

#include "nn/layers/dense/Linear.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @brief Kaiming (He) uniform initializer, adapted for spiking neural networks.
 *
 * This initializer is often preferred for SNNs, especially with ReLU or spiking activations,
 * as it helps maintain variance in the forward pass. It samples from a uniform distribution
 * in [-limit, +limit], where limit = sqrt(6 / fan_in).
 *
 * Practical note for deep SNN stacks:
 * - Even with Kaiming init, deeply stacked Linear→Spiking blocks can still be numerically
 *   “hot” early in training (large currents/membrane). Some demos additionally scale weights
 *   down after initialization to keep the initial firing rates reasonable.
 *
 * Reproducibility note:
 * - This helper seeds its own RNG from `std::random_device`, so it is non-deterministic across
 * runs. For deterministic experiments, seed `gen` from a fixed value (or thread a seed through).
 *
 * @param layer The linear layer to initialize.
 */
template <typename Backend>
inline auto kaimingSNNInitializer(const std::shared_ptr<LinearImpl<Backend>>& layer,
    std::optional<unsigned int> seed = std::nullopt,
    const std::string& sampler_default_type = "") -> void
{
    using Tensor = typename LinearImpl<Backend>::Tensor;

    // Kaiming/He uniform limit: sqrt(6 / fan_in)
    float const limit = std::sqrt(6.0F / static_cast<float>(layer->in_features));

    // Uniform distribution in [-limit, +limit]
    std::mt19937 gen;
    if (seed.has_value())
    {
        const unsigned int sampler_hash =
            static_cast<unsigned int>(std::hash<std::string>{}(sampler_default_type));
        const unsigned int mixed_seed =
            *seed ^ (sampler_hash + 0x9e3779b9U + (*seed << 6U) + (*seed >> 2U));
        gen.seed(mixed_seed);
    }
    else
    {
        gen.seed(static_cast<unsigned int>(std::random_device{}()));
    }

    layer->weight = Tensor::rand(static_cast<nn::Index>(layer->out_features),
        static_cast<nn::Index>(layer->in_features),
        gen)
                        .multiply_scalar(2.0F * limit)
                        .add_scalar(-limit);
    layer->bias.fill(0.0F);
}

#endif // NN_INITIALIZERS_KAIMING_SNN_HPP
