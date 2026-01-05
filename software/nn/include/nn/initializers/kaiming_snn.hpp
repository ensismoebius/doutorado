#pragma once
#include <cmath>
#include <memory>
#include <random>

#include "nn/layers/Linear.hpp"
#include "nn/tensor/Tensor.hpp"

/**
 * @brief Kaiming (He) uniform initializer, adapted for spiking neural networks.
 *
 * This initializer is often preferred for SNNs, especially with ReLU or spiking activations,
 * as it helps maintain variance in the forward pass. It samples from a uniform distribution
 * in [-limit, +limit], where limit = sqrt(6 / fan_in).
 *
 * @param layer The linear layer to initialize.
 */
inline auto kaimingSNNInitializer(const std::shared_ptr<Linear>& layer) -> void
{
    // Kaiming/He uniform limit: sqrt(6 / fan_in)
    float const limit = std::sqrt(6.0F / static_cast<float>(layer->in_features));

    // Uniform distribution in [-limit, +limit]
    std::uniform_real_distribution<float> dist(-limit, limit);
    std::mt19937 gen(static_cast<int>(std::random_device{}()));

    // Initialize weights
    for (int i = 0; i < layer->out_features; ++i)
    {
        for (int j = 0; j < layer->in_features; ++j)
        {
            layer->weight.at(i, j) = dist(gen);
        }
    }

    // Initialize biases to zero (common in SNNs)
    for (int i = 0; i < layer->out_features; ++i)
    {
        layer->bias.at(i, 0) = 0.0F;
    }
}
