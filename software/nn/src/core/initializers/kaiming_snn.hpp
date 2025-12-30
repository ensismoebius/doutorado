#pragma once
#include <cmath>
#include <memory>
#include <random>

#include "core/tensor/Tensor.hpp"
#include "core/layers/Linear.hpp"

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
    layer->weight.set_data(Eigen::MatrixXf(layer->out_features, layer->in_features)
                             .unaryExpr([&](float) { return dist(gen); }));

    // Initialize biases to zero (common in SNNs)
    layer->bias.set_data(Eigen::VectorXf::Zero(layer->out_features));
}
