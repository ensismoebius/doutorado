#pragma once
#include "../tensor/Tensor.hpp"
#include <cmath>
#include <random>

/**
 * @brief Kaiming (He) uniform initializer, adapted for spiking neural networks.
 *
 * This initializer is often preferred for SNNs, especially with ReLU or spiking activations,
 * as it helps maintain variance in the forward pass. It samples from a uniform distribution
 * in [-limit, +limit], where limit = sqrt(6 / fan_in).
 *
 * @param in_features  Number of input features (fan-in).
 * @param out_features Number of output features (fan-out).
 * @param weights      Reference to the tensor where the weight matrix will be stored.
 *                     Must be pre-allocated as a 2D tensor of shape (out_features, in_features).
 * @param bias         Reference to the tensor where the bias vector will be stored.
 *                     Must be pre-allocated as a 1D tensor of shape (out_features, 1).
 */
inline auto kaimingSNNInitializer(int in_features, int out_features, Tensor &weights, Tensor &bias) -> void {
    // Kaiming/He uniform limit: sqrt(6 / fan_in)
    float const limit = std::sqrt(6.0F / static_cast<float>(in_features));

    // Uniform distribution in [-limit, +limit]
    std::uniform_real_distribution<float> dist(-limit, limit);
    std::mt19937 gen(static_cast<int>(std::random_device{}()));

    // Initialize weights
    weights.data = Eigen::MatrixXf(out_features, in_features).unaryExpr([&](float) { return dist(gen); });
    // Initialize biases to zero (common in SNNs)
    bias.data = Eigen::VectorXf::Zero(out_features);
}
