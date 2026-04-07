#pragma once

#include <cmath>
#include <functional>
#include <optional>
#include <random>
#include <string>

#include "nn/tensor/Tensor.hpp"

/**
 * @file xavier.hpp
 * @brief Xavier/Glorot weight initialization helper.
 *
 * Notes:
 * - This is a standalone function (not a Module) that writes directly into
 *   provided `weights` and `bias` tensors.
 * - It uses `std::random_device` seeding, so results are non-deterministic
 *   unless callers provide a deterministic seeding strategy elsewhere.
 * - On dimension mismatch, it returns early without throwing.
 */

/**
 * @brief Initializes weights and biases using the Xavier (Glorot) uniform initialization.
 *
 * This initializer sets the weights and biases using values drawn from a uniform distribution
 * within [-limit, +limit], where limit = sqrt(6 / (in_features + out_features)).
 * It is designed to keep the variance of activations approximately constant across layers,
 * which helps with stable and efficient training of neural networks.
 *
 * @param in_features  Number of input features (fan-in).
 * @param out_features Number of output features (fan-out).
 * @param weights      Reference to the tensor where the weight matrix will be stored.
 *                     Must be pre-allocated as a 2D tensor of shape (out_features, in_features).
 * @param bias         Reference to the tensor where the bias vector will be stored.
 *                     Must be pre-allocated as a 1D tensor of shape (out_features).
 */
inline auto xavierInitializer(int in_features,
    int out_features,
    nn::Tensor& weights,
    nn::Tensor& bias,
    std::optional<unsigned int> seed = std::nullopt,
    const std::string& sampler_default_type = "") -> void
{
    // Handle zero dimensions gracefully
    if (in_features == 0 || out_features == 0)
    {
        return;
    }

    // Check dimensions match expected shape
    if (weights.rows() != static_cast<nn::Index>(out_features) ||
        weights.cols() != static_cast<nn::Index>(in_features))
    {
        return;
    }
    if (bias.rows() != static_cast<nn::Index>(out_features) ||
        bias.cols() != static_cast<nn::Index>(1))
    {
        return;
    }

    // Xavier limit: sqrt(6 / (in_features + out_features))
    float const limit = std::sqrt(6.0F / static_cast<float>(in_features + out_features));

    // Uniform distribution in [-limit, +limit]
    std::uniform_real_distribution dist(-limit, limit);

    // Seed policy: when a seed is provided, fold sampler_default_type into the final seed
    // so runs are deterministic yet sampler-aware. Without a seed, use random_device.
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

    // Use Tensor helpers instead of manual element-wise loops.
    weights = nn::Tensor::rand(               //
        static_cast<nn::Index>(out_features), //
        static_cast<nn::Index>(in_features),  //
        gen                                   //
        )
                  .multiply_scalar(2.0F * limit)
                  .add_scalar(-limit);
    bias.fill(0.0F);
}
