#include <random>

#include "nn/tensor/Tensor.hpp"

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
inline auto xavierInitializer(int in_features, int out_features, nn::Tensor& weights,
                              nn::Tensor& bias) -> void
{
    // Handle zero dimensions gracefully
    if (in_features == 0 || out_features == 0)
    {
        return;
    }

    // Check dimensions match expected shape
    if (weights.rows() != out_features || weights.cols() != in_features)
    {
        return;
    }
    if (bias.rows() != out_features || bias.cols() != 1)
    {
        return;
    }

    // Xavier limit: sqrt(6 / (in_features + out_features))
    float const limit = std::sqrt(6.0F / static_cast<float>(in_features + out_features));

    // Uniform distribution in [-limit, +limit]
    std::uniform_real_distribution dist(-limit, limit);

    // Mersenne Twister PRNG seeded with a random device
    std::mt19937 gen(std::random_device{}());

    // Initialize weights with random values from the uniform distribution
    for (int i = 0; i < out_features; ++i)
    {
        for (int j = 0; j < in_features; ++j)
        {
            weights(i, j) = dist(gen);
        }
    }

    // Initialize biases with random values from the same distribution
    for (int i = 0; i < out_features; ++i)
    {
        bias(i, 0) = dist(gen);
    }
}
