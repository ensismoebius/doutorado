#include <random>

#include "../tensor/Tensor.hpp"

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
inline auto xavierInitializer(int in_features, int out_features, nn::Tensor& weights, nn::Tensor& bias)
    -> void
{
    // Xavier limit: sqrt(6 / (in_features + out_features))
    float const limit = std::sqrt(6.0F / static_cast<float>(in_features + out_features));

    // Uniform distribution in [-limit, +limit]
    std::uniform_real_distribution dist(-limit, limit);

    // Mersenne Twister PRNG seeded with a random device
    std::mt19937 gen(std::random_device{}());

    // Initialize weights with random values from the uniform distribution
    weights.set_data(
        Eigen::MatrixXf(out_features, in_features).unaryExpr([&](float) { return dist(gen); }));

    // Initialize biases with random values from the same distribution
    bias.set_data(Eigen::VectorXf(out_features).unaryExpr([&](float) { return dist(gen); }));
}
