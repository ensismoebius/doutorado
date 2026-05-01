/**
 * @file initializers_gtest.cpp
 * @brief Unit tests for weight initialization helpers.
 */

#include <gtest/gtest.h>

#include "nn/initializers/kaiming_snn.hpp"
#include "nn/initializers/xavier.hpp"
#include "nn/layers/Layers.hpp"
#include "nn/tensor/Tensor.hpp"

using nn::Linear;

// Initializer: kaiming_snn
TEST(InitializerTest, KaimingSNN)
{
    auto layer = std::make_shared<Linear>(2, 4);
    kaimingSNNInitializer(layer, 42U);
    ASSERT_NE(layer->weight.sum(), 0.0F);
    ASSERT_NEAR(layer->bias.sum(), 0.0F, 1e-6F);
}

// Initializer: Xavier
TEST(InitializerTest, Xavier)
{
    nn::Tensor weights(4, 2);
    nn::Tensor bias(4, 1);
    xavierInitializer(2, 4, weights, bias);
    ASSERT_NE(weights.sum(), 0.0F);
    ASSERT_NEAR(bias.sum(), 0.0F, 1e-6F);
}

TEST(InitializerTest, XavierZeroDimensions)
{
    nn::Tensor weights(0, 0);
    nn::Tensor bias(0, 0);
    // Expect no throw, but initializers should handle this gracefully
    ASSERT_NO_THROW(xavierInitializer(0, 0, weights, bias));
    ASSERT_EQ(weights.rows(), 0);
    ASSERT_EQ(weights.cols(), 0);
    ASSERT_EQ(bias.rows(), 0);
    ASSERT_EQ(bias.cols(), 0);

    // Test with valid fan_in/fan_out but 0-dimension tensors initially
    nn::Tensor weights_empty(0, 2);
    nn::Tensor bias_empty(0, 1);
    ASSERT_NO_THROW(xavierInitializer(2, 4, weights_empty, bias_empty));
    ASSERT_EQ(weights_empty.rows(), 0);
    ASSERT_EQ(weights_empty.cols(), 2);
    ASSERT_EQ(bias_empty.rows(), 0);
    ASSERT_EQ(bias_empty.cols(), 1);
}