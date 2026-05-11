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

TEST(InitializerTest, XavierSeededSamplerMixingIsDeterministic)
{
    nn::Tensor w1(4, 2), b1(4, 1);
    nn::Tensor w2(4, 2), b2(4, 1);
    nn::Tensor w3(4, 2), b3(4, 1);

    xavierInitializer(2, 4, w1, b1, 123U, "samplerA");
    xavierInitializer(2, 4, w2, b2, 123U, "samplerA");
    xavierInitializer(2, 4, w3, b3, 123U, "samplerB");

    for (size_t r = 0; r < w1.rows(); ++r)
    {
        for (size_t c = 0; c < w1.cols(); ++c)
        {
            EXPECT_FLOAT_EQ(w1.at(r, c), w2.at(r, c));
        }
    }
    EXPECT_NE(w1.sum(), w3.sum());
}

TEST(InitializerTest, XavierBiasDimensionMismatchReturnsWithoutChanges)
{
    nn::Tensor weights(4, 2);
    nn::Tensor bias_wrong(4, 2); // expected shape is (out_features, 1)
    weights.fill(7.0F);
    bias_wrong.fill(9.0F);

    xavierInitializer(2, 4, weights, bias_wrong, 1U, "sampler");

    EXPECT_FLOAT_EQ(weights.at(0, 0), 7.0F);
    EXPECT_FLOAT_EQ(weights.at(3, 1), 7.0F);
    EXPECT_FLOAT_EQ(bias_wrong.at(0, 0), 9.0F);
    EXPECT_FLOAT_EQ(bias_wrong.at(3, 1), 9.0F);
}