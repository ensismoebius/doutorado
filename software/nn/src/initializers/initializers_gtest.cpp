#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "kaiming_snn.hpp"
#include "layers/Linear.hpp"
#include "tensor/Tensor.hpp"
#include "xavier.hpp"

// Initializer: kaiming_snn
TEST(InitializerTest, KaimingSNN)
{
    auto layer = std::make_shared<Linear>(2, 4);
    kaimingSNNInitializer(layer);
    ASSERT_NE(layer->weight.data.sum(), 0.0F);
    ASSERT_EQ(layer->bias.data.sum(), 0.0F);
}

// Initializer: Xavier
TEST(InitializerTest, Xavier)
{
    Tensor weights(Eigen::MatrixXf::Zero(4, 2));
    Tensor bias(Eigen::MatrixXf::Zero(4, 1));
    xavierInitializer(2, 4, weights, bias);
    ASSERT_NE(weights.data.sum(), 0.0F);
    ASSERT_NE(bias.data.sum(), 0.0F);
}
