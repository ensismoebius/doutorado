/**
 * @file fundamental_mechanisms_convolution_gtest.cpp
 * @brief Fundamental-mechanism correctness for Conv1d, MaxPool1d, and MaxPool2d.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <vector>

#include "FundamentalMechanismsHelpers.hpp"
#include "layers/activations/LeakyReLU.hpp"
#include "layers/activations/ReLU.hpp"
#include "layers/activations/Sigmoid.hpp"
#include "layers/activations/Tanh.hpp"
#include "layers/convolution/Conv1d.hpp"
#include "layers/convolution/MaxPool1d.hpp"
#include "layers/convolution/MaxPool2d.hpp"
#include "layers/dense/Linear.hpp"
#include "layers/losses/CrossEntropyLoss.hpp"
#include "layers/losses/MAELoss.hpp"
#include "layers/losses/SpikeTimeLoss.hpp"
#include "layers/lstm/LSTMLayer.hpp"
#include "layers/residual/ResNetBlock.hpp"
#include "layers/residual/ResidualBlock.hpp"
#include "layers/residual/SimpleResNet.hpp"
#include "layers/spiking/Lif.hpp"
#include "layers/spiking/LifBPTT.hpp"
#include "layers/spiking/LifIntegrator.hpp"
#include "layers/spiking/PoissonLatentLayer.hpp"
#include "layers/spiking/ThresholdDependentBatchNorm.hpp"
#include "tensor/Tensor.hpp"

using Tensor = nn::Tensor;
using Backend = nn::Backend;

// ===========================================================================
// Conv1dTest
// ===========================================================================

TEST(Conv1dTest, OutputShape)
{
    // L_out = (L + 2P - D*(K-1) - 1) / S + 1 = (5+0-2-1)/1+1 = 3
    Conv1dImpl<Backend> conv(1, 2, 3, 1, 0, 1); // C_in=1, C_out=2, K=3, S=1, P=0
    Tensor x(1, 1, 5);
    x.setZero();
    Tensor out = conv.forward(x, false);
    auto shape = out.get_shape();
    ASSERT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 1u); // B
    EXPECT_EQ(shape[1], 2u); // C_out
    EXPECT_EQ(shape[2], 3u); // L_out = 5-2 = 3
}

TEST(Conv1dTest, BiasShape)
{
    Conv1dImpl<Backend> conv(2, 4, 3, 1, 1, 1);
    const Tensor& b = conv.get_bias();
    // bias shape: (out_channels, 1) or (1, out_channels) — just check element count
    EXPECT_EQ(b.size(), 4u);
}

TEST(Conv1dTest, BackwardIdentityKernelOne)
{
    // K=1, S=1, P=0 with w=1 and b=0 is identity: y=x, so dx must equal grad_out.
    Conv1dImpl<Backend> conv(1, 1, 1, 1, 0, 1);
    conv.set_weights(Tensor::ones(1, 1));
    Tensor bz(1, 1);
    bz.setZero();
    conv.set_bias(bz);

    Tensor x(1, 1, 5);
    x.at(0, 0, 0) = 1.0f;
    x.at(0, 0, 1) = 2.0f;
    x.at(0, 0, 2) = 3.0f;
    x.at(0, 0, 3) = 4.0f;
    x.at(0, 0, 4) = 5.0f;
    Tensor out = conv.forward(x, true);
    ASSERT_EQ(out.get_shape()[2], 5u);

    Tensor grad(1, 1, 5);
    grad.at(0, 0, 0) = 1.0f;
    grad.at(0, 0, 1) = 2.0f;
    grad.at(0, 0, 2) = 3.0f;
    grad.at(0, 0, 3) = 4.0f;
    grad.at(0, 0, 4) = 5.0f;
    Tensor dx = conv.backward(grad);

    for (size_t i = 0; i < grad.size(); ++i) EXPECT_NEAR(dx.at(i), grad.at(i), 1e-5f);
}

TEST(Conv1dTest, KernelOneIdentity)
{
    // K=1, P=0, S=1 conv with weight=1, bias=0: output should equal input
    Conv1dImpl<Backend> conv(1, 1, 1, 1, 0, 1);
    // Weight layout is (C_in * K, C_out); here (1, 1)
    conv.set_weights(Tensor::ones(1, 1));
    Tensor bz(1, 1);
    bz.setZero();
    conv.set_bias(bz);

    Tensor x(1, 1, 4);
    x.at(0, 0, 0) = 1.f;
    x.at(0, 0, 1) = 2.f;
    x.at(0, 0, 2) = 3.f;
    x.at(0, 0, 3) = 4.f;

    Tensor out = conv.forward(x, false);
    ASSERT_EQ(out.get_shape()[2], 4u);
    EXPECT_NEAR(out.at(0, 0, 0), 1.f, 1e-5f);
    EXPECT_NEAR(out.at(0, 0, 1), 2.f, 1e-5f);
    EXPECT_NEAR(out.at(0, 0, 2), 3.f, 1e-5f);
    EXPECT_NEAR(out.at(0, 0, 3), 4.f, 1e-5f);
}

// ===========================================================================
// MaxPoolTest
// ===========================================================================

TEST(MaxPoolTest, MaxPool1dKnownValues)
{
    MaxPool1dImpl<Backend> pool(2, 2);
    Tensor x(1, 1, 6);
    x.at(0, 0, 0) = 1.0F;
    x.at(0, 0, 1) = 3.0F;
    x.at(0, 0, 2) = 2.0F;
    x.at(0, 0, 3) = 5.0F;
    x.at(0, 0, 4) = 4.0F;
    x.at(0, 0, 5) = 6.0F;

    Tensor out = pool.forward(x, false);
    ASSERT_EQ(out.get_shape().size(), 3u);
    EXPECT_EQ(out.get_shape()[2], 3u);
    EXPECT_NEAR(out.at(0, 0, 0), 3.0F, 1e-6F);
    EXPECT_NEAR(out.at(0, 0, 1), 5.0F, 1e-6F);
    EXPECT_NEAR(out.at(0, 0, 2), 6.0F, 1e-6F);
}

TEST(MaxPoolTest, MaxPool2dKnownValues)
{
    MaxPool2dImpl<Backend> pool(2, 2);
    Tensor x(1, 1, 4, 4);
    x.at(0, 0, 0, 0) = 1.0F;
    x.at(0, 0, 0, 1) = 2.0F;
    x.at(0, 0, 0, 2) = 3.0F;
    x.at(0, 0, 0, 3) = 4.0F;
    x.at(0, 0, 1, 0) = 5.0F;
    x.at(0, 0, 1, 1) = 6.0F;
    x.at(0, 0, 1, 2) = 7.0F;
    x.at(0, 0, 1, 3) = 8.0F;
    x.at(0, 0, 2, 0) = 9.0F;
    x.at(0, 0, 2, 1) = 10.0F;
    x.at(0, 0, 2, 2) = 11.0F;
    x.at(0, 0, 2, 3) = 12.0F;
    x.at(0, 0, 3, 0) = 13.0F;
    x.at(0, 0, 3, 1) = 14.0F;
    x.at(0, 0, 3, 2) = 15.0F;
    x.at(0, 0, 3, 3) = 16.0F;

    Tensor out = pool.forward(x, false);
    auto shape = out.get_shape();
    ASSERT_EQ(shape.size(), 4u);
    EXPECT_EQ(shape[2], 2u);
    EXPECT_EQ(shape[3], 2u);
    EXPECT_NEAR(out.at(0, 0, 0, 0), 6.0F, 1e-6F);
    EXPECT_NEAR(out.at(0, 0, 0, 1), 8.0F, 1e-6F);
    EXPECT_NEAR(out.at(0, 0, 1, 0), 14.0F, 1e-6F);
    EXPECT_NEAR(out.at(0, 0, 1, 1), 16.0F, 1e-6F);
}

// ===========================================================================
// MaxPoolTest additions — backward gradient routing to argmax positions
// ===========================================================================

TEST(MaxPoolTest, MaxPool1dBackwardRoutesToArgmax)
{
    // Pool(kernel=2, stride=2): windows [0..1]→max=3@pos0, [2..3]→max=4@pos2
    MaxPool1dImpl<Backend> pool(2, 2);
    Tensor x(1, 1, 4);
    x.at(0, 0, 0) = 3.0f;
    x.at(0, 0, 1) = 1.0f;
    x.at(0, 0, 2) = 4.0f;
    x.at(0, 0, 3) = 2.0f;
    pool.forward(x, true);

    Tensor go(1, 1, 2);
    go.at(0, 0, 0) = 1.0f;
    go.at(0, 0, 1) = 1.0f;
    Tensor dx = pool.backward(go);

    EXPECT_NEAR(dx.at(0, 0, 0), 1.0f, 1e-6f) << "argmax at pos 0 must receive gradient";
    EXPECT_NEAR(dx.at(0, 0, 1), 0.0f, 1e-6f) << "non-max pos 1 must have zero gradient";
    EXPECT_NEAR(dx.at(0, 0, 2), 1.0f, 1e-6f) << "argmax at pos 2 must receive gradient";
    EXPECT_NEAR(dx.at(0, 0, 3), 0.0f, 1e-6f) << "non-max pos 3 must have zero gradient";
}

TEST(MaxPoolTest, MaxPool2dBackwardRoutesToArgmax)
{
    // Pool(kernel=2, stride=2) on (1,1,4,4): 4 windows, each 2x2
    // Window top-left (0..1,0..1): max at (1,1)=6
    // Window top-right (0..1,2..3): max at (1,3)=8
    // Window bottom-left (2..3,0..1): max at (3,1)=14
    // Window bottom-right (2..3,2..3): max at (3,3)=16
    MaxPool2dImpl<Backend> pool(2, 2);
    Tensor x(1, 1, 4, 4);
    for (size_t r = 0; r < 4; ++r)
        for (size_t c = 0; c < 4; ++c) x.at(0, 0, r, c) = static_cast<float>(r * 4 + c + 1);
    // Values: 1..16, row-major → max of each 2x2 block is bottom-right element
    pool.forward(x, true);

    Tensor go(1, 1, 2, 2);
    for (size_t r = 0; r < 2; ++r)
        for (size_t c = 0; c < 2; ++c) go.at(0, 0, r, c) = 1.0f;
    Tensor dx = pool.backward(go);

    // Argmax positions: (1,1), (1,3), (3,1), (3,3) — each should get gradient 1
    // Non-argmax positions should get 0
    EXPECT_NEAR(dx.at(0, 0, 1, 1), 1.0f, 1e-6f); // argmax of top-left block
    EXPECT_NEAR(dx.at(0, 0, 1, 3), 1.0f, 1e-6f); // argmax of top-right block
    EXPECT_NEAR(dx.at(0, 0, 3, 1), 1.0f, 1e-6f); // argmax of bottom-left block
    EXPECT_NEAR(dx.at(0, 0, 3, 3), 1.0f, 1e-6f); // argmax of bottom-right block
    // Spot-check non-argmax positions
    EXPECT_NEAR(dx.at(0, 0, 0, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(dx.at(0, 0, 1, 0), 0.0f, 1e-6f);
    EXPECT_NEAR(dx.at(0, 0, 0, 1), 0.0f, 1e-6f);
}

// MaxPool1d exception: wrong number of input dimensions (MaxPool1d.hpp line 35)
// MaxPool1d exception: wrong number of input dimensions (MaxPool1d.hpp line 35)
TEST(MaxPoolTest, MaxPool1dThrowsOnWrongInputShape)
{
    MaxPool1dImpl<Backend> pool(2, 2);
    Tensor x(4, 4); // 2D, not 3D
    x.setZero();
    EXPECT_THROW(pool.forward(x, false), std::invalid_argument);
}

// MaxPool1d exception: output length <= 0 (MaxPool1d.hpp line 43)
TEST(MaxPoolTest, MaxPool1dThrowsOnOutputLengthZero)
{
    MaxPool1dImpl<Backend> pool(10, 1); // kernel=10 > length=4 -> L_out <= 0
    Tensor x(1, 1, 4);                  // 3D: B=1, C=1, L=4
    x.setZero();
    EXPECT_THROW(pool.forward(x, false), std::invalid_argument);
}

// MaxPool2d exception: wrong number of input dimensions (MaxPool2d.hpp line 40)
TEST(MaxPoolTest, MaxPool2dThrowsOnWrongInputShape)
{
    MaxPool2dImpl<Backend> pool(2, 2);
    Tensor x(4, 4); // 2D, not 4D
    x.setZero();
    EXPECT_THROW(pool.forward(x, false), std::invalid_argument);
}

// MaxPool2d exception: output size <= 0 (MaxPool2d.hpp line 50)
TEST(MaxPoolTest, MaxPool2dThrowsOnOutputSizeZero)
{
    MaxPool2dImpl<Backend> pool(10, 1); // kernel=10 > H=4 -> H_out <= 0
    Tensor x(1, 1, 4, 4);               // 4D: B=1, C=1, H=4, W=4
    x.setZero();
    EXPECT_THROW(pool.forward(x, false), std::invalid_argument);
}
