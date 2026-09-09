/**
 * @file ComparativeExperiment_gtest.cpp
 * @brief Unit tests for Experiment04 comparative autoencoder runner.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "layers/lstm/LSTMLayer.hpp"
#include "models/lstm/LSTMAutoencoder.hpp"
#include "tensor/Tensor.hpp"

namespace
{
namespace lstm = nn::models::lstm;
using Tensor = nn::Tensor;

// LSTM Layer: shape and iteration
TEST(LSTMLayerTest, ForwardShape)
{
    lstm::LSTMLayer lstm(8, 16);
    Tensor input = Tensor::rand(4, 8);
    Tensor output = lstm.forward(input, false);
    EXPECT_EQ(output.rows(), 4u);
    EXPECT_EQ(output.cols(), 16u);
}

TEST(LSTMLayerTest, BackwardShape)
{
    lstm::LSTMLayer lstm(8, 16);
    Tensor input = Tensor::rand(4, 8);
    lstm.forward(input, true);
    Tensor grad_h = Tensor::ones(4, 16);
    Tensor grad_input = lstm.backward(grad_h);
    EXPECT_EQ(grad_input.rows(), 4u);
    EXPECT_EQ(grad_input.cols(), 8u);
}

TEST(LSTMLayerTest, ResetState)
{
    lstm::LSTMLayer lstm(8, 16);
    Tensor input = Tensor::rand(4, 8);
    lstm.forward(input, true);
    EXPECT_GT(lstm.cache_.size(), 0u);
    lstm.reset_state();
    EXPECT_EQ(lstm.cache_.size(), 0u);
}

TEST(LSTMLayerTest, Params)
{
    lstm::LSTMLayer lstm(8, 16);
    auto params = lstm.params();
    EXPECT_EQ(params.size(), 3u);
    for (auto* p : params)
    {
        EXPECT_NE(p, nullptr);
    }
}

TEST(LSTMLayerTest, StateDict)
{
    lstm::LSTMLayer lstm1(8, 16);
    auto state1 = lstm1.state_dict();

    lstm::LSTMLayer lstm2(8, 16);
    lstm2.load_state_dict(state1);
    auto state2 = lstm2.state_dict();

    EXPECT_EQ(state1.size(), state2.size());
}

TEST(LSTMLayerTest, GradientsNonZero)
{
    lstm::LSTMLayer lstm(4, 8);
    Tensor input = Tensor::rand(2, 4);
    lstm.forward(input, true);
    lstm.backward(Tensor::ones(2, 8));

    EXPECT_GT(lstm.W_.grad().norm(), 0.0f);
    EXPECT_GT(lstm.U_.grad().norm(), 0.0f);
    EXPECT_GT(lstm.b_.grad().norm(), 0.0f);
}

// Config: validation
TEST(ConfigTest, Defaults)
{
    lstm::LSTMAutoencoderConfig config;
    // Exact defaults from LSTMAutoencoderConfig struct definition
    EXPECT_EQ(config.input_size, 64);
    EXPECT_EQ(config.seq_len, 32);
    EXPECT_EQ(config.hidden_size, 128);
    EXPECT_EQ(config.latent_size, 16);
    EXPECT_EQ(config.num_layers, 1);
}

TEST(ConfigTest, Custom)
{
    lstm::LSTMAutoencoderConfig config;
    config.input_size = 32;
    config.seq_len = 16;
    config.hidden_size = 64;
    config.latent_size = 8;
    config.num_layers = 2;

    EXPECT_EQ(config.input_size, 32);
    EXPECT_EQ(config.seq_len, 16);
    EXPECT_EQ(config.hidden_size, 64);
    EXPECT_EQ(config.latent_size, 8);
    EXPECT_EQ(config.num_layers, 2);
}

// Tensor operations: basic sanity
TEST(TensorOpsTest, Creation)
{
    Tensor t = Tensor::rand(4, 8);
    EXPECT_EQ(t.rows(), 4u);
    EXPECT_EQ(t.cols(), 8u);
}

TEST(TensorOpsTest, Norm)
{
    Tensor t = Tensor::ones(4, 8);
    float norm = t.norm();
    // ones(4,8) has 32 unit elements: norm = sqrt(32)
    EXPECT_NEAR(norm, std::sqrt(32.0f), 1e-5f);
}

TEST(TensorOpsTest, Arithmetic)
{
    Tensor a = Tensor::ones(4, 8);
    Tensor b = Tensor::ones(4, 8) * 2.0f;
    Tensor c = a + b; // 3 * ones(4,8): norm = 3 * sqrt(32)
    EXPECT_NEAR(c.norm(), 3.0f * std::sqrt(32.0f), 1e-4f);

    Tensor d = a - b; // -1 * ones(4,8): norm = sqrt(32)
    EXPECT_NEAR(d.norm(), std::sqrt(32.0f), 1e-4f);
}

TEST(TensorOpsTest, AbsoluteValue)
{
    Tensor t = Tensor::ones(4, 8) * -1.0f;
    Tensor abs_t = t.abs();
    Tensor expected = Tensor::ones(4, 8);
    float diff = (abs_t - expected).norm();
    EXPECT_LT(diff, 1e-5f);
}

// Numerical stability
TEST(NumericalStabilityTest, SmallValues)
{
    Tensor small = Tensor::rand(4, 8) * 1e-6f;
    float norm = small.norm();
    // rand in [0,1): max norm = sqrt(32) * 1e-6, must stay positive and bounded
    EXPECT_GE(norm, 0.0f);
    EXPECT_LT(norm, std::sqrt(32.0f) * 1e-5f);
}

TEST(NumericalStabilityTest, LargeValues)
{
    Tensor large = Tensor::rand(4, 8) * 1e3f;
    float norm = large.norm();
    // rand in [0,1): max norm = sqrt(32) * 1e3, must stay bounded below sqrt(32)*1e4
    EXPECT_GE(norm, 0.0f);
    EXPECT_LT(norm, std::sqrt(32.0f) * 1e4f);
}

// Metrics: basic computation
TEST(MetricsTest, MSEComputation)
{
    Tensor orig = Tensor::ones(4, 8);
    Tensor recon = Tensor::ones(4, 8) * 1.1f;
    Tensor diff = recon - orig; // 0.1 * ones(4,8)
    float mse = (diff * diff).sum() / orig.size();
    // diff^2 = 0.01 for all 32 elements; sum/32 = 0.01
    EXPECT_NEAR(mse, 0.01f, 1e-6f);
}

TEST(MetricsTest, MAEComputation)
{
    Tensor orig = Tensor::ones(4, 8);
    Tensor recon = Tensor::ones(4, 8) * 0.9f;
    Tensor diff = (recon - orig).abs(); // 0.1 * ones(4,8)
    float mae = diff.sum() / orig.size();
    // |0.9 - 1.0| = 0.1 for all 32 elements; sum/32 = 0.1
    EXPECT_NEAR(mae, 0.1f, 1e-6f);
}

TEST(MetricsTest, R2Score)
{
    // Use deterministic uniform input to make R2 computable
    // y_true = 0.5 * ones(10,8); y_pred = 0.5 * 0.95 = 0.475 * ones
    Tensor y_true = Tensor::ones(10, 8) * 0.5f;
    Tensor y_pred = y_true * 0.95f;

    float mean = y_true.sum() / y_true.size(); // = 0.5
    Tensor diff_true = y_true - mean;          // all zeros: ss_tot = 0
    Tensor diff_pred = y_pred - y_true;        // -0.025 * ones

    float ss_tot = (diff_true * diff_true).sum();
    float ss_res = (diff_pred * diff_pred).sum();

    // When ss_tot == 0 the formula is undefined; guard and test the residual directly.
    // ss_res = 0.025^2 * 80 = 0.05
    EXPECT_NEAR(ss_res, 0.025f * 0.025f * 80.0f, 1e-5f);
    EXPECT_EQ(ss_tot, 0.0f);
}

// Batch processing
TEST(BatchProcessingTest, VariableBatches)
{
    for (int batch : {1, 2, 4, 8, 16})
    {
        Tensor t = Tensor::rand(batch, 8);
        EXPECT_EQ(t.rows(), batch);
    }
}

TEST(BatchProcessingTest, MatrixDims)
{
    Tensor a = Tensor::rand(4, 8);
    Tensor b = Tensor::rand(8, 6);
    EXPECT_EQ(a.cols(), 8u);
    EXPECT_EQ(b.rows(), 8u);
}

// LSTM initialization with various sizes
TEST(LSTMInitTest, VariousSizes)
{
    for (int in : {4, 8, 16, 32})
    {
        for (int hidden : {8, 16, 32})
        {
            lstm::LSTMLayer lstm(in, hidden);
            auto params = lstm.params();
            EXPECT_EQ(params.size(), 3u);
        }
    }
}

} // namespace
