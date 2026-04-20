/**
 * @file ComparativeExperiment_gtest.cpp
 * @brief Unit tests for Experiment04 comparative autoencoder runner.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "LSTMAutoencoder.hpp"
#include "LSTMLayer.hpp"
#include "nn/tensor/Tensor.hpp"

namespace
{
using Tensor = nn::Tensor;

// LSTM Layer: shape and iteration
TEST(LSTMLayerTest, ForwardShape)
{
    lstm_autoencoder_experiment::LSTMLayer lstm(8, 16);
    Tensor input = Tensor::rand(4, 8);
    Tensor output = lstm.forward(input, false);
    EXPECT_EQ(output.rows(), 4u);
    EXPECT_EQ(output.cols(), 16u);
}

TEST(LSTMLayerTest, BackwardShape)
{
    lstm_autoencoder_experiment::LSTMLayer lstm(8, 16);
    Tensor input = Tensor::rand(4, 8);
    lstm.forward(input, true);
    Tensor grad_h = Tensor::ones(4, 16);
    Tensor grad_input = lstm.backward(grad_h);
    EXPECT_EQ(grad_input.rows(), 4u);
    EXPECT_EQ(grad_input.cols(), 8u);
}

TEST(LSTMLayerTest, ResetState)
{
    lstm_autoencoder_experiment::LSTMLayer lstm(8, 16);
    Tensor input = Tensor::rand(4, 8);
    lstm.forward(input, true);
    EXPECT_GT(lstm.cache_.size(), 0u);
    lstm.reset_state();
    EXPECT_EQ(lstm.cache_.size(), 0u);
}

TEST(LSTMLayerTest, Params)
{
    lstm_autoencoder_experiment::LSTMLayer lstm(8, 16);
    auto params = lstm.params();
    EXPECT_EQ(params.size(), 3u);
    for (auto* p : params)
    {
        EXPECT_NE(p, nullptr);
    }
}

TEST(LSTMLayerTest, StateDict)
{
    lstm_autoencoder_experiment::LSTMLayer lstm1(8, 16);
    auto state1 = lstm1.state_dict();

    lstm_autoencoder_experiment::LSTMLayer lstm2(8, 16);
    lstm2.load_state_dict(state1);
    auto state2 = lstm2.state_dict();

    EXPECT_EQ(state1.size(), state2.size());
}

TEST(LSTMLayerTest, GradientsNonZero)
{
    lstm_autoencoder_experiment::LSTMLayer lstm(4, 8);
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
    lstm_autoencoder_experiment::LSTMAutoencoderConfig config;
    EXPECT_GT(config.input_size, 0);
    EXPECT_GT(config.seq_len, 0);
    EXPECT_GT(config.hidden_size, 0);
    EXPECT_GT(config.latent_size, 0);
    EXPECT_GE(config.num_layers, 1);
}

TEST(ConfigTest, Custom)
{
    lstm_autoencoder_experiment::LSTMAutoencoderConfig config;
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
    EXPECT_GT(norm, 0.0f);
    EXPECT_TRUE(std::isfinite(norm));
}

TEST(TensorOpsTest, Arithmetic)
{
    Tensor a = Tensor::ones(4, 8);
    Tensor b = Tensor::ones(4, 8) * 2.0f;
    Tensor c = a + b;
    EXPECT_GT(c.norm(), a.norm());

    Tensor d = a - b;
    EXPECT_GT(d.norm(), 0.0f);
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
    EXPECT_TRUE(std::isfinite(norm));
}

TEST(NumericalStabilityTest, LargeValues)
{
    Tensor large = Tensor::rand(4, 8) * 1e3f;
    float norm = large.norm();
    EXPECT_TRUE(std::isfinite(norm));
}

// Metrics: basic computation
TEST(MetricsTest, MSEComputation)
{
    Tensor orig = Tensor::ones(4, 8);
    Tensor recon = Tensor::ones(4, 8) * 1.1f;
    Tensor diff = recon - orig;
    float mse = (diff * diff).sum() / orig.size();
    EXPECT_GT(mse, 0.0f);
    EXPECT_LT(mse, 1.0f);
}

TEST(MetricsTest, MAEComputation)
{
    Tensor orig = Tensor::ones(4, 8);
    Tensor recon = Tensor::ones(4, 8) * 0.9f;
    Tensor diff = (recon - orig).abs();
    float mae = diff.sum() / orig.size();
    EXPECT_GT(mae, 0.0f);
    EXPECT_LT(mae, 1.0f);
}

TEST(MetricsTest, R2Score)
{
    Tensor y_true = Tensor::rand(10, 8);
    Tensor y_pred = y_true * 0.95f;

    float mean = y_true.sum() / y_true.size();
    Tensor diff_true = y_true - mean;
    Tensor diff_pred = y_pred - y_true;

    float ss_tot = (diff_true * diff_true).sum();
    float ss_res = (diff_pred * diff_pred).sum();

    float r2 = 1.0f - (ss_res / ss_tot);
    EXPECT_LT(r2, 1.0f);
    EXPECT_TRUE(std::isfinite(r2));
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
            lstm_autoencoder_experiment::LSTMLayer lstm(in, hidden);
            auto params = lstm.params();
            EXPECT_EQ(params.size(), 3u);
        }
    }
}

} // namespace
