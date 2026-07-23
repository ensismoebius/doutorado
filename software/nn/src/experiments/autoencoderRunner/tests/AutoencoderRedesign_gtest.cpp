/**
 * @file AutoencoderRedesign_gtest.cpp
 * @brief Tests for autoencoderRunner redesigned multimodal autoencoders.
 */

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "AutoencoderConfig.hpp"
#include "FusedWindowAutoencoder.hpp"
#include "FusedWindowSpikingAutoencoder.hpp"
#include "ProtocolAutoencoder.hpp"
#include "ProtocolSpikingAutoencoder.hpp"
#include "layers/Layers.hpp"
#include "layers/base/Sequential.hpp"
#include "layers/spiking/Lif.hpp"
#include "tensor/Tensor.hpp"

namespace
{

using nn::Lif;
using nn::LifBPTT;
using nn::Sequential;

auto make_fused_cfg() -> AutoencoderConfig
{
    AutoencoderConfig cfg;
    cfg.input_features = 14;
    cfg.eeg_features = 6;
    cfg.audio_features = 8;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.depth = 2;
    cfg.residual_blocks = 1;
    // Single frame per sample in these unit tests, so the BPTT unroll is one step.
    // Declared explicitly: leaving time_steps unset now raises by design.
    cfg.time_steps = 1;
    cfg.time_step = 1.0F;
    cfg.resistance = 1.0F;
    cfg.capacitance = 1.0F;
    return cfg;
}

auto has_initialized_membrane_state(const std::vector<Sequential*>& blocks) -> bool
{
    for (const auto* seq : blocks)
    {
        for (const auto& layer : seq->layers)
        {
            if (auto leaky = std::dynamic_pointer_cast<LifBPTT>(layer))
            {
                if (leaky->v_mem.size() > 0)
                {
                    return true;
                }
            }
        }
    }
    return false;
}

auto membrane_state_is_zeroed(const std::vector<Sequential*>& blocks) -> bool
{
    for (const auto* seq : blocks)
    {
        for (const auto& layer : seq->layers)
        {
            if (auto leaky = std::dynamic_pointer_cast<LifBPTT>(layer))
            {
                if (leaky->v_mem.size() > 0 && leaky->v_mem.sum() != 0.0F)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

} // namespace

TEST(AutoencoderRunnerRedesignTest, FusedAnnRequiresSplitHints)
{
    AutoencoderConfig cfg;
    cfg.input_features = 14;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.depth = 2;

    EXPECT_THROW({ FusedWindowAutoencoder model(cfg); }, std::invalid_argument);
}

TEST(AutoencoderRunnerRedesignTest, FusedAnnForwardBackwardAndParams)
{
    AutoencoderConfig cfg = make_fused_cfg();
    FusedWindowAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::rand(3, cfg.input_features);
    nn::Tensor reconstruction = model.forward(input, true);
    EXPECT_EQ(reconstruction.rows(), input.rows());
    EXPECT_EQ(reconstruction.cols(), input.cols());

    nn::Tensor grad_output = nn::Tensor::ones(input.rows(), input.cols());
    nn::Tensor grad_input = model.backward(grad_output);
    EXPECT_EQ(grad_input.rows(), input.rows());
    EXPECT_EQ(grad_input.cols(), input.cols());

    EXPECT_FALSE(model.params().empty());
}

TEST(AutoencoderRunnerRedesignTest, FusedSnnForwardBackwardAndParams)
{
    AutoencoderConfig cfg = make_fused_cfg();
    FusedWindowSpikingAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::rand(2, cfg.input_features);
    nn::Tensor reconstruction = model.forward(input, true);
    EXPECT_EQ(reconstruction.rows(), input.rows());
    EXPECT_EQ(reconstruction.cols(), input.cols());

    nn::Tensor grad_output = nn::Tensor::ones(input.rows(), input.cols());
    nn::Tensor grad_input = model.backward(grad_output);
    EXPECT_EQ(grad_input.rows(), input.rows());
    EXPECT_EQ(grad_input.cols(), input.cols());

    EXPECT_FALSE(model.params().empty());
}

TEST(AutoencoderRunnerRedesignTest, FusedSnnResetStateClearsMembranes)
{
    AutoencoderConfig cfg = make_fused_cfg();
    FusedWindowSpikingAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::ones(2, cfg.input_features);
    (void) model.forward(input, true);

    std::vector<Sequential*> blocks = {&model.eeg_encoder_,
        &model.audio_encoder_,
        &model.fusion_encoder_,
        &model.fusion_decoder_,
        &model.eeg_decoder_,
        &model.audio_decoder_};

    EXPECT_TRUE(has_initialized_membrane_state(blocks));

    model.reset_state();

    EXPECT_TRUE(membrane_state_is_zeroed(blocks));
}

TEST(AutoencoderRunnerRedesignTest, ProtocolAnnDualBranchForwardBackwardAndParams)
{
    AutoencoderConfig cfg = make_fused_cfg();
    cfg.architecture = AutoencoderArchitecture::DualBranchFusion;

    ProtocolAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::rand(3, cfg.input_features);
    nn::Tensor reconstruction = model.forward(input, true);
    EXPECT_EQ(reconstruction.rows(), input.rows());
    EXPECT_EQ(reconstruction.cols(), input.cols());

    nn::Tensor grad_output = nn::Tensor::ones(input.rows(), input.cols());
    nn::Tensor grad_input = model.backward(grad_output);
    EXPECT_EQ(grad_input.rows(), input.rows());
    EXPECT_EQ(grad_input.cols(), input.cols());

    EXPECT_FALSE(model.params().empty());
}

TEST(AutoencoderRunnerRedesignTest, ProtocolSnnDualBranchForwardBackwardAndParams)
{
    AutoencoderConfig cfg = make_fused_cfg();
    cfg.architecture = AutoencoderArchitecture::DualBranchFusion;

    ProtocolSpikingAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::rand(2, cfg.input_features);
    nn::Tensor reconstruction = model.forward(input, true);
    EXPECT_EQ(reconstruction.rows(), input.rows());
    EXPECT_EQ(reconstruction.cols(), input.cols());

    nn::Tensor grad_output = nn::Tensor::ones(input.rows(), input.cols());
    nn::Tensor grad_input = model.backward(grad_output);
    EXPECT_EQ(grad_input.rows(), input.rows());
    EXPECT_EQ(grad_input.cols(), input.cols());

    EXPECT_FALSE(model.params().empty());
}

TEST(AutoencoderRunnerRedesignTest, ProtocolSnnDualBranchResetStateClearsMembranes)
{
    AutoencoderConfig cfg = make_fused_cfg();
    cfg.architecture = AutoencoderArchitecture::DualBranchFusion;

    ProtocolSpikingAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::ones(2, cfg.input_features);
    (void) model.forward(input, true);

    std::vector<Sequential*> blocks = {&model.eeg_encoder_,
        &model.audio_encoder_,
        &model.fusion_encoder_,
        &model.fusion_decoder_,
        &model.eeg_decoder_,
        &model.audio_decoder_};

    EXPECT_TRUE(has_initialized_membrane_state(blocks));

    model.reset_state();

    EXPECT_TRUE(membrane_state_is_zeroed(blocks));
}

TEST(AutoencoderRunnerRedesignTest, ProtocolAnnDenseFallbackForwardBackwardAndParams)
{
    AutoencoderConfig cfg;
    cfg.input_features = 14;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.depth = 2;
    cfg.architecture = AutoencoderArchitecture::ResidualDense;

    ProtocolAutoencoder model(cfg);

    EXPECT_FALSE(model.use_dual_branch_);
    EXPECT_FALSE(model.encoder_.layers.empty());
    EXPECT_TRUE(model.eeg_encoder_.layers.empty());
    EXPECT_TRUE(model.audio_encoder_.layers.empty());

    nn::Tensor input = nn::Tensor::rand(3, cfg.input_features);
    nn::Tensor reconstruction = model.forward(input, true);
    EXPECT_EQ(reconstruction.rows(), input.rows());
    EXPECT_EQ(reconstruction.cols(), input.cols());

    nn::Tensor grad_output = nn::Tensor::ones(input.rows(), input.cols());
    nn::Tensor grad_input = model.backward(grad_output);
    EXPECT_EQ(grad_input.rows(), input.rows());
    EXPECT_EQ(grad_input.cols(), input.cols());

    EXPECT_FALSE(model.params().empty());
}

TEST(AutoencoderRunnerRedesignTest, ProtocolSnnDenseFallbackForwardBackwardAndParams)
{
    AutoencoderConfig cfg;
    cfg.input_features = 14;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.depth = 2;
    cfg.architecture = AutoencoderArchitecture::ResidualDense;
    // Single frame per sample in these unit tests, so the BPTT unroll is one step.
    // Declared explicitly: leaving time_steps unset now raises by design.
    cfg.time_steps = 1;
    cfg.time_step = 1.0F;
    cfg.resistance = 1.0F;
    cfg.capacitance = 1.0F;

    ProtocolSpikingAutoencoder model(cfg);

    EXPECT_FALSE(model.use_dual_branch_);
    EXPECT_FALSE(model.encoder_.layers.empty());
    EXPECT_TRUE(model.eeg_encoder_.layers.empty());
    EXPECT_TRUE(model.audio_encoder_.layers.empty());

    nn::Tensor input = nn::Tensor::rand(2, cfg.input_features);
    nn::Tensor reconstruction = model.forward(input, true);
    EXPECT_EQ(reconstruction.rows(), input.rows());
    EXPECT_EQ(reconstruction.cols(), input.cols());

    nn::Tensor grad_output = nn::Tensor::ones(input.rows(), input.cols());
    nn::Tensor grad_input = model.backward(grad_output);
    EXPECT_EQ(grad_input.rows(), input.rows());
    EXPECT_EQ(grad_input.cols(), input.cols());

    EXPECT_FALSE(model.params().empty());
}

TEST(AutoencoderRunnerRedesignTest, ProtocolAnnDenseFallbackSupportsBroaderLayerGrammar)
{
    AutoencoderConfig cfg;
    cfg.input_features = 14;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.depth = 2;
    cfg.architecture = AutoencoderArchitecture::ResidualDense;
    cfg.encoder_layer_spec = {
        "linear:32",
        "relu",
        "residual:2",
        "linear:latent",
        "identity",
    };
    cfg.decoder_layer_spec = {
        "linear:32",
        "leaky_relu",
        "residual",
        "linear:output",
        "identity",
    };

    ProtocolAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::rand(3, cfg.input_features);
    nn::Tensor reconstruction = model.forward(input, true);
    EXPECT_EQ(reconstruction.rows(), input.rows());
    EXPECT_EQ(reconstruction.cols(), input.cols());

    nn::Tensor grad_output = nn::Tensor::ones(input.rows(), input.cols());
    nn::Tensor grad_input = model.backward(grad_output);
    EXPECT_EQ(grad_input.rows(), input.rows());
    EXPECT_EQ(grad_input.cols(), input.cols());
}

TEST(AutoencoderRunnerRedesignTest, ProtocolSnnDenseFallbackSupportsBroaderLayerGrammar)
{
    AutoencoderConfig cfg;
    cfg.input_features = 14;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.depth = 2;
    cfg.architecture = AutoencoderArchitecture::ResidualDense;
    // Single frame per sample in these unit tests, so the BPTT unroll is one step.
    // Declared explicitly: leaving time_steps unset now raises by design.
    cfg.time_steps = 1;
    cfg.time_step = 1.0F;
    cfg.resistance = 1.0F;
    cfg.capacitance = 1.0F;
    cfg.encoder_layer_spec = {
        "linear:32",
        "leaky",
        "residual",
        "linear:latent",
        "identity",
    };
    cfg.decoder_layer_spec = {
        "linear:32",
        "leaky_integrator",
        "linear:output",
        "identity",
    };

    ProtocolSpikingAutoencoder model(cfg);

    nn::Tensor input = nn::Tensor::rand(2, cfg.input_features);
    nn::Tensor reconstruction = model.forward(input, true);
    EXPECT_EQ(reconstruction.rows(), input.rows());
    EXPECT_EQ(reconstruction.cols(), input.cols());

    nn::Tensor grad_output = nn::Tensor::ones(input.rows(), input.cols());
    nn::Tensor grad_input = model.backward(grad_output);
    EXPECT_EQ(grad_input.rows(), input.rows());
    EXPECT_EQ(grad_input.cols(), input.cols());
}

// Firing-rate regularization (D1 follow-up): lambda > 0 must locate the
// encoder's Lif layer(s) and inject the band-penalty gradient during
// backward, while lambda == 0 (the default) stays a no-op. Two models built
// from an identical seed produce identical forward passes, so any gradient
// difference under an otherwise-identical backward pass is attributable to
// the regularization term alone.
TEST(AutoencoderRunnerRedesignTest, ProtocolSnnFiringRateRegularizationInjectsGradientWhenEnabled)
{
    auto make_cfg = [](float lambda)
    {
        AutoencoderConfig cfg;
        cfg.input_features = 14;
        cfg.hidden_size = 16;
        cfg.latent_size = 4;
        cfg.depth = 1;
        // Single frame per sample in these unit tests, so the BPTT unroll is one step.
        // Declared explicitly: leaving time_steps unset now raises by design.
        cfg.time_steps = 1;
        cfg.time_step = 1.0F;
        cfg.resistance = 1.0F;
        cfg.capacitance = 1.0F;
        cfg.initializer_seed = 42u;
        // fr_min set above any plausible native firing rate so the penalty is
        // guaranteed to engage regardless of the actual (unmeasured) rate.
        cfg.firing_rate_reg_lambda = lambda;
        cfg.firing_rate_min = 0.9F;
        cfg.firing_rate_max = 0.95F;
        return cfg;
    };

    ProtocolSpikingAutoencoder baseline(make_cfg(0.0F));
    ProtocolSpikingAutoencoder regularized(make_cfg(5.0F));

    ASSERT_FALSE(baseline.encoder_lif_indices_.empty());
    ASSERT_FALSE(regularized.encoder_lif_indices_.empty());
    EXPECT_EQ(baseline.encoder_lif_indices_, regularized.encoder_lif_indices_);

    nn::Tensor input = nn::Tensor::rand(3, 14);
    nn::Tensor grad_output = nn::Tensor::ones(3, 14);

    nn::Tensor recon_baseline = baseline.forward(input, true);
    nn::Tensor recon_regularized = regularized.forward(input, true);
    // Identical seed + identical input => identical forward pass; the two
    // models diverge only in the regularization term applied during backward.
    for (int i = 0; i < recon_baseline.rows(); ++i)
        for (int j = 0; j < recon_baseline.cols(); ++j)
            EXPECT_FLOAT_EQ(recon_baseline.at(i, j), recon_regularized.at(i, j));

    nn::Tensor grad_baseline = baseline.backward(grad_output);
    nn::Tensor grad_regularized = regularized.backward(grad_output);

    bool any_difference = false;
    for (int i = 0; i < grad_baseline.rows() && !any_difference; ++i)
        for (int j = 0; j < grad_baseline.cols() && !any_difference; ++j)
            if (grad_baseline.at(i, j) != grad_regularized.at(i, j)) any_difference = true;

    EXPECT_TRUE(any_difference) << "lambda > 0 with an out-of-band firing rate must perturb "
                                   "the encoder's input gradient relative to lambda == 0";
}

// lambda == 0 (the default for every existing profile) must remain exactly
// inert: backward_with_firing_rate_reg degenerates to plain Sequential::backward.
TEST(AutoencoderRunnerRedesignTest, ProtocolSnnFiringRateRegularizationInertWhenLambdaZero)
{
    AutoencoderConfig cfg;
    cfg.input_features = 14;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.depth = 1;
    // Single frame per sample in these unit tests, so the BPTT unroll is one step.
    // Declared explicitly: leaving time_steps unset now raises by design.
    cfg.time_steps = 1;
    cfg.time_step = 1.0F;
    cfg.resistance = 1.0F;
    cfg.capacitance = 1.0F;
    cfg.initializer_seed = 7u;
    EXPECT_FLOAT_EQ(cfg.firing_rate_reg_lambda, 0.0F);

    ProtocolSpikingAutoencoder model(cfg);
    nn::Tensor input = nn::Tensor::rand(2, 14);
    model.forward(input, true);
    nn::Tensor grad_output = nn::Tensor::ones(2, 14);
    nn::Tensor grad_input = model.backward(grad_output);

    EXPECT_EQ(grad_input.rows(), input.rows());
    EXPECT_EQ(grad_input.cols(), input.cols());
}
