/**
 * @file AutoencoderRedesign_gtest.cpp
 * @brief Tests for experiment03 redesigned multimodal autoencoders.
 */

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "AutoencoderConfig.hpp"
#include "FusedWindowAutoencoder.hpp"
#include "FusedWindowSpikingAutoencoder.hpp"
#include "ProtocolAutoencoder.hpp"
#include "ProtocolSpikingAutoencoder.hpp"
#include "nn/layers/base/Sequential.hpp"
#include "nn/layers/eigen/Layers.hpp"
#include "nn/layers/spiking/Leaky.hpp"
#include "nn/tensor/Tensor.hpp"

namespace
{

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
            if (auto leaky = std::dynamic_pointer_cast<Leaky>(layer))
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
            if (auto leaky = std::dynamic_pointer_cast<Leaky>(layer))
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

TEST(Experiment03RedesignTest, FusedAnnRequiresSplitHints)
{
    AutoencoderConfig cfg;
    cfg.input_features = 14;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.depth = 2;

    EXPECT_THROW({ FusedWindowAutoencoder model(cfg); }, std::invalid_argument);
}

TEST(Experiment03RedesignTest, FusedAnnForwardBackwardAndParams)
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

TEST(Experiment03RedesignTest, FusedSnnForwardBackwardAndParams)
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

TEST(Experiment03RedesignTest, FusedSnnResetStateClearsMembranes)
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

TEST(Experiment03RedesignTest, ProtocolAnnDualBranchForwardBackwardAndParams)
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

TEST(Experiment03RedesignTest, ProtocolSnnDualBranchForwardBackwardAndParams)
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

TEST(Experiment03RedesignTest, ProtocolSnnDualBranchResetStateClearsMembranes)
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

TEST(Experiment03RedesignTest, ProtocolAnnDenseFallbackForwardBackwardAndParams)
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

TEST(Experiment03RedesignTest, ProtocolSnnDenseFallbackForwardBackwardAndParams)
{
    AutoencoderConfig cfg;
    cfg.input_features = 14;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.depth = 2;
    cfg.architecture = AutoencoderArchitecture::ResidualDense;
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

TEST(Experiment03RedesignTest, ProtocolAnnDenseFallbackSupportsBroaderLayerGrammar)
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

TEST(Experiment03RedesignTest, ProtocolSnnDenseFallbackSupportsBroaderLayerGrammar)
{
    AutoencoderConfig cfg;
    cfg.input_features = 14;
    cfg.hidden_size = 16;
    cfg.latent_size = 4;
    cfg.depth = 2;
    cfg.architecture = AutoencoderArchitecture::ResidualDense;
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
