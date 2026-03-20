/**
 * @file src/experiments/03/lib/src/FusedWindowSpikingAutoencoder.cpp
 * @brief SNN variant of the fused EEG+audio window autoencoder.
 */

#include "FusedWindowSpikingAutoencoder.hpp"

#include <stdexcept>
#include "AutoencoderBuilders.hpp"

FusedWindowSpikingAutoencoder::FusedWindowSpikingAutoencoder(const AutoencoderConfig& cfg)
    : eeg_encoder_(experiment03::autoencoders::build_snn_encoder(
          cfg, cfg.eeg_features, experiment03::autoencoders::resolved_branch_hidden_size(cfg))),
      audio_encoder_(experiment03::autoencoders::build_snn_encoder(
          cfg, cfg.audio_features, experiment03::autoencoders::resolved_branch_hidden_size(cfg))),
      fusion_encoder_(experiment03::autoencoders::build_snn_encoder(cfg,
          cfg.latent_size * 2,
          experiment03::autoencoders::resolved_fusion_hidden_size(cfg))),
      fusion_decoder_(experiment03::autoencoders::build_snn_decoder(cfg,
          cfg.latent_size * 2,
          experiment03::autoencoders::resolved_fusion_hidden_size(cfg))),
      eeg_decoder_(experiment03::autoencoders::build_snn_decoder(
          cfg, cfg.eeg_features, experiment03::autoencoders::resolved_branch_hidden_size(cfg))),
      audio_decoder_(experiment03::autoencoders::build_snn_decoder(
          cfg, cfg.audio_features, experiment03::autoencoders::resolved_branch_hidden_size(cfg))),
      eeg_features_(cfg.eeg_features),
      audio_features_(cfg.audio_features)
{
    if (eeg_features_ <= 0 || audio_features_ <= 0)
    {
        throw std::invalid_argument(
            "FusedWindowSpikingAutoencoder requires positive eeg_features and audio_features");
    }
}

auto FusedWindowSpikingAutoencoder::encode(const nn::Tensor& input, bool requires_grad)
    -> nn::Tensor
{
    auto eeg = experiment03::autoencoders::slice_columns(input, 0, eeg_features_);
    auto audio =
        experiment03::autoencoders::slice_columns(input, eeg_features_, audio_features_);
    auto eeg_latent = eeg_encoder_.forward(eeg, requires_grad);
    auto audio_latent = audio_encoder_.forward(audio, requires_grad);
    auto fused = experiment03::autoencoders::concat_columns(eeg_latent, audio_latent);
    return fusion_encoder_.forward(fused, requires_grad);
}

auto FusedWindowSpikingAutoencoder::decode(const nn::Tensor& latent, bool requires_grad)
    -> nn::Tensor
{
    auto fused = fusion_decoder_.forward(latent, requires_grad);
    const int branch_cols = fused.cols() / 2;
    auto eeg_branch = experiment03::autoencoders::slice_columns(fused, 0, branch_cols);
    auto audio_branch =
        experiment03::autoencoders::slice_columns(fused, branch_cols, fused.cols() - branch_cols);
    auto eeg_reconstruction = eeg_decoder_.forward(eeg_branch, requires_grad);
    auto audio_reconstruction = audio_decoder_.forward(audio_branch, requires_grad);
    return experiment03::autoencoders::concat_columns(eeg_reconstruction, audio_reconstruction);
}

auto FusedWindowSpikingAutoencoder::forward(const nn::Tensor& input, bool requires_grad)
    -> nn::Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto FusedWindowSpikingAutoencoder::backward(const nn::Tensor& grad_output) -> nn::Tensor
{
    auto eeg_grad = experiment03::autoencoders::slice_columns(grad_output, 0, eeg_features_);
    auto audio_grad =
        experiment03::autoencoders::slice_columns(grad_output, eeg_features_, audio_features_);
    auto eeg_branch_grad = eeg_decoder_.backward(eeg_grad);
    auto audio_branch_grad = audio_decoder_.backward(audio_grad);
    auto fused_branch_grad =
        experiment03::autoencoders::concat_columns(eeg_branch_grad, audio_branch_grad);
    auto latent_grad = fusion_decoder_.backward(fused_branch_grad);
    auto fused_encoder_grad = fusion_encoder_.backward(latent_grad);
    auto eeg_encoder_grad =
        experiment03::autoencoders::slice_columns(fused_encoder_grad, 0, eeg_branch_grad.cols());
    auto audio_encoder_grad = experiment03::autoencoders::slice_columns(
        fused_encoder_grad, eeg_branch_grad.cols(), audio_branch_grad.cols());
    auto eeg_input_grad = eeg_encoder_.backward(eeg_encoder_grad);
    auto audio_input_grad = audio_encoder_.backward(audio_encoder_grad);
    return experiment03::autoencoders::concat_columns(eeg_input_grad, audio_input_grad);
}

auto FusedWindowSpikingAutoencoder::params() -> std::vector<nn::Tensor*>
{
    return experiment03::autoencoders::join_params({eeg_encoder_.params(),
        audio_encoder_.params(),
        fusion_encoder_.params(),
        fusion_decoder_.params(),
        eeg_decoder_.params(),
        audio_decoder_.params()});
}

void FusedWindowSpikingAutoencoder::reset_state()
{
    experiment03::autoencoders::reset_sequential_state(eeg_encoder_);
    experiment03::autoencoders::reset_sequential_state(audio_encoder_);
    experiment03::autoencoders::reset_sequential_state(fusion_encoder_);
    experiment03::autoencoders::reset_sequential_state(fusion_decoder_);
    experiment03::autoencoders::reset_sequential_state(eeg_decoder_);
    experiment03::autoencoders::reset_sequential_state(audio_decoder_);
}
