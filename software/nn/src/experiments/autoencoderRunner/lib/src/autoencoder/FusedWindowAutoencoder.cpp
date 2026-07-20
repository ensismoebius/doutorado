/**
 * @file src/experiments/autoencoderRunner/lib/src/autoencoder/FusedWindowAutoencoder.cpp
 * @brief ANN variant of the fused EEG+audio window autoencoder.
 */

#include "FusedWindowAutoencoder.hpp"

#include <stdexcept>

#include "AutoencoderBuilders.hpp"

FusedWindowAutoencoder::FusedWindowAutoencoder(const AutoencoderConfig& cfg)
    : eeg_encoder_(
          [&cfg]()
          {
              auto branch_cfg = cfg;
              if (!cfg.branch_encoder_layer_spec.empty())
              {
                  branch_cfg.encoder_layer_spec = cfg.branch_encoder_layer_spec;
              }
              return autoencoderRunner::autoencoders::build_ann_encoder(branch_cfg,
                  cfg.eeg_features,
                  autoencoderRunner::autoencoders::resolved_branch_hidden_size(cfg));
          }()),
      audio_encoder_(
          [&cfg]()
          {
              auto branch_cfg = cfg;
              if (!cfg.branch_encoder_layer_spec.empty())
              {
                  branch_cfg.encoder_layer_spec = cfg.branch_encoder_layer_spec;
              }
              return autoencoderRunner::autoencoders::build_ann_encoder(branch_cfg,
                  cfg.audio_features,
                  autoencoderRunner::autoencoders::resolved_branch_hidden_size(cfg));
          }()),
      fusion_encoder_(
          [&cfg]()
          {
              auto fusion_cfg = cfg;
              if (!cfg.fusion_encoder_layer_spec.empty())
              {
                  fusion_cfg.encoder_layer_spec = cfg.fusion_encoder_layer_spec;
              }
              return autoencoderRunner::autoencoders::build_ann_encoder(fusion_cfg,
                  cfg.latent_size * 2,
                  autoencoderRunner::autoencoders::resolved_fusion_hidden_size(cfg));
          }()),
      fusion_decoder_(
          [&cfg]()
          {
              auto fusion_cfg = cfg;
              if (!cfg.fusion_decoder_layer_spec.empty())
              {
                  fusion_cfg.decoder_layer_spec = cfg.fusion_decoder_layer_spec;
              }
              return autoencoderRunner::autoencoders::build_ann_decoder(fusion_cfg,
                  cfg.latent_size * 2,
                  autoencoderRunner::autoencoders::resolved_fusion_hidden_size(cfg));
          }()),
      eeg_decoder_(
          [&cfg]()
          {
              auto branch_cfg = cfg;
              if (!cfg.branch_decoder_layer_spec.empty())
              {
                  branch_cfg.decoder_layer_spec = cfg.branch_decoder_layer_spec;
              }
              return autoencoderRunner::autoencoders::build_ann_decoder(branch_cfg,
                  cfg.eeg_features,
                  autoencoderRunner::autoencoders::resolved_branch_hidden_size(cfg));
          }()),
      audio_decoder_(
          [&cfg]()
          {
              auto branch_cfg = cfg;
              if (!cfg.branch_decoder_layer_spec.empty())
              {
                  branch_cfg.decoder_layer_spec = cfg.branch_decoder_layer_spec;
              }
              return autoencoderRunner::autoencoders::build_ann_decoder(branch_cfg,
                  cfg.audio_features,
                  autoencoderRunner::autoencoders::resolved_branch_hidden_size(cfg));
          }()),
      eeg_features_(cfg.eeg_features),
      audio_features_(cfg.audio_features)
{
    if (eeg_features_ <= 0 || audio_features_ <= 0)
    {
        throw std::invalid_argument(
            "FusedWindowAutoencoder requires positive eeg_features and audio_features");
    }
}

auto FusedWindowAutoencoder::encode(const Tensor& input, bool requires_grad) -> Tensor
{
    auto eeg = autoencoderRunner::autoencoders::slice_columns(input, 0, eeg_features_);
    auto audio =
        autoencoderRunner::autoencoders::slice_columns(input, eeg_features_, audio_features_);

    auto eeg_latent = eeg_encoder_.forward(eeg, requires_grad);
    auto audio_latent = audio_encoder_.forward(audio, requires_grad);
    auto fused = autoencoderRunner::autoencoders::concat_columns(eeg_latent, audio_latent);
    return fusion_encoder_.forward(fused, requires_grad);
}

auto FusedWindowAutoencoder::decode(const Tensor& latent, bool requires_grad) -> Tensor
{
    auto fused = fusion_decoder_.forward(latent, requires_grad);
    const int branch_cols = fused.cols() / 2;
    auto eeg_branch = autoencoderRunner::autoencoders::slice_columns(fused, 0, branch_cols);
    auto audio_branch = autoencoderRunner::autoencoders::slice_columns(
        fused, branch_cols, fused.cols() - branch_cols);
    auto eeg_reconstruction = eeg_decoder_.forward(eeg_branch, requires_grad);
    auto audio_reconstruction = audio_decoder_.forward(audio_branch, requires_grad);
    return autoencoderRunner::autoencoders::concat_columns(
        eeg_reconstruction, audio_reconstruction);
}

auto FusedWindowAutoencoder::forward(const Tensor& input, bool requires_grad) -> Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto FusedWindowAutoencoder::backward(const Tensor& grad_output) -> Tensor
{
    auto eeg_grad = autoencoderRunner::autoencoders::slice_columns(grad_output, 0, eeg_features_);
    auto audio_grad =
        autoencoderRunner::autoencoders::slice_columns(grad_output, eeg_features_, audio_features_);

    auto eeg_branch_grad = eeg_decoder_.backward(eeg_grad);
    auto audio_branch_grad = audio_decoder_.backward(audio_grad);
    auto fused_branch_grad =
        autoencoderRunner::autoencoders::concat_columns(eeg_branch_grad, audio_branch_grad);
    auto latent_grad = fusion_decoder_.backward(fused_branch_grad);
    auto fused_encoder_grad = fusion_encoder_.backward(latent_grad);

    auto eeg_encoder_grad = autoencoderRunner::autoencoders::slice_columns(
        fused_encoder_grad, 0, eeg_branch_grad.cols());
    auto audio_encoder_grad = autoencoderRunner::autoencoders::slice_columns(
        fused_encoder_grad, eeg_branch_grad.cols(), audio_branch_grad.cols());

    auto eeg_input_grad = eeg_encoder_.backward(eeg_encoder_grad);
    auto audio_input_grad = audio_encoder_.backward(audio_encoder_grad);
    return autoencoderRunner::autoencoders::concat_columns(eeg_input_grad, audio_input_grad);
}

auto FusedWindowAutoencoder::params() -> std::span<Tensor*>
{
    param_ptrs_.clear();
    auto a = eeg_encoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), a.begin(), a.end());
    auto b = audio_encoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), b.begin(), b.end());
    auto c = fusion_encoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), c.begin(), c.end());
    auto d = fusion_decoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), d.begin(), d.end());
    auto e = eeg_decoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), e.begin(), e.end());
    auto f = audio_decoder_.params();
    param_ptrs_.insert(param_ptrs_.end(), f.begin(), f.end());
    return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
}
