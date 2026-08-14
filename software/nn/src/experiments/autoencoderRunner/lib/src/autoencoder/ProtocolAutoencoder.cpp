/**
 * @file src/experiments/autoencoderRunner/lib/src/autoencoder/ProtocolAutoencoder.cpp
 * @brief Deterministic (ANN) autoencoder implementation for protocol inputs.
 */

#include "ProtocolAutoencoder.hpp"

#include "AutoencoderBuilders.hpp"

ProtocolAutoencoder::ProtocolAutoencoder(const AutoencoderConfig& cfg)
{
    use_dual_branch_ = cfg.architecture == AutoencoderArchitecture::DualBranchFusion &&
                       cfg.eeg_features > 0 && cfg.audio_features > 0 &&
                       (cfg.eeg_features + cfg.audio_features == cfg.input_features);

    if (use_dual_branch_)
    {
        eeg_features_ = cfg.eeg_features;
        audio_features_ = cfg.audio_features;

        eeg_encoder_ = autoencoderRunner::autoencoders::build_ann_encoder(cfg,
            cfg.eeg_features,
            autoencoderRunner::autoencoders::resolved_branch_hidden_size(cfg));
        audio_encoder_ = autoencoderRunner::autoencoders::build_ann_encoder(cfg,
            cfg.audio_features,
            autoencoderRunner::autoencoders::resolved_branch_hidden_size(cfg));
        fusion_encoder_ = autoencoderRunner::autoencoders::build_ann_encoder(cfg,
            cfg.latent_size * 2,
            autoencoderRunner::autoencoders::resolved_fusion_hidden_size(cfg));
        fusion_decoder_ = autoencoderRunner::autoencoders::build_ann_decoder(cfg,
            cfg.latent_size * 2,
            autoencoderRunner::autoencoders::resolved_fusion_hidden_size(cfg));
        eeg_decoder_ = autoencoderRunner::autoencoders::build_ann_decoder(cfg,
            cfg.eeg_features,
            autoencoderRunner::autoencoders::resolved_branch_hidden_size(cfg));
        audio_decoder_ = autoencoderRunner::autoencoders::build_ann_decoder(cfg,
            cfg.audio_features,
            autoencoderRunner::autoencoders::resolved_branch_hidden_size(cfg));
    }
    else
    {
        encoder_ = autoencoderRunner::autoencoders::build_ann_encoder(
            cfg, cfg.input_features, cfg.hidden_size);
        decoder_ = autoencoderRunner::autoencoders::build_ann_decoder(
            cfg, cfg.input_features, cfg.hidden_size);
    }
}

auto ProtocolAutoencoder::encode(const Tensor& input, bool requires_grad) -> Tensor
{
    if (!use_dual_branch_)
    {
        return encoder_.forward(input, requires_grad);
    }

    auto eeg = autoencoderRunner::autoencoders::slice_columns(input, 0, eeg_features_);
    auto audio =
        autoencoderRunner::autoencoders::slice_columns(input, eeg_features_, audio_features_);
    auto eeg_latent = eeg_encoder_.forward(eeg, requires_grad);
    auto audio_latent = audio_encoder_.forward(audio, requires_grad);
    auto fused = autoencoderRunner::autoencoders::concat_columns(eeg_latent, audio_latent);
    return fusion_encoder_.forward(fused, requires_grad);
}

auto ProtocolAutoencoder::decode(const Tensor& latent, bool requires_grad) -> Tensor
{
    if (!use_dual_branch_)
    {
        return decoder_.forward(latent, requires_grad);
    }

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

auto ProtocolAutoencoder::forward(const Tensor& input, bool requires_grad) -> Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto ProtocolAutoencoder::backward(const Tensor& grad_output) -> Tensor
{
    if (!use_dual_branch_)
    {
        Tensor grad = decoder_.backward(grad_output);
        return encoder_.backward(grad);
    }

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

auto ProtocolAutoencoder::params() -> std::span<Tensor*>
{
    param_ptrs_.clear();
    if (!use_dual_branch_)
    {
        auto ep = encoder_.params();
        param_ptrs_.insert(param_ptrs_.end(), ep.begin(), ep.end());
        auto dp = decoder_.params();
        param_ptrs_.insert(param_ptrs_.end(), dp.begin(), dp.end());
    }
    else
    {
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
    }
    return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
}

namespace
{
void merge_prefixed(std::map<std::string, ProtocolAutoencoder::Tensor>& out,
    const char* prefix,
    const nn::Sequential& seq)
{
    for (const auto& [k, v] : seq.state_dict()) out[std::string(prefix) + "." + k] = v;
}

void split_prefixed(const std::map<std::string, ProtocolAutoencoder::Tensor>& sd,
    const char* prefix,
    nn::Sequential& seq)
{
    const std::string p = std::string(prefix) + ".";
    std::map<std::string, ProtocolAutoencoder::Tensor> child;
    for (const auto& [k, v] : sd)
        if (k.rfind(p, 0) == 0) child[k.substr(p.size())] = v;
    if (!child.empty()) seq.load_state_dict(child);
}
} // namespace

// Merge every sub-Sequential's own state_dict, prefixed by member name. Whichever
// sub-Sequentials this instance did not build (dual-branch vs single encoder/decoder,
// see the constructor) are default-constructed/empty and contribute nothing — no
// use_dual_branch_ branching needed here.
auto ProtocolAutoencoder::state_dict() const -> std::map<std::string, Tensor>
{
    std::map<std::string, Tensor> out;
    merge_prefixed(out, "encoder", encoder_);
    merge_prefixed(out, "decoder", decoder_);
    merge_prefixed(out, "eeg_encoder", eeg_encoder_);
    merge_prefixed(out, "audio_encoder", audio_encoder_);
    merge_prefixed(out, "fusion_encoder", fusion_encoder_);
    merge_prefixed(out, "fusion_decoder", fusion_decoder_);
    merge_prefixed(out, "eeg_decoder", eeg_decoder_);
    merge_prefixed(out, "audio_decoder", audio_decoder_);
    return out;
}

void ProtocolAutoencoder::load_state_dict(const std::map<std::string, Tensor>& sd)
{
    split_prefixed(sd, "encoder", encoder_);
    split_prefixed(sd, "decoder", decoder_);
    split_prefixed(sd, "eeg_encoder", eeg_encoder_);
    split_prefixed(sd, "audio_encoder", audio_encoder_);
    split_prefixed(sd, "fusion_encoder", fusion_encoder_);
    split_prefixed(sd, "fusion_decoder", fusion_decoder_);
    split_prefixed(sd, "eeg_decoder", eeg_decoder_);
    split_prefixed(sd, "audio_decoder", audio_decoder_);
}
