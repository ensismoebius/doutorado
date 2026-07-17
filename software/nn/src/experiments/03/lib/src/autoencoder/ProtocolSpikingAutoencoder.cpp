/**
 * @file src/experiments/03/lib/src/autoencoder/ProtocolSpikingAutoencoder.cpp
 * @brief Spiking autoencoder implementation for protocol-formatted inputs.
 *
 * Builds SNN encoder and decoder stacks using `Lif` and
 * `LifIntegrator` layers and exposes the `ProtocolSpikingAutoencoder`
 * wrapper that conforms to the project's `Module` interface.
 */

#include "ProtocolSpikingAutoencoder.hpp"

#include <algorithm>
#include <cmath>

#include "AutoencoderBuilders.hpp"
#include "layers/Layers.hpp"

namespace
{
// Indices of Lif layers within a Sequential, in construction order. Computed once
// since encoder structure is static after the constructor builds it.
auto find_lif_layer_indices(const nn::Sequential& seq) -> std::vector<size_t>
{
    std::vector<size_t> indices;
    for (size_t i = 0; i < seq.layers.size(); ++i)
        if (dynamic_cast<nn::Lif*>(seq.layers[i].get()) != nullptr) indices.push_back(i);
    return indices;
}

// Backward through `seq`, injecting a firing-rate band-penalty gradient into the
// spike output of each Lif layer listed in `lif_indices` before propagating through
// it. Mirrors E05DsnnClassifier::add_firing_rate_grad's math exactly:
//   reg = lambda * (max(0, fr_min - r)^2 + max(0, r - fr_max)^2)
//   d_reg/d_spike = 2*lambda*(r - clamp(r, fr_min, fr_max)) / n
// Inert (identity to plain Sequential::backward) when lambda <= 0.
auto backward_with_firing_rate_reg(nn::Sequential& seq,
    const std::vector<size_t>& lif_indices,
    float lambda,
    float fr_min,
    float fr_max,
    const ProtocolSpikingAutoencoder::Tensor& grad_output) -> ProtocolSpikingAutoencoder::Tensor
{
    ProtocolSpikingAutoencoder::Tensor grad = grad_output;
    for (size_t i = seq.layers.size(); i-- > 0;)
    {
        if (lambda > 0.0F && i < seq.outputs.size() &&
            std::find(lif_indices.begin(), lif_indices.end(), i) != lif_indices.end())
        {
            const auto& spikes = seq.outputs[i];
            if (spikes.size() > 0)
            {
                const float n = static_cast<float>(spikes.size());
                const float r = spikes.sum() / n;
                const float clamped = std::clamp(r, fr_min, fr_max);
                const float d_reg = 2.0F * lambda * (r - clamped) / n;
                if (d_reg != 0.0F) grad.add_scalar_inplace(d_reg);
            }
        }
        grad = seq.layers[i]->backward(grad);
    }
    return grad;
}
} // namespace

ProtocolSpikingAutoencoder::ProtocolSpikingAutoencoder(const AutoencoderConfig& cfg)
    : fr_lambda_(cfg.firing_rate_reg_lambda),
      fr_min_(cfg.firing_rate_min),
      fr_max_(cfg.firing_rate_max)
{
    use_dual_branch_ = cfg.architecture == AutoencoderArchitecture::DualBranchFusion &&
                       cfg.eeg_features > 0 && cfg.audio_features > 0 &&
                       (cfg.eeg_features + cfg.audio_features == cfg.input_features);

    if (use_dual_branch_)
    {
        eeg_features_ = cfg.eeg_features;
        audio_features_ = cfg.audio_features;

        eeg_encoder_ = experiment03::autoencoders::build_snn_encoder(
            cfg, cfg.eeg_features, experiment03::autoencoders::resolved_branch_hidden_size(cfg));
        audio_encoder_ = experiment03::autoencoders::build_snn_encoder(
            cfg, cfg.audio_features, experiment03::autoencoders::resolved_branch_hidden_size(cfg));
        fusion_encoder_ = experiment03::autoencoders::build_snn_encoder(
            cfg, cfg.latent_size * 2, experiment03::autoencoders::resolved_fusion_hidden_size(cfg));
        fusion_decoder_ = experiment03::autoencoders::build_snn_decoder(
            cfg, cfg.latent_size * 2, experiment03::autoencoders::resolved_fusion_hidden_size(cfg));
        eeg_decoder_ = experiment03::autoencoders::build_snn_decoder(
            cfg, cfg.eeg_features, experiment03::autoencoders::resolved_branch_hidden_size(cfg));
        audio_decoder_ = experiment03::autoencoders::build_snn_decoder(
            cfg, cfg.audio_features, experiment03::autoencoders::resolved_branch_hidden_size(cfg));
    }
    else
    {
        encoder_ =
            experiment03::autoencoders::build_snn_encoder(cfg, cfg.input_features, cfg.hidden_size);
        decoder_ =
            experiment03::autoencoders::build_snn_decoder(cfg, cfg.input_features, cfg.hidden_size);
    }

    if (use_dual_branch_)
    {
        eeg_encoder_lif_indices_ = find_lif_layer_indices(eeg_encoder_);
        audio_encoder_lif_indices_ = find_lif_layer_indices(audio_encoder_);
        fusion_encoder_lif_indices_ = find_lif_layer_indices(fusion_encoder_);
    }
    else
    {
        encoder_lif_indices_ = find_lif_layer_indices(encoder_);
    }
}

auto ProtocolSpikingAutoencoder::encode(const Tensor& input, bool requires_grad) -> Tensor
{
    if (!use_dual_branch_)
    {
        return encoder_.forward(input, requires_grad);
    }

    auto eeg = experiment03::autoencoders::slice_columns(input, 0, eeg_features_);
    auto audio = experiment03::autoencoders::slice_columns(input, eeg_features_, audio_features_);
    auto eeg_latent = eeg_encoder_.forward(eeg, requires_grad);
    auto audio_latent = audio_encoder_.forward(audio, requires_grad);
    auto fused = experiment03::autoencoders::concat_columns(eeg_latent, audio_latent);
    return fusion_encoder_.forward(fused, requires_grad);
}

auto ProtocolSpikingAutoencoder::decode(const Tensor& latent, bool requires_grad) -> Tensor
{
    if (!use_dual_branch_)
    {
        return decoder_.forward(latent, requires_grad);
    }

    auto fused = fusion_decoder_.forward(latent, requires_grad);
    const int branch_cols = fused.cols() / 2;
    auto eeg_branch = experiment03::autoencoders::slice_columns(fused, 0, branch_cols);
    auto audio_branch =
        experiment03::autoencoders::slice_columns(fused, branch_cols, fused.cols() - branch_cols);
    auto eeg_reconstruction = eeg_decoder_.forward(eeg_branch, requires_grad);
    auto audio_reconstruction = audio_decoder_.forward(audio_branch, requires_grad);
    return experiment03::autoencoders::concat_columns(eeg_reconstruction, audio_reconstruction);
}

auto ProtocolSpikingAutoencoder::forward(const Tensor& input, bool requires_grad) -> Tensor
{
    return decode(encode(input, requires_grad), requires_grad);
}

auto ProtocolSpikingAutoencoder::backward(const Tensor& grad_output) -> Tensor
{
    if (!use_dual_branch_)
    {
        Tensor grad = decoder_.backward(grad_output);
        return backward_with_firing_rate_reg(
            encoder_, encoder_lif_indices_, fr_lambda_, fr_min_, fr_max_, grad);
    }

    auto eeg_grad = experiment03::autoencoders::slice_columns(grad_output, 0, eeg_features_);
    auto audio_grad =
        experiment03::autoencoders::slice_columns(grad_output, eeg_features_, audio_features_);
    auto eeg_branch_grad = eeg_decoder_.backward(eeg_grad);
    auto audio_branch_grad = audio_decoder_.backward(audio_grad);
    auto fused_branch_grad =
        experiment03::autoencoders::concat_columns(eeg_branch_grad, audio_branch_grad);
    auto latent_grad = fusion_decoder_.backward(fused_branch_grad);
    auto fused_encoder_grad = backward_with_firing_rate_reg(
        fusion_encoder_, fusion_encoder_lif_indices_, fr_lambda_, fr_min_, fr_max_, latent_grad);
    auto eeg_encoder_grad =
        experiment03::autoencoders::slice_columns(fused_encoder_grad, 0, eeg_branch_grad.cols());
    auto audio_encoder_grad = experiment03::autoencoders::slice_columns(
        fused_encoder_grad, eeg_branch_grad.cols(), audio_branch_grad.cols());
    auto eeg_input_grad = backward_with_firing_rate_reg(
        eeg_encoder_, eeg_encoder_lif_indices_, fr_lambda_, fr_min_, fr_max_, eeg_encoder_grad);
    auto audio_input_grad = backward_with_firing_rate_reg(audio_encoder_,
        audio_encoder_lif_indices_,
        fr_lambda_,
        fr_min_,
        fr_max_,
        audio_encoder_grad);
    return experiment03::autoencoders::concat_columns(eeg_input_grad, audio_input_grad);
}

auto ProtocolSpikingAutoencoder::params() -> std::span<Tensor*>
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

void ProtocolSpikingAutoencoder::reset_state()
{
    if (!use_dual_branch_)
    {
        experiment03::autoencoders::reset_sequential_state(encoder_);
        experiment03::autoencoders::reset_sequential_state(decoder_);
        return;
    }

    experiment03::autoencoders::reset_sequential_state(eeg_encoder_);
    experiment03::autoencoders::reset_sequential_state(audio_encoder_);
    experiment03::autoencoders::reset_sequential_state(fusion_encoder_);
    experiment03::autoencoders::reset_sequential_state(fusion_decoder_);
    experiment03::autoencoders::reset_sequential_state(eeg_decoder_);
    experiment03::autoencoders::reset_sequential_state(audio_decoder_);
}
