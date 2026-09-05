/**
 * @file src/demos/cppDemos/autoencoder_leakyrelu/autoencoder_scaffold_example.cpp
 * @brief Minimal usage example for the autoencoderRunner autoencoder scaffold models.
 *
 * This file shows how to construct each of the 8 autoencoder variants, run a
 * forward pass on synthetic data, and (for SNN models) reset state between
 * independent sequences.  No real dataset loading is performed here.
 *
 * Build target: add this file to EXPERIMENT03_LIB_SOURCES _or_ compile it as a
 * standalone executable that links autoencoderRunner_lib.
 */

#include <iostream>

#include "models/autoencoder/AudioWindowAutoencoder.hpp"
#include "models/autoencoder/AudioWindowSpikingAutoencoder.hpp"
#include "models/autoencoder/AutoencoderConfig.hpp"
#include "models/autoencoder/EegWindowAutoencoder.hpp"
#include "models/autoencoder/EegWindowSpikingAutoencoder.hpp"
#include "models/autoencoder/FusedWindowAutoencoder.hpp"
#include "models/autoencoder/FusedWindowSpikingAutoencoder.hpp"
#include "models/autoencoder/ProtocolAutoencoder.hpp"
#include "models/autoencoder/ProtocolSpikingAutoencoder.hpp"
#include "tensor/Tensor.hpp"

using nn::models::autoencoder::AudioWindowAutoencoder;
using nn::models::autoencoder::AudioWindowSpikingAutoencoder;
using nn::models::autoencoder::AutoencoderConfig;
using nn::models::autoencoder::EegWindowAutoencoder;
using nn::models::autoencoder::EegWindowSpikingAutoencoder;
using nn::models::autoencoder::FusedWindowAutoencoder;
using nn::models::autoencoder::FusedWindowSpikingAutoencoder;
using nn::models::autoencoder::ProtocolAutoencoder;
using nn::models::autoencoder::ProtocolSpikingAutoencoder;

namespace
{

void demo_protocol_autoencoder(const AutoencoderConfig& cfg, int kBatch, int kProtocolFeatures)
{
    AutoencoderConfig c = cfg;
    c.input_features = kProtocolFeatures;
    ProtocolAutoencoder model(c);

    nn::Tensor x = nn::Tensor::rand(kBatch, kProtocolFeatures);
    nn::Tensor latent = model.encode(x);
    nn::Tensor recon = model.decode(latent);
    nn::Tensor recon2 = model.forward(x);

    std::cout << "[ANN] ProtocolAutoencoder  "
              << "input=" << x.rows() << "×" << x.cols() << "  latent=" << latent.rows() << "×"
              << latent.cols() << "  recon=" << recon.rows() << "×" << recon.cols() << "\n";
}

void demo_eeg_window_autoencoder(const AutoencoderConfig& cfg, int kBatch, int kEegFeatures)
{
    AutoencoderConfig c = cfg;
    c.input_features = kEegFeatures;
    EegWindowAutoencoder model(c);

    nn::Tensor x = nn::Tensor::rand(kBatch, kEegFeatures);
    nn::Tensor latent = model.encode(x);
    nn::Tensor recon = model.decode(latent);

    std::cout << "[ANN] EegWindowAutoencoder "
              << "input=" << x.rows() << "×" << x.cols() << "  latent=" << latent.rows() << "×"
              << latent.cols() << "  recon=" << recon.rows() << "×" << recon.cols() << "\n";
}

void demo_audio_window_autoencoder(const AutoencoderConfig& cfg, int kBatch, int kAudioFeatures)
{
    AutoencoderConfig c = cfg;
    c.input_features = kAudioFeatures;
    AudioWindowAutoencoder model(c);

    nn::Tensor x = nn::Tensor::rand(kBatch, kAudioFeatures);
    nn::Tensor latent = model.encode(x);
    nn::Tensor recon = model.decode(latent);

    std::cout << "[ANN] AudioWindowAutoencoder "
              << "input=" << x.rows() << "×" << x.cols() << "  latent=" << latent.rows() << "×"
              << latent.cols() << "  recon=" << recon.rows() << "×" << recon.cols() << "\n";
}

void demo_fused_window_autoencoder(const AutoencoderConfig& cfg,
    int kBatch,
    int kEegFeatures,
    int kAudioFeatures,
    int kFusedFeatures)
{
    AutoencoderConfig c = cfg;
    c.input_features = kFusedFeatures;
    c.eeg_features = kEegFeatures;
    c.audio_features = kAudioFeatures;
    FusedWindowAutoencoder model(c);

    // Caller concatenates EEG and audio feature tensors before passing in.
    nn::Tensor x_eeg = nn::Tensor::rand(kBatch, kEegFeatures);
    nn::Tensor x_audio = nn::Tensor::rand(kBatch, kAudioFeatures);
    // Simple horizontal concatenation via Eigen block ops:
    nn::Tensor x(kBatch, kFusedFeatures);
    for (int r = 0; r < kBatch; ++r)
    {
        for (int col = 0; col < kEegFeatures; ++col) x(r, col) = x_eeg(r, col);
        for (int col = 0; col < kAudioFeatures; ++col) x(r, kEegFeatures + col) = x_audio(r, col);
    }

    nn::Tensor latent = model.encode(x);
    nn::Tensor recon = model.decode(latent);

    std::cout << "[ANN] FusedWindowAutoencoder "
              << "input=" << x.rows() << "×" << x.cols() << "  latent=" << latent.rows() << "×"
              << latent.cols() << "  recon=" << recon.rows() << "×" << recon.cols() << "\n";
}

void demo_protocol_spiking_autoencoder(
    const AutoencoderConfig& cfg, int kBatch, int kProtocolFeatures)
{
    AutoencoderConfig c = cfg;
    c.input_features = kProtocolFeatures;
    ProtocolSpikingAutoencoder model(c);

    // Simulate two time steps (e.g. two windows of the same trial).
    for (int t = 0; t < 2; ++t)
    {
        nn::Tensor x = nn::Tensor::rand(kBatch, kProtocolFeatures);
        nn::Tensor latent = model.encode(x);
        nn::Tensor recon = model.decode(latent);
        std::cout << "[SNN] ProtocolSpikingAutoencoder t=" << t << " latent=" << latent.rows()
                  << "×" << latent.cols() << "\n";
    }
    model.reset_state(); // reset between sequences/trials
}

void demo_eeg_window_spiking_autoencoder(const AutoencoderConfig& cfg, int kBatch, int kEegFeatures)
{
    AutoencoderConfig c = cfg;
    c.input_features = kEegFeatures;
    EegWindowSpikingAutoencoder model(c);

    nn::Tensor x = nn::Tensor::rand(kBatch, kEegFeatures);
    nn::Tensor latent = model.encode(x);
    nn::Tensor recon = model.decode(latent);

    std::cout << "[SNN] EegWindowSpikingAutoencoder "
              << "latent=" << latent.rows() << "×" << latent.cols() << "\n";
    model.reset_state();
}

void demo_audio_window_spiking_autoencoder(
    const AutoencoderConfig& cfg, int kBatch, int kAudioFeatures)
{
    AutoencoderConfig c = cfg;
    c.input_features = kAudioFeatures;
    AudioWindowSpikingAutoencoder model(c);

    nn::Tensor x = nn::Tensor::rand(kBatch, kAudioFeatures);
    nn::Tensor latent = model.encode(x);
    nn::Tensor recon = model.decode(latent);

    std::cout << "[SNN] AudioWindowSpikingAutoencoder "
              << "latent=" << latent.rows() << "×" << latent.cols() << "\n";
    model.reset_state();
}

void demo_fused_window_spiking_autoencoder(const AutoencoderConfig& cfg,
    int kBatch,
    int kEegFeatures,
    int kAudioFeatures,
    int kFusedFeatures)
{
    AutoencoderConfig c = cfg;
    c.input_features = kFusedFeatures;
    c.eeg_features = kEegFeatures;
    c.audio_features = kAudioFeatures;
    FusedWindowSpikingAutoencoder model(c);

    nn::Tensor x = nn::Tensor::rand(kBatch, kFusedFeatures);
    nn::Tensor latent = model.encode(x);
    nn::Tensor recon = model.decode(latent);

    std::cout << "[SNN] FusedWindowSpikingAutoencoder "
              << "latent=" << latent.rows() << "×" << latent.cols() << "\n";
    model.reset_state();
}

} // namespace

int main()
{
    constexpr int kBatch = 4;
    constexpr int kEegFeatures = 64;
    constexpr int kAudioFeatures = 48;
    constexpr int kFusedFeatures = kEegFeatures + kAudioFeatures;
    constexpr int kProtocolFeatures = 128;

    // -----------------------------------------------------------------------
    // Shared config for all models.
    // -----------------------------------------------------------------------
    AutoencoderConfig cfg;
    cfg.input_features = kEegFeatures; // overridden per model below
    cfg.hidden_size = 64;
    cfg.latent_size = 32;
    cfg.depth = 2;
    cfg.delta_t = 1.0F;
    cfg.resistance = 1.0F;
    cfg.capacitance = 1.0F;

    // -----------------------------------------------------------------------
    // ANN autoencoders
    // -----------------------------------------------------------------------
    demo_protocol_autoencoder(cfg, kBatch, kProtocolFeatures);
    demo_eeg_window_autoencoder(cfg, kBatch, kEegFeatures);
    demo_audio_window_autoencoder(cfg, kBatch, kAudioFeatures);
    demo_fused_window_autoencoder(cfg, kBatch, kEegFeatures, kAudioFeatures, kFusedFeatures);

    // -----------------------------------------------------------------------
    // SNN autoencoders
    // -----------------------------------------------------------------------
    demo_protocol_spiking_autoencoder(cfg, kBatch, kProtocolFeatures);
    demo_eeg_window_spiking_autoencoder(cfg, kBatch, kEegFeatures);
    demo_audio_window_spiking_autoencoder(cfg, kBatch, kAudioFeatures);
    demo_fused_window_spiking_autoencoder(
        cfg, kBatch, kEegFeatures, kAudioFeatures, kFusedFeatures);

    return 0;
}
