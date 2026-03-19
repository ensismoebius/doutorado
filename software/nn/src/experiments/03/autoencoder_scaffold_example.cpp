/**
 * @file autoencoder_scaffold_example.cpp
 * @brief Minimal usage example for the experiment03 autoencoder scaffold models.
 *
 * This file shows how to construct each of the 8 autoencoder variants, run a
 * forward pass on synthetic data, and (for SNN models) reset state between
 * independent sequences.  No real dataset loading is performed here.
 *
 * Build target: add this file to EXPERIMENT03_LIB_SOURCES _or_ compile it as a
 * standalone executable that links experiment03_lib.
 */

#include <iostream>

#include "AudioWindowAutoencoder.hpp"
#include "AudioWindowSpikingAutoencoder.hpp"
#include "AutoencoderConfig.hpp"
#include "EegWindowAutoencoder.hpp"
#include "EegWindowSpikingAutoencoder.hpp"
#include "FusedWindowAutoencoder.hpp"
#include "FusedWindowSpikingAutoencoder.hpp"
#include "ProtocolAutoencoder.hpp"
#include "ProtocolSpikingAutoencoder.hpp"
#include "nn/tensor/Tensor.hpp"

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
    cfg.time_step = 1.0F;
    cfg.resistance = 1.0F;
    cfg.capacitance = 1.0F;

    // -----------------------------------------------------------------------
    // ANN autoencoders
    // -----------------------------------------------------------------------
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

    {
        AutoencoderConfig c = cfg;
        c.input_features = kFusedFeatures;
        FusedWindowAutoencoder model(c);

        // Caller concatenates EEG and audio feature tensors before passing in.
        nn::Tensor x_eeg = nn::Tensor::rand(kBatch, kEegFeatures);
        nn::Tensor x_audio = nn::Tensor::rand(kBatch, kAudioFeatures);
        // Simple horizontal concatenation via Eigen block ops:
        nn::Tensor x(kBatch, kFusedFeatures);
        for (int r = 0; r < kBatch; ++r)
        {
            for (int col = 0; col < kEegFeatures; ++col) x(r, col) = x_eeg(r, col);
            for (int col = 0; col < kAudioFeatures; ++col)
                x(r, kEegFeatures + col) = x_audio(r, col);
        }

        nn::Tensor latent = model.encode(x);
        nn::Tensor recon = model.decode(latent);

        std::cout << "[ANN] FusedWindowAutoencoder "
                  << "input=" << x.rows() << "×" << x.cols() << "  latent=" << latent.rows() << "×"
                  << latent.cols() << "  recon=" << recon.rows() << "×" << recon.cols() << "\n";
    }

    // -----------------------------------------------------------------------
    // SNN autoencoders
    // -----------------------------------------------------------------------
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

    {
        AutoencoderConfig c = cfg;
        c.input_features = kFusedFeatures;
        FusedWindowSpikingAutoencoder model(c);

        nn::Tensor x = nn::Tensor::rand(kBatch, kFusedFeatures);
        nn::Tensor latent = model.encode(x);
        nn::Tensor recon = model.decode(latent);

        std::cout << "[SNN] FusedWindowSpikingAutoencoder "
                  << "latent=" << latent.rows() << "×" << latent.cols() << "\n";
        model.reset_state();
    }

    return 0;
}
