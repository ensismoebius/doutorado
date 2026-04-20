#pragma once

/**
 * @file Experiment04Config.hpp
 * @brief Runtime configuration for Experiment04 (LSTM Autoencoder).
 *
 * Mirrors the experiment03 profile-first workflow while keeping the LSTM
 * autoencoder settings flat and explicit.
 */

#include <string>

struct Experiment04Config
{
    std::string program_device = "cpu";

    std::string dataset_root = ".";
    std::string subject_regex = ".*";
    std::string dataset_type = "audio-window";

    int batch_size = 1;
    int max_batches_per_epoch = 0;
    unsigned sampler_shuffle_seed = 42u;

    int input_size = 64;
    int seq_len = 32;
    int hidden_size = 128;
    int latent_size = 16;
    int num_layers = 1;

    int epochs = 30;
    float learning_rate = 1e-3f;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_epsilon = 1e-8f;

    float grad_clip_norm = 1.0f;

    std::string results_dir;
    std::string run_tag = "experiment04";
};