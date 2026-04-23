#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace comparative_autoencoder_experiment
{

struct ComparativeConfig
{
    std::string dataset_root = ".";
    std::string results_dir = "results";
    std::string run_tag = "snn_lstm_compare";

    std::uint32_t seed = 1337u;
    int repeats = 3;
    bool seed_deterministic = true;
    bool check_determinism = false;

    int window_size = 128;
    int batch_size = 8;
    int max_train_samples = 512;
    int max_val_samples = 128;

    int epochs = 100;
    int early_stop_patience = 20;
    float learning_rate = 1e-3f;
    float anomaly_tau = 0.25f;

    int hidden_size = 64;
    int latent_size = 16;

    std::vector<std::string> datasets = {"fsdd"};
    std::vector<std::string> encodings = {"direct", "poisson", "latency"};
    std::vector<std::string> snn_architectures = {"dense", "conv1d", "recurrent"};
    std::vector<int> layers = {1, 2, 3};
    std::vector<float> v_th_values = {0.5f, 1.0f, 1.5f};
    std::vector<float> alpha_values = {0.8f, 0.9f, 0.99f};
};

} // namespace comparative_autoencoder_experiment
