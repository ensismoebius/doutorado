#ifndef NN_EXPERIMENTS_02_EXPERIMENT02CONFIG_HPP
#define NN_EXPERIMENTS_02_EXPERIMENT02CONFIG_HPP

#include <string>
#include <vector>

struct ExperimentConfig
{
    std::string id = "M1_WaveletPacket_Baseline";
    int random_seed = 42;
    bool enforce_single_thread = false;
    bool allow_parallelism = true;
    double numerical_tolerance = 1e-9;

    std::vector<std::string> modalities = {"EEG", "Audio"};
    int eeg_sampling_rate = 1000;
    int audio_sampling_rate = 44100;

    std::string window_type = "hanning";
    double window_duration_sec = 1.5;
    double overlap_sec = 0.5;

    std::vector<std::string> wavelet_families;
    int max_decomposition_depth = 0;

    std::string feature_extraction = "subband_energy";
    std::string normalization_method = "min_max";
    std::vector<double> normalization_range = {0.0, 1.0};

    bool paraconsistent_enabled = true;
    std::vector<std::string> paraconsistent_metrics = {"alpha", "beta", "G1", "G2"};

    std::string classifier_model = "ResNet";
    std::string classifier_paradigm = "spiking_neural_network";
    int embedding_dim = 128;
    int resnet_depth = 3;

    int k_folds = 10;
    bool stratified = true;
    bool shuffle = true;

    std::string output_dir = "results";
    std::vector<std::string> output_formats = {"csv"};
};

auto get_available_wavelet_families() -> std::vector<std::string>;
auto load_experiment_config(const std::string& spec_path) -> ExperimentConfig;

#endif // NN_EXPERIMENTS_02_EXPERIMENT02CONFIG_HPP
