#include "Experiment02Config.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>

// Get all available discrete wavelet families from core library
auto get_available_wavelet_families() -> std::vector<std::string>
{
    return {"Haar",   "Daub4",  "Daub6",  "Daub8",  "Daub10", "Daub12", "Daub14", "Daub16",
            "Daub18", "Daub20", "Daub22", "Daub24", "Daub26", "Daub28", "Daub30", "Daub32",
            "Daub34", "Daub36", "Daub38", "Daub40", "Daub42", "Daub44", "Daub46"};
}

// Load config from spec.yaml
auto load_experiment_config(const std::string& spec_path) -> ExperimentConfig
{
    YAML::Node config = YAML::LoadFile(spec_path);

    ExperimentConfig exp_config;

    exp_config.id = config["experiment"]["id"].as<std::string>();

    // Keep backward compatibility with both old and nested determinism layouts.
    if (config["determinism"] && config["determinism"]["random_seed"]) {
        exp_config.random_seed = config["determinism"]["random_seed"].as<int>();
    } else {
        exp_config.random_seed = config["experiment"]["determinism"]["random_seed"].as<int>();
    }

    exp_config.eeg_sampling_rate = config["data"]["sampling_rate"]["EEG"].as<int>();
    exp_config.audio_sampling_rate = config["data"]["sampling_rate"]["Audio"].as<int>();

    exp_config.window_duration_sec =
        config["data"]["segmentation"]["window"]["duration_sec"].as<double>();
    exp_config.overlap_sec = config["data"]["segmentation"]["window"]["overlap_sec"].as<double>();

    int window_samples =
        static_cast<int>(exp_config.window_duration_sec * exp_config.eeg_sampling_rate);
    exp_config.max_decomposition_depth = static_cast<int>(std::floor(std::log2(window_samples)));

    exp_config.wavelet_families = get_available_wavelet_families();

    exp_config.normalization_range = {0.0, 1.0};
    exp_config.paraconsistent_enabled = config["paraconsistent_metrics"]["enabled"].as<bool>();
    exp_config.k_folds = config["validation_protocol"]["k"].as<int>();

    return exp_config;
}
