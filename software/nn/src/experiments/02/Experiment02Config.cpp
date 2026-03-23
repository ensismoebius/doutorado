#include "Experiment02Config.hpp"

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

// Get all available discrete wavelet families from core library
auto get_available_wavelet_families() -> std::vector<std::string>
{
    return {"Haar",
        "Daub4",
        "Daub6",
        "Daub8",
        "Daub10",
        "Daub12",
        "Daub14",
        "Daub16",
        "Daub18",
        "Daub20",
        "Daub22",
        "Daub24",
        "Daub26",
        "Daub28",
        "Daub30",
        "Daub32",
        "Daub34",
        "Daub36",
        "Daub38",
        "Daub40",
        "Daub42",
        "Daub44",
        "Daub46"};
}

// Load config from spec.json
auto load_experiment_config(const std::string& spec_path) -> ExperimentConfig
{
    // read file + parse
    std::ifstream in(spec_path);
    if (!in)
    {
        throw std::runtime_error("Unable to open experiment spec: " + spec_path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    nlohmann::json config;
    try
    {
        config = nlohmann::json::parse(ss.str());
    }
    catch (const nlohmann::json::parse_error& e)
    {
        throw std::runtime_error(std::string("Experiment spec parse error: ") + e.what());
    }

    ExperimentConfig exp_config;

    exp_config.id = config.at("experiment").at("id").get<std::string>();

    // Keep backward compatibility with both old and nested determinism layouts.
    if (config.contains("determinism") && config["determinism"].contains("random_seed"))
    {
        exp_config.random_seed = config["determinism"]["random_seed"].get<int>();
    }
    else
    {
        exp_config.random_seed =
            config.at("experiment").at("determinism").at("random_seed").get<int>();
    }

    exp_config.eeg_sampling_rate = config.at("data").at("sampling_rate").at("EEG").get<int>();
    exp_config.audio_sampling_rate = config.at("data").at("sampling_rate").at("Audio").get<int>();

    exp_config.window_duration_sec =
        config.at("data").at("segmentation").at("window").at("duration_sec").get<double>();
    exp_config.overlap_sec =
        config.at("data").at("segmentation").at("window").at("overlap_sec").get<double>();

    int window_samples =
        static_cast<int>(exp_config.window_duration_sec * exp_config.eeg_sampling_rate);
    exp_config.max_decomposition_depth = static_cast<int>(std::floor(std::log2(window_samples)));

    exp_config.wavelet_families = get_available_wavelet_families();

    exp_config.normalization_range = {0.0, 1.0};
    exp_config.paraconsistent_enabled =
        config.at("paraconsistent_metrics").at("enabled").get<bool>();
    exp_config.k_folds = config.at("validation_protocol").at("k").get<int>();

    return exp_config;
}
