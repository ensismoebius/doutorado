/**
 * @file Config.hpp
 * @brief YAML-backed experiment configuration (validated, immutable struct).
 *
 * This module centralizes runtime configuration for experiment executables.
 * The `Config::load()` implementation performs validation and can enforce
 * “frozen” settings for specific phases.
 */

#ifndef EXPERIMENTS_CONFIG_HPP
#define EXPERIMENTS_CONFIG_HPP

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

/**
 * @brief Immutable configuration loaded from a YAML file.
 *
 * This struct represents a validated, thread-safe configuration object.
 * No global mutable state is used.
 */
struct Config
{
    // -------------------- window --------------------
    double duration_sec;
    int overlap_percent;

    // -------------------- normalization --------------------
    std::vector<double> range; // size == 2
    std::string method;
    bool paraconsistent_prerequisite;

    // -------------------- classifier --------------------
    std::string type;
    std::string implementation;

    // -------------------- dataset --------------------
    std::string dataset_base_path;
    int sampling_rate;
    int eeg_sampling_rate;

    // -------------------- experiment --------------------
    int seed;
    bool cross_validation;
    int folds;

    // -------------------- classifier hyperparameters --------------------
    int resnet_hidden_dim;
    int resnet_depth;
    float learning_rate;
    int batch_size;
    int max_epochs;

    // -------------------- paraconsistent --------------------
    bool enabled;
    std::vector<double> optimal_point; // size == 2

    // -------------------- output --------------------
    std::string results_dir;
    std::string metrics_file;
    std::string torch_state_file;

    /**
     * @brief Load and validate configuration from YAML file.
     * @param path Path to YAML configuration file.
     * @return std::optional<Config> Valid config or empty on failure.
     */
    static auto load(const std::string& path) -> std::optional<Config>;
};

#endif // EXPERIMENTS_CONFIG_HPP
