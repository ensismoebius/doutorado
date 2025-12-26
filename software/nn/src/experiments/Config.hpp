#pragma once

#include <yaml-cpp/yaml.h>

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
    int sampling_rate;
    int eeg_sampling_rate;

    // -------------------- paraconsistent --------------------
    bool enabled;
    std::vector<double> optimal_point; // size == 2

    /**
     * @brief Load and validate configuration from YAML file.
     * @param path Path to YAML configuration file.
     * @return std::optional<Config> Valid config or empty on failure.
     */
    static auto load(const std::string& path) -> std::optional<Config>;
};
