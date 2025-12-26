#include <yaml-cpp/yaml.h>

/**
 * @brief Represents the global configuration for experiments, loaded from a YAML file.
 *
 * This struct holds various parameters controlling aspects of windowing, normalization,
 * classifier, dataset, and paraconsistent analysis, all parsed from a structured
 * YAML configuration file.
 */
struct Config
{
    /// @brief Duration of the window in seconds, from `window.duration_sec`.
    double duration_sec;
    /// @brief Overlap percentage for windowing, from `window.overlap_percent`.
    int overlap_percent;
    /// @brief Normalization range (min, max) for data, from `normalization.range`.
    std::vector<double> range;
    /// @brief Normalization method to be applied, from `normalization.method`.
    std::string method;
    /// @brief Flag indicating if paraconsistent prerequisite normalization is enabled, from `normalization.paraconsistent_prerequisite`.
    bool paraconsistent_prerequisite;
    /// @brief Type of classifier used, from `classifier.type`.
    std::string type;
    /// @brief Implementation details or specific model for the classifier, from `classifier.implementation`.
    std::string implementation;
    /// @brief Sampling rate of the dataset, from `dataset.sampling_rate`.
    int sampling_rate;
    /// @brief EEG specific sampling rate of the dataset, from `dataset.eeg_sampling_rate`.
    int eeg_sampling_rate;
    /// @brief Flag indicating if paraconsistent analysis is enabled, from `paraconsistent.enabled`.
    bool enabled;
    /// @brief Optimal point for paraconsistent analysis (e.g., [1.0, 0.0]), from `paraconsistent.optimal_point`.
    std::vector<double> optimal_point;

    /**
     * @brief Constructs a Config object by loading parameters from a YAML file.
     * @param path The file path to the YAML configuration file.
     *
     * This constructor parses the YAML file at the given path and populates
     * the struct's member variables with the corresponding values from the file.
     * It expects a specific structure within the YAML file to correctly
     * extract all configuration parameters.
     */
    Config(const std::string& path)
    {
        // Load the configuration file
        YAML::Node node = YAML::LoadFile(path);

        duration_sec = node["window"]["duration_sec"].as<double>();
        overlap_percent = node["window"]["overlap_percent"].as<int>();
        range = node["normalization"]["range"].as<std::vector<double>>();
        method = node["normalization"]["method"].as<std::string>();
        paraconsistent_prerequisite =
            node["normalization"]["paraconsistent_prerequisite"].as<bool>();
        type = node["classifier"]["type"].as<std::string>();
        implementation = node["classifier"]["implementation"].as<std::string>();
        sampling_rate = node["dataset"]["sampling_rate"].as<int>();
        eeg_sampling_rate = node["dataset"]["eeg_sampling_rate"].as<int>();
        enabled = node["paraconsistent"]["enabled"].as<bool>();
        optimal_point = node["paraconsistent"]["optimal_point"].as<std::vector<double>>();
    }
};