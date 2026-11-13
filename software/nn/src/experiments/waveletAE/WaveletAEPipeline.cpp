/**
 * @file src/experiments/waveletAE/WaveletAEPipeline.cpp
 * @brief Implementation for WaveletAEpipeline.
 *

 */

#include "WaveletAEPipeline.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "WaveletAEData.hpp"
#include "WaveletAEEvaluation.hpp"
#include "WaveletAEReporting.hpp"
#include "WaveletAETraining.hpp"
#include "WaveletAEWavelets.hpp"
#include "linear_algebra/linear_algebra.hpp"
#include "logging/Logger.hpp"
#include "wavelet/waveletOperations.hpp"

namespace fs = std::filesystem;

namespace
{
} // namespace

// Normalization (using core implementation)
void min_max_normalize(
    std::vector<std::vector<double>>& features, const std::vector<double>& range = {0.0, 1.0})
{
    linearAlgebra::min_max_normalize_features(features, range);
}

auto run_wavelet_baseline_experiment(const ExperimentConfig& config) -> void
{
    NN_LOG_INFO(std::string("Starting experiment: ") + config.id);
    fs::create_directories(config.output_dir);

    for (const std::string& wavelet_name : config.wavelet_families)
    {
        NN_LOG_INFO(std::string("Processing wavelet: ") + wavelet_name);

        // The data paths come from the spec. A load failure is an error: this experiment used
        // to fall back to synthetic samples (with placeholder paths, it always did) and report
        // their metrics as if they were the data's.
        const std::vector<EEGSample> eeg_samples = load_eeg_data(config.eeg_path);
        const std::vector<AudioSample> audio_samples = load_audio_data(config.audio_path);
        NN_LOG_INFO("Loaded " + std::to_string(eeg_samples.size()) + " EEG samples and " +
                    std::to_string(audio_samples.size()) + " audio samples");

        auto windows = extract_windows(eeg_samples,
            audio_samples,
            config.window_duration_sec,
            config.overlap_sec,
            config.eeg_sampling_rate,
            config.audio_sampling_rate);
        NN_LOG_INFO("Extracted " + std::to_string(windows.size()) + " windows");

        if (windows.empty())
        {
            throw std::runtime_error("no valid windows for " + wavelet_name +
                                     ": check data.segmentation against the signal lengths");
        }

        std::vector<std::vector<double>> features;
        std::vector<int> labels;

        for (const auto& window : windows)
        {
            std::vector<double> combined_signal = window.eeg_window;
            combined_signal.insert(
                combined_signal.end(), window.audio_window.begin(), window.audio_window.end());

            auto wavelet_coeffs =
                get_wavelet_coeffs(wavelet_name, combined_signal, config.max_decomposition_depth);

            auto energies =
                wavelets::extract_subband_energies(wavelet_coeffs, config.max_decomposition_depth);
            features.push_back(energies);
            labels.push_back(window.label);
        }

        if (features.empty() || labels.empty())
        {
            throw std::runtime_error("no features produced for " + wavelet_name);
        }

        min_max_normalize(features, config.normalization_range);

        auto fold_results =
            k_fold_cross_validation(features, labels, config.k_folds, config.random_seed);

        const auto aggregated = aggregate_fold_results(fold_results);

        std::string csv_path = config.output_dir + "/" + wavelet_name + "_results.csv";
        write_wavelet_results_csv(csv_path, config, wavelet_name, aggregated);

        NN_LOG_INFO("Completed " + wavelet_name +
                    " - F1: " + std::to_string(aggregated.classification.f1_score) +
                    ", Alpha: " + std::to_string(aggregated.paraconsistent.alpha));
    }

    NN_LOG_INFO(std::string("Experiment completed. Results saved to ") + config.output_dir);
}
