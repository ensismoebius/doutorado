#include "Experiment02Pipeline.hpp"

#include <filesystem>
#include <string>
#include <vector>

#include "Experiment02Data.hpp"
#include "Experiment02Evaluation.hpp"
#include "Experiment02Reporting.hpp"
#include "Experiment02Training.hpp"
#include "Experiment02Wavelets.hpp"
#include "nn/linearAlgebra/linear_algebra.hpp"
#include "nn/logging/Logger.hpp"
#include "nn/wavelet/waveletOperations.h"

namespace fs = std::filesystem;

namespace
{
constexpr int kSyntheticSampleCount = 100;
constexpr const char* kDefaultEegPath = "/path/to/S01_EEG.mat";
constexpr const char* kDefaultAudioPath = "/path/to/S01_Audio.mat";
} // namespace

// Normalization (using core implementation)
void min_max_normalize(
    std::vector<std::vector<double>>& features, const std::vector<double>& range = {0.0, 1.0})
{
    linearAlgebra::minMaxNormalizeFeatures(features, range);
}

auto run_wavelet_baseline_experiment(const ExperimentConfig& config) -> void
{
    NN_LOG_INFO(std::string("Starting experiment: ") + config.id);
    fs::create_directories(config.output_dir);

    for (const std::string& wavelet_name : config.wavelet_families)
    {
        NN_LOG_INFO(std::string("Processing wavelet: ") + wavelet_name);

        std::vector<EEGSample> eeg_samples;
        std::vector<AudioSample> audio_samples;

        try
        {
            std::string eeg_path = kDefaultEegPath;
            std::string audio_path = kDefaultAudioPath;
            eeg_samples = load_eeg_data(eeg_path);
            audio_samples = load_audio_data(audio_path);
            NN_LOG_INFO("Loaded " + std::to_string(eeg_samples.size()) + " EEG samples and " +
                        std::to_string(audio_samples.size()) + " audio samples");
        }
        catch (const std::exception& e)
        {
            NN_LOG_WARN(std::string("Data loading failed: ") + e.what());
            NN_LOG_INFO("Using synthetic data for demonstration");

            generate_synthetic_samples(
                eeg_samples, audio_samples, kSyntheticSampleCount, config.random_seed);
        }

        auto windows = extract_windows(eeg_samples,
            audio_samples,
            config.window_duration_sec,
            config.overlap_sec,
            config.eeg_sampling_rate,
            config.audio_sampling_rate);
        NN_LOG_INFO("Extracted " + std::to_string(windows.size()) + " windows");

        if (windows.empty())
        {
            NN_LOG_WARN("No valid windows for " + wavelet_name + "; skipping this wavelet.");
            continue;
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
            NN_LOG_WARN("No features produced for " + wavelet_name + "; skipping this wavelet.");
            continue;
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
