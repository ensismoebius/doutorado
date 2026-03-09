#include "Experiment02Pipeline.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "Experiment02Data.hpp"
#include "Experiment02Evaluation.hpp"
#include "Experiment02Reporting.hpp"
#include "Experiment02Training.hpp"
#include "Experiment02Wavelets.hpp"
#include "nn/linearAlgebra/linear_algebra.hpp"
#include "nn/wavelet/waveletOperations.h"

namespace fs = std::filesystem;

namespace
{
constexpr int kSyntheticSampleCount = 100;
constexpr const char* kDefaultEegPath = "/path/to/S01_EEG.mat";
constexpr const char* kDefaultAudioPath = "/path/to/S01_Audio.mat";
} // namespace

// Normalization (using core implementation)
void min_max_normalize(std::vector<std::vector<double>>& features,
                       const std::vector<double>& range = {0.0, 1.0})
{
    linearAlgebra::minMaxNormalizeFeatures(features, range);
}

auto run_wavelet_baseline_experiment(const ExperimentConfig& config) -> void
{
    std::cout << "Starting experiment: " << config.id << std::endl;
    fs::create_directories(config.output_dir);

    for (const std::string& wavelet_name : config.wavelet_families)
    {
        std::cout << "Processing wavelet: " << wavelet_name << std::endl;

        std::vector<EEGSample> eeg_samples;
        std::vector<AudioSample> audio_samples;

        try
        {
            std::string eeg_path = kDefaultEegPath;
            std::string audio_path = kDefaultAudioPath;
            eeg_samples = load_eeg_data(eeg_path);
            audio_samples = load_audio_data(audio_path);
            std::cout << "Loaded " << eeg_samples.size() << " EEG samples and "
                      << audio_samples.size() << " audio samples" << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cout << "Data loading failed: " << e.what() << std::endl;
            std::cout << "Using synthetic data for demonstration" << std::endl;

            generate_synthetic_samples(
                eeg_samples, audio_samples, kSyntheticSampleCount, config.random_seed);
        }

        auto windows = extract_windows(eeg_samples,
                                       audio_samples,
                                       config.window_duration_sec,
                                       config.overlap_sec,
                                       config.eeg_sampling_rate,
                                       config.audio_sampling_rate);
        std::cout << "Extracted " << windows.size() << " windows" << std::endl;

        if (windows.empty())
        {
            std::cout << "No valid windows for " << wavelet_name << "; skipping this wavelet."
                      << std::endl;
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
            std::cout << "No features produced for " << wavelet_name << "; skipping this wavelet."
                      << std::endl;
            continue;
        }

        min_max_normalize(features, config.normalization_range);

        auto fold_results =
            k_fold_cross_validation(features, labels, config.k_folds, config.random_seed);

        const auto aggregated = aggregate_fold_results(fold_results);

        std::string csv_path = config.output_dir + "/" + wavelet_name + "_results.csv";
        write_wavelet_results_csv(csv_path, config, wavelet_name, aggregated);

        std::cout << "Completed " << wavelet_name << " - F1: " << aggregated.classification.f1_score
                  << ", Alpha: " << aggregated.paraconsistent.alpha << std::endl;
    }

    std::cout << "Experiment completed. Results saved to " << config.output_dir << std::endl;
}
