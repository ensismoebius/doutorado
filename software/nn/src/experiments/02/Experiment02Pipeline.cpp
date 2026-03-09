#include "Experiment02Pipeline.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <random>
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

// Wavelet feature extraction (using core implementation)
auto extract_subband_energy(const wavelets::WaveletTransformResults& transform, int level)
    -> std::vector<double>
{
    return wavelets::extract_subband_energies(transform, level);
}

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
            std::string eeg_path = "/path/to/S01_EEG.mat";
            std::string audio_path = "/path/to/S01_Audio.mat";
            eeg_samples = load_eeg_data(eeg_path);
            audio_samples = load_audio_data(audio_path);
            std::cout << "Loaded " << eeg_samples.size() << " EEG samples and "
                      << audio_samples.size() << " audio samples" << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cout << "Data loading failed: " << e.what() << std::endl;
            std::cout << "Using synthetic data for demonstration" << std::endl;

            eeg_samples.resize(100);
            audio_samples.resize(100);

            std::mt19937 rng(config.random_seed);
            std::normal_distribution<double> dist(0.0, 1.0);

            for (int i = 0; i < 100; ++i)
            {
                EEGSample eeg;
                eeg.channels.resize(6, std::vector<double>(4096));
                for (auto& ch : eeg.channels)
                {
                    for (double& s : ch) s = dist(rng);
                }
                eeg.modality = 1;
                eeg.stimulus = (i % 5) + 1;
                eeg.artifacts = 1;
                eeg_samples[i] = eeg;

                AudioSample audio;
                audio.signal.resize(176400);
                for (double& s : audio.signal) s = dist(rng);
                audio.stimulus = eeg.stimulus;
                audio.eeg_index = i;
                audio_samples[i] = audio;
            }
        }

        auto windows = extract_windows(eeg_samples,
                                       audio_samples,
                                       config.window_duration_sec,
                                       config.overlap_sec,
                                       config.eeg_sampling_rate,
                                       config.audio_sampling_rate);
        std::cout << "Extracted " << windows.size() << " windows" << std::endl;

        std::vector<std::vector<double>> features;
        std::vector<int> labels;

        for (const auto& window : windows)
        {
            std::vector<double> combined_signal = window.eeg_window;
            combined_signal.insert(
                combined_signal.end(), window.audio_window.begin(), window.audio_window.end());

            auto wavelet_coeffs =
                get_wavelet_coeffs(wavelet_name, combined_signal, config.max_decomposition_depth);

            auto energies = extract_subband_energy(wavelet_coeffs, config.max_decomposition_depth);
            features.push_back(energies);
            labels.push_back(window.label);
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
