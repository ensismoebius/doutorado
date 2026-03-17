/**
 * @file phase00_data.cpp
 * @brief PHASE 0 dataset traversal and trial extraction.
 *
 * This translation unit:
 * - iterates subjects under `cfg.dataset_base_path`
 * - checks for `*_Audio.mat` and `*_EEG.mat`
 * - loads trials, aligns EEG/audio by index, and produces `TrialData` records
 */

#include "phase00_data.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "nn/dataLoaders/10.1117/AudioLoader.h"
#include "nn/dataLoaders/10.1117/EEGLoader.h"
#include "nn/dataLoaders/mat_file_utils.hpp"
#include "nn/tensor/Tensor.hpp"
#include "phase00_features.hpp"

using matioCpp::utils::get_variable_dimensions;
using nn::dataLoaders::loadAudioFromMat;
using nn::dataLoaders::loadEEGFromMat;

namespace phase00
{

auto default_config_path() -> std::filesystem::path
{
    const std::filesystem::path here{__FILE__};
    return here.parent_path().parent_path() / "config.yaml";
}

auto aggregate_trials(const Config& cfg) -> std::vector<TrialData>
{
    std::vector<TrialData> trials;

    for (const auto& entry : std::filesystem::directory_iterator(cfg.dataset_base_path))
    {
        if (!entry.is_directory())
        {
            continue;
        }

        const auto subject_path = entry.path();
        const std::string subject_name = subject_path.filename().string();
        const auto audio_file = subject_path / (subject_name + "_Audio.mat");
        const auto eeg_file = subject_path / (subject_name + "_EEG.mat");

        if (!std::filesystem::exists(audio_file) || !std::filesystem::exists(eeg_file))
        {
            continue;
        }

        auto eeg_dims_opt = get_variable_dimensions(eeg_file.string(), "EEG");
        auto audio_dims_opt = get_variable_dimensions(audio_file.string(), "Audio");

        if (!eeg_dims_opt || eeg_dims_opt->empty() || !audio_dims_opt || audio_dims_opt->empty())
        {
            std::cerr << "Failed to get dimensions for " << subject_name << ". Skipping.\n";
            continue;
        }

        const size_t num_eeg_trials = eeg_dims_opt->at(0);
        const size_t num_audio_trials = audio_dims_opt->at(0);

        if (num_eeg_trials == 0 || num_audio_trials == 0)
        {
            std::cerr << "No trials found for " << subject_name << ". Skipping.\n";
            continue;
        }

        if (num_eeg_trials != num_audio_trials)
        {
            std::cerr << "Mismatch in number of trials for " << subject_name
                      << ". EEG: " << num_eeg_trials << ", Audio: " << num_audio_trials
                      << ". Skipping.\n";
            continue;
        }

        std::cout << "Processing subject " << subject_name << " with " << num_eeg_trials
                  << " trials...\n";

        for (size_t trial_idx = 0; trial_idx < num_eeg_trials; ++trial_idx)
        {
            auto [eeg_trial_data, eeg_labels_array] = loadEEGFromMat(eeg_file.string(), trial_idx);
            const int eeg_label = eeg_labels_array[1];

            auto [audio_trial_data, audio_stimulus_label, audio_eeg_index] =
                loadAudioFromMat(audio_file.string(), trial_idx);
            (void) audio_stimulus_label;
            (void) audio_eeg_index;

            std::vector<double> eeg_features_single_trial =
                extract_wavelet_features_single_trial(nn::Tensor(eeg_trial_data),
                    cfg.duration_sec,
                    cfg.overlap_percent,
                    cfg.eeg_sampling_rate);

            std::vector<double> audio_features_single_trial =
                extract_wavelet_features_single_trial(nn::Tensor(audio_trial_data),
                    cfg.duration_sec,
                    cfg.overlap_percent,
                    cfg.sampling_rate);

            std::vector<double> combined_features_current_trial = eeg_features_single_trial;
            combined_features_current_trial.insert(combined_features_current_trial.end(),
                audio_features_single_trial.begin(),
                audio_features_single_trial.end());

            trials.push_back({std::move(combined_features_current_trial), eeg_label});
        }
    }

    return trials;
}

} // namespace phase00
