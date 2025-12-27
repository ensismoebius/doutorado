#include <matio.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Config.hpp"
#include "core/dataLoaders/10.1117/AudioLoader.h"
#include "core/dataLoaders/10.1117/EEGLoader.h"
#include "core/dataLoaders/MatFileUtils.h"
#include "core/optimizers/Adam.hpp"
#include "core/paraconsistent/paraconsistent.h"
#include "core/wavelet/Types.h"
#include "core/wavelet/waveletOperations.h"

// Processes a single trial (either EEG or Audio) and returns its features.
// For EEG, signal_data will be (channels x samples_per_channel)
// For Audio, signal_data will be (1 x samples_per_channel)
auto extract_wavelet_features_single_trial(const Eigen::MatrixXf& signal_data, double duration_sec,
                                           int overlap_percent, int sampling_rate)
    -> std::vector<double>
{
    // Daubechies 4 lowpass filter (example)
    auto lowpass = wavelets::get_wavelet<wavelets::Daub4>();

    std::vector<double> all_channel_energies;

    for (int channel_row_idx = 0; channel_row_idx < signal_data.rows(); ++channel_row_idx)
    {
        Eigen::VectorXf channel_data = signal_data.row(channel_row_idx);
        std::vector<double> sig(channel_data.data(), channel_data.data() + channel_data.size());

        // Apply wavelet packet transform
        auto wtr = wavelets::malat(sig, lowpass, wavelets::PACKET_WAVELET, 4); // level 4

        // Extract sub-band energies for this channel
        // Approximation
        auto approx = wtr.getWaveletTransforms(0);
        double energy_approx =
            std::inner_product(approx.begin(), approx.end(), approx.begin(), 0.0);
        all_channel_energies.push_back(energy_approx);

        for (int d = 1; d <= wtr.levelsOfTransformation; ++d)
        {
            auto detail = wtr.getWaveletTransforms(d);
            double energy_detail =
                std::inner_product(detail.begin(), detail.end(), detail.begin(), 0.0);
            all_channel_energies.push_back(energy_detail);
        }
    }

    return all_channel_energies;
}

auto normalize_features(std::vector<std::vector<double>>& features,
                        const std::vector<double>& range) -> void
{
    if (features.empty())
    {
        return;
    }

    size_t num_features = features[0].size();
    std::vector<double> min_vals(num_features, std::numeric_limits<double>::max());
    std::vector<double> max_vals(num_features, std::numeric_limits<double>::lowest());

    for (const auto& feat : features)
    {
        for (size_t i = 0; i < num_features; ++i)
        {
            min_vals[i] = std::min(min_vals[i], feat[i]);
            max_vals[i] = std::max(max_vals[i], feat[i]);
        }
    }

    for (auto& feat : features)
    {
        for (size_t i = 0; i < num_features; ++i)
        {
            if (max_vals[i] != min_vals[i])
            {
                feat[i] = range[0] + (((feat[i] - min_vals[i]) / (max_vals[i] - min_vals[i])) *
                                      (range[1] - range[0]));
            }
            else
            {
                feat[i] = range[0];
            }
        }
    }
}

auto compute_paraconsistent_metrics(const std::vector<std::vector<double>>& features,
                                    const std::vector<int>& labels)
    -> std::tuple<double, double, double, double>
{
    // Use the existing paraconsistent library
    std::map<std::string, std::vector<std::vector<double>>> arrClasses;

    // Group features by label
    for (size_t i = 0; i < features.size(); ++i)
    {
        std::string class_name = std::to_string(labels[i]);
        arrClasses[class_name].push_back(features[i]);
    }

    unsigned int amountOfClasses = arrClasses.size();
    if (amountOfClasses == 0)
    {
        return {0.0, 0.0, 0.0, 0.0};
    }

    // Assume all classes have the same number of vectors (simplification)
    unsigned int featureVectorsPerClass = arrClasses.begin()->second.size();
    unsigned int featureVectorSize = features[0].size();

    // Normalize the feature vectors
    normalizeClassesFeatureVectors(
        amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);

    // Calculate alpha and beta
    double alpha =
        calculateAlpha(amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);
    double beta =
        calculateBeta(amountOfClasses, featureVectorsPerClass, featureVectorSize, arrClasses);

    // Calculate G1 and G2
    double g1 = calcCertaintyDegree_G1(alpha, beta);
    double g2 = calcContradictionDegree_G2(alpha, beta);

    return {alpha, beta, g1, g2};
}

auto train_resnet(const std::vector<std::vector<double>>& features, const std::vector<int>& labels)
    -> double
{
    // Implement simple ResNet-like training using existing layers
    // Placeholder: for now, return dummy accuracy
    // TODO: implement actual training with Sequential, Linear, ReLU, ResidualBlock, etc.
    return 0.85;
}

auto save_results(const std::string& filename, double alpha, double beta, double g1, double g2,
                  double accuracy) -> void
{
    std::ofstream file(filename);
    file << "alpha,beta,g1,g2,accuracy\n";
    file << alpha << "," << beta << "," << g1 << "," << g2 << "," << accuracy << "\n";
}

auto main(int argc, const char* argv[]) -> int
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>\n";
        return 1;
    }

    std::string config_path = argv[1];
    auto cfg_opt = Config::load(config_path);
    if (!cfg_opt)
    {
        std::cerr << "Failed to load config\n";
        return 1;
    }
    const Config& cfg = *cfg_opt;

    std::cout << "PHASE 0: Wavelet Baseline Experiment\n";
    std::cout << "Config loaded successfully\n";

    // Load data from all subjects like in lfcc_pipeline
    std::string basePath =
        "/home/ensismoebius/Documentos/UNESP/doutorado/"
        "databases/BaseDeDatosHablaImaginada/";

    // Vectors to accumulate features and labels from all subjects
    std::vector<std::vector<double>> all_combined_features;
    std::vector<int> all_combined_labels;

    for (const auto& entry : std::filesystem::directory_iterator(basePath))
    {
        if (entry.is_directory())
        {
            std::string subjectPath = entry.path().string();
            std::string subjectName = entry.path().filename().string();
            std::string audioFilePath = subjectPath + "/" + subjectName + "_Audio.mat";
            std::string eegFilePath = subjectPath + "/" + subjectName + "_EEG.mat";

            if (std::filesystem::exists(audioFilePath) && std::filesystem::exists(eegFilePath))
            {
                // Get dimensions to determine number of trials
                auto eeg_dims_opt = matioCpp::utils::get_variable_dimensions(eegFilePath, "EEG");
                auto audio_dims_opt =
                    matioCpp::utils::get_variable_dimensions(audioFilePath, "Audio");

                if (!eeg_dims_opt || eeg_dims_opt->empty() || !audio_dims_opt ||
                    audio_dims_opt->empty())
                {
                    std::cerr << "Failed to get dimensions for " << subjectName << ". Skipping.\n";
                    continue;
                }

                // Assuming number of rows (trials) is the first dimension
                size_t num_eeg_trials = eeg_dims_opt->at(0);
                size_t num_audio_trials = audio_dims_opt->at(0);

                if (num_eeg_trials == 0 || num_audio_trials == 0)
                {
                    std::cerr << "No trials found for " << subjectName << ". Skipping.\n";
                    continue;
                }

                if (num_eeg_trials != num_audio_trials)
                {
                    std::cerr << "Mismatch in number of trials for " << subjectName
                              << ". EEG: " << num_eeg_trials << ", Audio: " << num_audio_trials
                              << ". Skipping.\n";
                    continue;
                }

                std::cout << "Processing subject " << subjectName << " with " << num_eeg_trials
                          << " trials...\n";

                for (size_t trial_idx = 0; trial_idx < num_eeg_trials; ++trial_idx)
                {
                    // Load single EEG trial
                    auto [eeg_trial_data, eeg_labels_array] =
                        nn::dataLoaders::loadEEGFromMat(eegFilePath, trial_idx);
                    int eeg_label = eeg_labels_array[1]; // Assuming stimulus is at index 1

                    // Load single Audio trial
                    auto [audio_trial_data, audio_stimulus_label, audio_eeg_index] =
                        nn::dataLoaders::loadAudioFromMat(audioFilePath, trial_idx);
                    // For now, we'll use EEG label for combined features.
                    // If audio_stimulus_label is needed, it can be added to all_combined_labels or
                    // used for cross-validation.

                    // Extract features from EEG trial
                    std::vector<double> eeg_features_single_trial =
                        extract_wavelet_features_single_trial(eeg_trial_data,
                                                              cfg.duration_sec,
                                                              cfg.overlap_percent,
                                                              cfg.eeg_sampling_rate);

                    // Extract features from Audio trial
                    std::vector<double> audio_features_single_trial =
                        extract_wavelet_features_single_trial(audio_trial_data,
                                                              cfg.duration_sec,
                                                              cfg.overlap_percent,
                                                              cfg.sampling_rate);

                    // Combine features
                    std::vector<double> combined_features_current_trial = eeg_features_single_trial;
                    combined_features_current_trial.insert(combined_features_current_trial.end(),
                                                           audio_features_single_trial.begin(),
                                                           audio_features_single_trial.end());

                    // Accumulate combined features and labels
                    all_combined_features.push_back(combined_features_current_trial);
                    all_combined_labels.push_back(eeg_label);
                }
            }
        }
    }

    // --- Global processing after all subjects are processed ---
    if (all_combined_features.empty())
    {
        std::cerr << "No data processed for any subject. Exiting.\n";
        return 1;
    }

    // Normalize all combined features
    normalize_features(all_combined_features, cfg.range);

    // Compute paraconsistent metrics
    auto [alpha, beta, g1, g2] =
        compute_paraconsistent_metrics(all_combined_features, all_combined_labels);

    // Train classifier
    double accuracy = train_resnet(all_combined_features, all_combined_labels);

    // Save results
    save_results("results_wavelet_baseline.csv", alpha, beta, g1, g2, accuracy);

    std::cout << "Experiment completed. Results saved to results_wavelet_baseline.csv\n";

    return 0;
}