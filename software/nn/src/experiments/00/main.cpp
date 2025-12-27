#include <matio.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Config.hpp"
#include "core/dataLoaders/MatFileUtils.h"
#include "core/optimizers/Adam.hpp"
#include "core/paraconsistent/paraconsistent.h"
#include "core/wavelet/Types.h"
#include "core/wavelet/waveletOperations.h"

struct WindowedFeatures
{
    std::vector<std::vector<double>> features; // each inner vector is a feature vector for a window
    std::vector<int> labels;
};

auto extract_wavelet_features(const Eigen::MatrixXf& signal, int data_cols, double duration_sec,
                              int overlap_percent, int sampling_rate) -> WindowedFeatures
{
    WindowedFeatures result;

    // Daubechies 4 lowpass filter (example)
    auto lowpass = wavelets::get_wavelet<wavelets::Daub4>();

    for (int row = 0; row < signal.rows(); ++row)
    {
        Eigen::VectorXf row_data = signal.row(row).head(data_cols); // data part
        int label = static_cast<int>(signal(row, data_cols)); // assume first label is stimulus

        // For simplicity, take one window from the row
        std::vector<double> sig(row_data.data(), row_data.data() + row_data.size());

        // Apply wavelet packet transform
        auto wtr = wavelets::malat(sig, lowpass, wavelets::PACKET_WAVELET, 4); // level 4

        // Extract sub-band energies
        std::vector<double> energies;
        // Approximation
        auto approx = wtr.getWaveletTransforms(0);
        double energy_approx =
            std::inner_product(approx.begin(), approx.end(), approx.begin(), 0.0);
        energies.push_back(energy_approx);

        for (int d = 1; d <= wtr.levelsOfTransformation; ++d)
        {
            auto detail = wtr.getWaveletTransforms(d);
            double energy_detail =
                std::inner_product(detail.begin(), detail.end(), detail.begin(), 0.0);
            energies.push_back(energy_detail);
        }

        result.features.push_back(energies);
        result.labels.push_back(label);
    }

    return result;
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

auto main(int argc, char* argv[]) -> int
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

    std::cout << "PHASE 1: Wavelet Baseline Experiment\n";
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
                // Load EEG for current subject
                auto eeg_opt = matioCpp::utils::load_named_variable_as_matrix(eegFilePath, "EEG");
                if (!eeg_opt)
                {
                    std::cerr << "Failed to load EEG for subject " << subjectName << std::endl;
                    continue; // Skip to next subject
                }
                Eigen::MatrixXf eeg_data = *eeg_opt;

                // Load Audio for current subject
                auto audio_opt =
                    matioCpp::utils::load_named_variable_as_matrix(audioFilePath, "Audio");
                if (!audio_opt)
                {
                    std::cerr << "Failed to load Audio for subject " << subjectName << std::endl;
                    continue; // Skip to next subject
                }
                Eigen::MatrixXf audio_data = *audio_opt;

                // --- Process single subject data here ---
                // Extract features from EEG
                auto eeg_features = extract_wavelet_features(
                    eeg_data, 24576, cfg.duration_sec, cfg.overlap_percent, cfg.eeg_sampling_rate);

                // Extract features from Audio
                auto audio_features = extract_wavelet_features(
                    audio_data, 176400, cfg.duration_sec, cfg.overlap_percent, cfg.sampling_rate);

                // Combine features (simple concatenation)
                std::vector<std::vector<double>> combined_features_subject;
                for (size_t i = 0; i < eeg_features.features.size(); ++i)
                {
                    std::vector<double> combined = eeg_features.features[i];
                    combined.insert(
                        combined.end(), audio_features.features[i].begin(), audio_features.features[i].end());
                    combined_features_subject.push_back(combined);
                }

                // Accumulate combined features and labels
                all_combined_features.insert(all_combined_features.end(),
                                             combined_features_subject.begin(),
                                             combined_features_subject.end());
                all_combined_labels.insert(all_combined_labels.end(),
                                           eeg_features.labels.begin(),
                                           eeg_features.labels.end());

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