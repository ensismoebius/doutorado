#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "core/dataLoaders/mat_file_utils.hpp"
#include "core/layers/CrossEntropyLoss.hpp"
#include "core/layers/SimpleResNet.hpp"
#include "core/linearAlgebra/linear_algebra.hpp"
#include "core/optimizers/Adam.hpp"
#include "core/paraconsistent/paraconsistent.h"
#include "core/statistics/multi_class_metrics.hpp"
#include "core/tensor/Tensor.hpp"
#include "core/utility/batching.hpp"
#include "core/wave/audioFeatureExtraction.h"
#include "core/wavelet/Types.h"
#include "core/wavelet/waveletOperations.h"

namespace fs = std::filesystem;

// Experiment configuration
struct ExperimentConfig
{
    std::string id = "M1_WaveletPacket_Baseline";
    int random_seed = 42;
    bool enforce_single_thread = false;
    bool allow_parallelism = true;
    double numerical_tolerance = 1e-9;

    // Data
    std::vector<std::string> modalities = {"EEG", "Audio"};
    int eeg_sampling_rate = 1000;
    int audio_sampling_rate = 44100;

    // Segmentation
    std::string window_type = "hanning";
    double window_duration_sec = 1.5;
    double overlap_sec = 0.5;

    // Wavelet
    std::vector<std::string> wavelet_families;
    int max_decomposition_depth = 0; // computed from window length

    // Features
    std::string feature_extraction = "subband_energy";
    std::string normalization_method = "min_max";
    std::vector<double> normalization_range = {0.0, 1.0};

    // Paraconsistent
    bool paraconsistent_enabled = true;
    std::vector<std::string> paraconsistent_metrics = {"alpha", "beta", "G1", "G2"};

    // Classifier
    std::string classifier_model = "ResNet";
    std::string classifier_paradigm = "spiking_neural_network";
    int embedding_dim = 128;
    int resnet_depth = 3; // number of residual blocks

    // Validation
    int k_folds = 10;
    bool stratified = true;
    bool shuffle = true;

    // Output
    std::string output_dir = "results";
    std::vector<std::string> output_formats = {"csv"};
};

// Get all available discrete wavelet families from core library
auto get_available_wavelet_families() -> std::vector<std::string>
{
    return {"Haar",   "Daub4",  "Daub6",  "Daub8",  "Daub10", "Daub12", "Daub14", "Daub16",
            "Daub18", "Daub20", "Daub22", "Daub24", "Daub26", "Daub28", "Daub30", "Daub32",
            "Daub34", "Daub36", "Daub38", "Daub40", "Daub42", "Daub44", "Daub46"};
}

// Load config from spec.yaml
auto load_experiment_config(const std::string& spec_path) -> ExperimentConfig
{
    YAML::Node config = YAML::LoadFile(spec_path);

    ExperimentConfig exp_config;

    exp_config.id = config["experiment"]["id"].as<std::string>();
    exp_config.random_seed = config["determinism"]["random_seed"].as<int>();

    exp_config.eeg_sampling_rate = config["data"]["sampling_rate"]["EEG"].as<int>();
    exp_config.audio_sampling_rate = config["data"]["sampling_rate"]["Audio"].as<int>();

    exp_config.window_duration_sec =
        config["data"]["segmentation"]["window"]["duration_sec"].as<double>();
    exp_config.overlap_sec = config["data"]["segmentation"]["window"]["overlap_sec"].as<double>();

    // Calculate max decomposition depth based on window length
    int window_samples =
        static_cast<int>(exp_config.window_duration_sec * exp_config.eeg_sampling_rate);
    exp_config.max_decomposition_depth = static_cast<int>(std::floor(std::log2(window_samples)));

    // Use all available discrete wavelets from core library
    exp_config.wavelet_families = get_available_wavelet_families();

    exp_config.normalization_range = {0.0, 1.0};
    exp_config.paraconsistent_enabled = config["paraconsistent_metrics"]["enabled"].as<bool>();
    exp_config.k_folds = config["validation_protocol"]["k"].as<int>();

    return exp_config;
}

// Windowing functions (using core implementations)
auto hanning_window(int length) -> std::vector<double>
{
    return nn::core::wave::hanning_window(length);
}

auto apply_window(const std::vector<double>& signal, const std::vector<double>& window)
    -> std::vector<double>
{
    return nn::core::wave::apply_window(signal, window);
}

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

// Paraconsistent metrics calculation
struct ParaconsistentMetrics
{
    double alpha = 0.0;
    double beta = 0.0;
    double G1 = 0.0;
    double G2 = 0.0;
};

auto compute_paraconsistent_metrics(const std::vector<std::vector<double>>& features,
                                    const std::vector<int>& labels) -> ParaconsistentMetrics
{
    if (features.empty() || labels.empty())
    {
        return ParaconsistentMetrics{};
    }

    // Group features by class
    std::map<std::string, std::vector<std::vector<double>>> class_features;
    for (size_t i = 0; i < features.size(); ++i)
    {
        std::string class_key = std::to_string(labels[i]);
        class_features[class_key].push_back(features[i]);
    }

    if (class_features.empty())
    {
        return ParaconsistentMetrics{};
    }

    unsigned int n_classes = class_features.size();
    unsigned int n_samples_per_class = class_features.begin()->second.size();
    unsigned int feature_dim = features[0].size();

    double alpha = calculate_alpha(n_classes, n_samples_per_class, feature_dim, class_features);
    double beta = calculate_beta(n_classes, n_samples_per_class, feature_dim, class_features);

    ParaconsistentMetrics metrics;
    metrics.alpha = alpha;
    metrics.beta = beta;
    metrics.G1 = calculate_certainty_degree_g1(alpha, beta);
    metrics.G2 = calculate_contradiction_degree_g2(alpha, beta);

    return metrics;
}

// Classification metrics (using core implementation)
using ClassificationMetrics = statistics::ClassificationMetrics;

auto compute_classification_metrics(const std::vector<int>& true_labels,
                                    const std::vector<int>& pred_labels) -> ClassificationMetrics
{
    return statistics::compute_classification_metrics(true_labels, pred_labels);
}

// Simple SNN-ResNet classifier (using core implementation)
using SNNResNet = SimpleResNet;

// K-fold cross validation (using core template)
struct FoldResult
{
    ClassificationMetrics metrics;
    ParaconsistentMetrics para_metrics;
    double fold_time_sec;
};

auto k_fold_cross_validation(const std::vector<std::vector<double>>& features,
                             const std::vector<int>& labels, int k_folds, int random_seed)
    -> std::vector<FoldResult>
{
    // Define fold function that returns FoldResult
    auto fold_function = [&](const std::vector<std::vector<double>>& train_features,
                             const std::vector<int>& train_labels,
                             const std::vector<std::vector<double>>& test_features,
                             const std::vector<int>& test_labels) -> FoldResult
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Compute paraconsistent metrics on training set
        ParaconsistentMetrics para_metrics =
            compute_paraconsistent_metrics(train_features, train_labels);

        // Train classifier
        int n_classes = 0;
        for (int label : labels) n_classes = std::max(n_classes, label + 1);

        SimpleResNet model(train_features[0].size(), 128, n_classes);
        Adam optimizer(0.001f);
        auto params = model.params();
        optimizer.attach(params);
        CrossEntropyLoss loss;

        // Convert to tensors for training
        std::vector<nn::Tensor> train_inputs, train_targets;
        for (size_t i = 0; i < train_features.size(); ++i)
        {
            nn::Tensor x(1, static_cast<size_t>(train_features[i].size()));
            for (size_t j = 0; j < train_features[i].size(); ++j)
            {
                x.at(0, static_cast<size_t>(j)) = train_features[i][j];
            }
            train_inputs.emplace_back(x);

            nn::Tensor y(1, n_classes);
            y.at(0, train_labels[i]) = 1.0f;
            train_targets.emplace_back(y);
        }

        // Training loop (simplified)
        for (int epoch = 0; epoch < 10; ++epoch)
        {
            auto batches = create_batches(train_inputs, train_targets, 32);
            for (const auto& batch : batches)
            {
                loss.set_target(batch.targets);
                nn::Tensor logits = model.forward(batch.inputs);
                nn::Tensor loss_tensor = loss.forward(logits);
                nn::Tensor grad_loss = loss.backward(logits);

                model.backward(grad_loss);
                optimizer.step(params);
            }
        }

        // Test
        std::vector<int> pred_labels;
        for (const auto& test_feat : test_features)
        {
            nn::Tensor x(1, static_cast<size_t>(test_feat.size()));
            for (size_t j = 0; j < test_feat.size(); ++j)
            {
                x.at(0, static_cast<size_t>(j)) = test_feat[j];
            }
            nn::Tensor input = x;
            nn::Tensor output = model.forward(input);

            // Get prediction
            int pred = 0;
            float max_val = output.at(0, 0);
            for (int c = 1; c < n_classes; ++c)
            {
                if (output.at(0, c) > max_val)
                {
                    max_val = output.at(0, c);
                    pred = c;
                }
            }
            pred_labels.push_back(pred);
        }

        ClassificationMetrics cls_metrics =
            compute_classification_metrics(test_labels, pred_labels);

        auto end_time = std::chrono::high_resolution_clock::now();
        double fold_time = std::chrono::duration<double>(end_time - start_time).count();

        return FoldResult{cls_metrics, para_metrics, fold_time};
    };

    return statistics::k_fold_cross_validation<FoldResult>(
        features, labels, k_folds, random_seed, fold_function);
}

// Get wavelet coefficients for a given wavelet family
auto get_wavelet_coeffs(const std::string& wavelet_name, const std::vector<double>& signal,
                        int max_level) -> wavelets::WaveletTransformResults
{
    // Map wavelet name to type using if-else chain for all available wavelets
    if (wavelet_name == "Haar")
    {
        auto filter = wavelets::get_wavelet<wavelets::Haar>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub4")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub4>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub6")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub6>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub8")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub8>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub10")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub10>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub12")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub12>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub14")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub14>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub16")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub16>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub18")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub18>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub20")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub20>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub22")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub22>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub24")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub24>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub26")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub26>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub28")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub28>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub30")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub30>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub32")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub32>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub34")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub34>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub36")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub36>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub38")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub38>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub40")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub40>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub42")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub42>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub44")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub44>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    else if (wavelet_name == "Daub46")
    {
        auto filter = wavelets::get_wavelet<wavelets::Daub46>();
        return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
    }
    // Default to Haar if wavelet not found
    auto filter = wavelets::get_wavelet<wavelets::Haar>();
    return wavelets::malat(signal, filter, wavelets::PACKET_WAVELET, max_level);
}

// Data structures for EEG/Audio samples
struct EEGSample
{
    std::vector<std::vector<double>> channels; // 6 channels × 4096 samples each
    int modality;                              // 1=imagined, 2=pronounced
    int stimulus;                              // 1-5: A,E,I,O,U; 6-11: commands
    int artifacts;                             // 1=no artifacts, 2=blink present
};

struct AudioSample
{
    std::vector<double> signal; // 176400 samples
    int stimulus;               // same as EEG
    int eeg_index;              // corresponding EEG row index
};

struct WindowedSample
{
    std::vector<double> eeg_window;   // concatenated EEG channels for one window
    std::vector<double> audio_window; // audio window
    int label;                        // stimulus label
};

// Load EEG data from MAT file
auto load_eeg_data(const std::string& mat_path) -> std::vector<EEGSample>
{
    auto mat_opt = matioCpp::utils::load_named_variable_as_matrix(mat_path, "Sxx_EEG");
    if (!mat_opt)
    {
        throw std::runtime_error("Failed to load EEG data from " + mat_path);
    }

    nn::Tensor mat = std::move(*mat_opt);
    std::vector<EEGSample> samples;

    for (int i = 0; i < static_cast<int>(mat.rows()); ++i)
    {
        EEGSample sample;
        // Extract 6 channels × 4096 samples each = 24576 total
        for (int ch = 0; ch < 6; ++ch)
        {
            std::vector<double> channel(4096);
            for (int s = 0; s < 4096; ++s)
            {
                channel[s] = mat.at(i, ch * 4096 + s);
            }
            sample.channels.push_back(channel);
        }
        // Labels
        sample.modality = static_cast<int>(mat.at(i, 24576));
        sample.stimulus = static_cast<int>(mat.at(i, 24577));
        sample.artifacts = static_cast<int>(mat.at(i, 24578));

        samples.push_back(sample);
    }

    return samples;
}

// Load Audio data from MAT file
auto load_audio_data(const std::string& mat_path) -> std::vector<AudioSample>
{
    auto mat_opt = matioCpp::utils::load_named_variable_as_matrix(mat_path, "Sxx_Audio");
    if (!mat_opt)
    {
        throw std::runtime_error("Failed to load Audio data from " + mat_path);
    }

    nn::Tensor mat = std::move(*mat_opt);
    std::vector<AudioSample> samples;

    for (int i = 0; i < static_cast<int>(mat.rows()); ++i)
    {
        AudioSample sample;
        // Extract 176400 audio samples
        sample.signal.resize(176400);
        for (int s = 0; s < 176400; ++s)
        {
            sample.signal[s] = mat.at(i, s);
        }
        // Labels
        sample.stimulus = static_cast<int>(mat.at(i, 176400));
        sample.eeg_index = static_cast<int>(mat.at(i, 176401));

        samples.push_back(sample);
    }

    return samples;
}

// Extract windows from synchronized EEG/Audio data
auto extract_windows(const std::vector<EEGSample>& eeg_samples,
                     const std::vector<AudioSample>& audio_samples, double window_duration_sec,
                     double overlap_sec, int eeg_rate, int audio_rate)
    -> std::vector<WindowedSample>
{
    std::vector<WindowedSample> windows;

    // For each EEG sample, find corresponding audio and extract windows
    for (size_t eeg_idx = 0; eeg_idx < eeg_samples.size(); ++eeg_idx)
    {
        const auto& eeg = eeg_samples[eeg_idx];

        // Find corresponding audio sample
        const AudioSample* audio = nullptr;
        for (const auto& a : audio_samples)
        {
            if (a.eeg_index == static_cast<int>(eeg_idx))
            {
                audio = &a;
                break;
            }
        }
        if (!audio) continue;

        // Skip if artifacts present
        if (eeg.artifacts != 1) continue;

        // Calculate window parameters
        int eeg_window_samples = static_cast<int>(window_duration_sec * eeg_rate); // 1500 samples
        int audio_window_samples =
            static_cast<int>(window_duration_sec * audio_rate); // 66150 samples
        int eeg_step =
            static_cast<int>((window_duration_sec - overlap_sec) * eeg_rate); // 1000 samples
        // Note: audio_step is calculated but may not be directly used in loop

        // Extract windows with overlap
        for (int start_eeg = 0; start_eeg + eeg_window_samples <= 4096; start_eeg += eeg_step)
        {
            int start_audio = start_eeg * audio_rate / eeg_rate; // synchronize sampling rates

            if (start_audio + audio_window_samples > static_cast<int>(audio->signal.size())) break;

            WindowedSample window;
            window.label = eeg.stimulus;

            // Extract EEG window (concatenate all channels)
            for (int ch = 0; ch < 6; ++ch)
            {
                for (int s = start_eeg; s < start_eeg + eeg_window_samples; ++s)
                {
                    window.eeg_window.push_back(eeg.channels[ch][s]);
                }
            }

            // Extract Audio window
            for (int s = start_audio; s < start_audio + audio_window_samples; ++s)
            {
                window.audio_window.push_back(audio->signal[s]);
            }

            // Apply Hanning window
            auto hanning_eeg = hanning_window(eeg_window_samples);
            for (size_t i = 0; i < window.eeg_window.size(); ++i)
            {
                window.eeg_window[i] *= hanning_eeg[i % eeg_window_samples];
            }

            auto hanning_audio = hanning_window(audio_window_samples);
            for (size_t i = 0; i < window.audio_window.size(); ++i)
            {
                window.audio_window[i] *= hanning_audio[i];
            }

            windows.push_back(window);
        }
    }

    return windows;
}

// Main experiment function
auto run_wavelet_baseline_experiment(const ExperimentConfig& config) -> void
{
    std::cout << "Starting experiment: " << config.id << std::endl;

    // Create output directory
    fs::create_directories(config.output_dir);

    // For each wavelet family
    for (const std::string& wavelet_name : config.wavelet_families)
    {
        std::cout << "Processing wavelet: " << wavelet_name << std::endl;

        std::vector<EEGSample> eeg_samples;
        std::vector<AudioSample> audio_samples;

        try
        {
            // Load data (assuming data files are available)
            std::string eeg_path = "/path/to/S01_EEG.mat";     // TODO: parameterize
            std::string audio_path = "/path/to/S01_Audio.mat"; // TODO: parameterize
            eeg_samples = load_eeg_data(eeg_path);
            audio_samples = load_audio_data(audio_path);
            std::cout << "Loaded " << eeg_samples.size() << " EEG samples and "
                      << audio_samples.size() << " audio samples" << std::endl;
        }
        catch (const std::exception& e)
        {
            std::cout << "Data loading failed: " << e.what() << std::endl;
            std::cout << "Using synthetic data for demonstration" << std::endl;

            // Generate synthetic data for testing
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

        // Extract windows
        auto windows = extract_windows(eeg_samples,
                                       audio_samples,
                                       config.window_duration_sec,
                                       config.overlap_sec,
                                       config.eeg_sampling_rate,
                                       config.audio_sampling_rate);
        std::cout << "Extracted " << windows.size() << " windows" << std::endl;

        // Extract features for all windows
        std::vector<std::vector<double>> features;
        std::vector<int> labels;

        for (const auto& window : windows)
        {
            // Combine EEG and Audio for wavelet analysis
            std::vector<double> combined_signal = window.eeg_window;
            combined_signal.insert(
                combined_signal.end(), window.audio_window.begin(), window.audio_window.end());

            // Perform wavelet packet decomposition
            auto wavelet_coeffs =
                get_wavelet_coeffs(wavelet_name, combined_signal, config.max_decomposition_depth);

            // Extract subband energies as features
            auto energies = extract_subband_energy(wavelet_coeffs, config.max_decomposition_depth);
            features.push_back(energies);
            labels.push_back(window.label);
        }

        // Normalize features
        min_max_normalize(features, config.normalization_range);

        // Run k-fold CV
        auto fold_results =
            k_fold_cross_validation(features, labels, config.k_folds, config.random_seed);

        // Aggregate results
        ClassificationMetrics avg_cls_metrics;
        ParaconsistentMetrics avg_para_metrics;
        double total_time = 0.0;

        for (const auto& result : fold_results)
        {
            avg_cls_metrics.accuracy += result.metrics.accuracy;
            avg_cls_metrics.precision += result.metrics.precision;
            avg_cls_metrics.recall += result.metrics.recall;
            avg_cls_metrics.f1_score += result.metrics.f1_score;
            avg_cls_metrics.mcc += result.metrics.mcc;

            avg_para_metrics.alpha += result.para_metrics.alpha;
            avg_para_metrics.beta += result.para_metrics.beta;
            avg_para_metrics.G1 += result.para_metrics.G1;
            avg_para_metrics.G2 += result.para_metrics.G2;

            total_time += result.fold_time_sec;
        }

        int n_folds = fold_results.size();
        avg_cls_metrics.accuracy /= n_folds;
        avg_cls_metrics.precision /= n_folds;
        avg_cls_metrics.recall /= n_folds;
        avg_cls_metrics.f1_score /= n_folds;
        avg_cls_metrics.mcc /= n_folds;

        avg_para_metrics.alpha /= n_folds;
        avg_para_metrics.beta /= n_folds;
        avg_para_metrics.G1 /= n_folds;
        avg_para_metrics.G2 /= n_folds;

        // Save results to CSV
        std::string csv_path = config.output_dir + "/" + wavelet_name + "_results.csv";
        std::ofstream csv_file(csv_path);
        csv_file << "experiment_id,wavelet_name,decomposition_depth,alpha,beta,G1,G2,"
                 << "accuracy,precision,recall,f1_score,mcc,total_time_sec\n";
        csv_file << config.id << "," << wavelet_name << "," << config.max_decomposition_depth << ","
                 << avg_para_metrics.alpha << "," << avg_para_metrics.beta << ","
                 << avg_para_metrics.G1 << "," << avg_para_metrics.G2 << ","
                 << avg_cls_metrics.accuracy << "," << avg_cls_metrics.precision << ","
                 << avg_cls_metrics.recall << "," << avg_cls_metrics.f1_score << ","
                 << avg_cls_metrics.mcc << "," << total_time << "\n";

        std::cout << "Completed " << wavelet_name << " - F1: " << avg_cls_metrics.f1_score
                  << ", Alpha: " << avg_para_metrics.alpha << std::endl;
    }

    std::cout << "Experiment completed. Results saved to " << config.output_dir << std::endl;
}

auto main(int argc, char const* const* argv) -> int
{
    try
    {
        std::string spec_path = "../src/experiments/02/spec.yaml";
        if (argc > 1)
        {
            spec_path = argv[1];
        }

        ExperimentConfig config = load_experiment_config(spec_path);
        run_wavelet_baseline_experiment(config);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}