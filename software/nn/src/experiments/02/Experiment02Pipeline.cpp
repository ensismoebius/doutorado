#include "Experiment02Pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "Experiment02Evaluation.hpp"
#include "Experiment02Wavelets.hpp"
#include "nn/dataLoaders/mat_file_utils.hpp"
#include "nn/layers/CrossEntropyLoss.hpp"
#include "nn/layers/SimpleResNet.hpp"
#include "nn/linearAlgebra/linear_algebra.hpp"
#include "nn/optimizers/Adam.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/utility/batching.hpp"
#include "nn/wave/audioFeatureExtraction.h"
#include "nn/wavelet/waveletOperations.h"

namespace fs = std::filesystem;

// Windowing functions (using core implementations)
auto hanning_window(int length) -> std::vector<double>
{
    return nn::core::wave::hanning_window(length);
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

// Simple SNN-ResNet classifier (using core implementation)
using SNNResNet = SimpleResNet;

auto k_fold_cross_validation(const std::vector<std::vector<double>>& features,
                             const std::vector<int>& labels, int k_folds, int random_seed)
    -> std::vector<FoldResult>
{
    auto fold_function = [&](const std::vector<std::vector<double>>& train_features,
                             const std::vector<int>& train_labels,
                             const std::vector<std::vector<double>>& test_features,
                             const std::vector<int>& test_labels) -> FoldResult
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        ParaconsistentMetrics para_metrics =
            compute_paraconsistent_metrics(train_features, train_labels);

        int n_classes = 0;
        for (int label : labels) n_classes = std::max(n_classes, label + 1);

        SNNResNet model(train_features[0].size(), 128, n_classes);
        Adam optimizer(0.001F);
        auto params = model.params();
        optimizer.attach(params);
        CrossEntropyLoss loss;

        std::vector<nn::Tensor> train_inputs;
        std::vector<nn::Tensor> train_targets;
        for (std::size_t i = 0; i < train_features.size(); ++i)
        {
            nn::Tensor x(1, static_cast<std::size_t>(train_features[i].size()));
            for (std::size_t j = 0; j < train_features[i].size(); ++j)
            {
                x.at(0, j) = train_features[i][j];
            }
            train_inputs.emplace_back(x);

            nn::Tensor y(1, n_classes);
            y.at(0, train_labels[i]) = 1.0F;
            train_targets.emplace_back(y);
        }

        for (int epoch = 0; epoch < 10; ++epoch)
        {
            auto batches = create_batches(train_inputs, train_targets, 32);
            for (const auto& batch : batches)
            {
                loss.set_target(batch.targets);
                nn::Tensor logits = model.forward(batch.inputs);
                nn::Tensor grad_loss = loss.backward(logits);

                model.backward(grad_loss);
                optimizer.step(params);
            }
        }

        std::vector<int> pred_labels;
        for (const auto& test_feat : test_features)
        {
            nn::Tensor x(1, static_cast<std::size_t>(test_feat.size()));
            for (std::size_t j = 0; j < test_feat.size(); ++j)
            {
                x.at(0, j) = test_feat[j];
            }
            nn::Tensor output = model.forward(x);

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

struct EEGSample
{
    std::vector<std::vector<double>> channels;
    int modality;
    int stimulus;
    int artifacts;
};

struct AudioSample
{
    std::vector<double> signal;
    int stimulus;
    int eeg_index;
};

struct WindowedSample
{
    std::vector<double> eeg_window;
    std::vector<double> audio_window;
    int label;
};

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
        for (int ch = 0; ch < 6; ++ch)
        {
            std::vector<double> channel(4096);
            for (int s = 0; s < 4096; ++s)
            {
                channel[s] = mat.at(i, ch * 4096 + s);
            }
            sample.channels.push_back(channel);
        }
        sample.modality = static_cast<int>(mat.at(i, 24576));
        sample.stimulus = static_cast<int>(mat.at(i, 24577));
        sample.artifacts = static_cast<int>(mat.at(i, 24578));

        samples.push_back(sample);
    }

    return samples;
}

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
        sample.signal.resize(176400);
        for (int s = 0; s < 176400; ++s)
        {
            sample.signal[s] = mat.at(i, s);
        }
        sample.stimulus = static_cast<int>(mat.at(i, 176400));
        sample.eeg_index = static_cast<int>(mat.at(i, 176401));

        samples.push_back(sample);
    }

    return samples;
}

auto extract_windows(const std::vector<EEGSample>& eeg_samples,
                     const std::vector<AudioSample>& audio_samples, double window_duration_sec,
                     double overlap_sec, int eeg_rate, int audio_rate)
    -> std::vector<WindowedSample>
{
    std::vector<WindowedSample> windows;

    for (std::size_t eeg_idx = 0; eeg_idx < eeg_samples.size(); ++eeg_idx)
    {
        const auto& eeg = eeg_samples[eeg_idx];

        const AudioSample* audio = nullptr;
        for (const auto& a : audio_samples)
        {
            if (a.eeg_index == static_cast<int>(eeg_idx))
            {
                audio = &a;
                break;
            }
        }
        if (!audio || eeg.artifacts != 1)
        {
            continue;
        }

        int eeg_window_samples = static_cast<int>(window_duration_sec * eeg_rate);
        int audio_window_samples = static_cast<int>(window_duration_sec * audio_rate);
        int eeg_step = static_cast<int>((window_duration_sec - overlap_sec) * eeg_rate);

        for (int start_eeg = 0; start_eeg + eeg_window_samples <= 4096; start_eeg += eeg_step)
        {
            int start_audio = start_eeg * audio_rate / eeg_rate;
            if (start_audio + audio_window_samples > static_cast<int>(audio->signal.size()))
            {
                break;
            }

            WindowedSample window;
            window.label = eeg.stimulus;

            for (int ch = 0; ch < 6; ++ch)
            {
                for (int s = start_eeg; s < start_eeg + eeg_window_samples; ++s)
                {
                    window.eeg_window.push_back(eeg.channels[ch][s]);
                }
            }

            for (int s = start_audio; s < start_audio + audio_window_samples; ++s)
            {
                window.audio_window.push_back(audio->signal[s]);
            }

            auto hanning_eeg = hanning_window(eeg_window_samples);
            for (std::size_t i = 0; i < window.eeg_window.size(); ++i)
            {
                window.eeg_window[i] *= hanning_eeg[i % eeg_window_samples];
            }

            auto hanning_audio = hanning_window(audio_window_samples);
            for (std::size_t i = 0; i < window.audio_window.size(); ++i)
            {
                window.audio_window[i] *= hanning_audio[i];
            }

            windows.push_back(window);
        }
    }

    return windows;
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

        int n_folds = static_cast<int>(fold_results.size());
        avg_cls_metrics.accuracy /= n_folds;
        avg_cls_metrics.precision /= n_folds;
        avg_cls_metrics.recall /= n_folds;
        avg_cls_metrics.f1_score /= n_folds;
        avg_cls_metrics.mcc /= n_folds;

        avg_para_metrics.alpha /= n_folds;
        avg_para_metrics.beta /= n_folds;
        avg_para_metrics.G1 /= n_folds;
        avg_para_metrics.G2 /= n_folds;

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
