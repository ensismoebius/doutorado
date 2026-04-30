#include "../include/ComparativeDataset.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

#include "nn/utility/SignalPreprocessing.hpp"
#include "nn/wave/Wav.h"

namespace comparative_autoencoder_experiment
{

auto to_window_tensor(const nn::Tensor& signal, int window_size) -> std::vector<nn::Tensor>
{
    std::vector<nn::Tensor> windows;
    if (window_size <= 0 || signal.size() == 0) return windows;

    std::size_t offset = 0;
    const std::size_t signal_len = static_cast<std::size_t>(signal.rows());
    while (offset < signal_len)
    {
        const std::size_t remaining = signal_len - offset;
        const std::size_t take =
            std::min<std::size_t>(remaining, static_cast<std::size_t>(window_size));

        // Produce 2D {window_size, 1} tensors — expected by Linear/LSTM layers.
        nn::Tensor sample(static_cast<nn::Index>(window_size), 1);
        for (int t = 0; t < window_size; ++t)
        {
            if (static_cast<std::size_t>(t) < take)
            {
                sample.at(static_cast<nn::Index>(t), 0) =
                    signal.at(static_cast<nn::Index>(offset + static_cast<std::size_t>(t)), 0);
            }
        }

        nn::utility::zscore_inplace(sample);
        windows.push_back(std::move(sample));
        offset += static_cast<std::size_t>(window_size);
    }

    return windows;
}

auto collect_signal_files(const ComparativeConfig& cfg, const std::string& dataset)
    -> std::vector<std::filesystem::path>
{
    namespace fs = std::filesystem;
    const fs::path root = fs::path(cfg.dataset.dataset_root);
    if (!fs::exists(root))
    {
        throw std::runtime_error("Dataset root does not exist: " + root.string());
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(root))
    {
        if (!entry.is_regular_file()) continue;
        const std::string path_str = entry.path().string();
        const std::string ext = entry.path().extension().string();

        if (dataset == "fsdd")
        {
            if (ext == ".wav")
            {
                files.push_back(entry.path());
            }
        }
        else if (dataset == "physionet")
        {
            if ((ext == ".csv" || ext == ".txt") &&
                (path_str.find("physionet") != std::string::npos ||
                    path_str.find("PhysioNet") != std::string::npos))
            {
                files.push_back(entry.path());
            }
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

auto build_split(const ComparativeConfig& cfg, const std::string& dataset) -> DatasetSplit
{
    DatasetSplit split;
    const auto files = collect_signal_files(cfg, dataset);

    if (files.empty())
    {
        throw std::runtime_error("No files found for dataset token: " + dataset);
    }

    std::vector<Tensor> all_samples;
    for (const auto& file : files)
    {
        nn::Tensor signal;
        if (dataset == "fsdd")
        {
            Wav wav_file;
            wav_file.read(file.string());
            const auto& raw_data = wav_file.get_data();
            signal = nn::Tensor(static_cast<nn::Index>(raw_data.size()), 1);
            for (std::size_t i = 0; i < raw_data.size(); ++i)
            {
                signal.at(static_cast<nn::Index>(i), 0) = static_cast<float>(raw_data[i]);
            }
        }
        else
        {
            signal = nn::utility::read_csv_signal(file);
        }
        const auto windows = to_window_tensor(signal, cfg.dataset.window_size);
        all_samples.insert(all_samples.end(), windows.begin(), windows.end());
    }

    if (all_samples.empty())
    {
        throw std::runtime_error("No windows created for dataset token: " + dataset);
    }

    const std::size_t max_total =
        static_cast<std::size_t>(cfg.dataset.max_loaded_train_samples + cfg.dataset.max_validation_samples);
    if (all_samples.size() > max_total)
    {
        all_samples.resize(max_total);
    }

    const std::size_t val_count = std::min<std::size_t>(cfg.dataset.max_validation_samples, all_samples.size() / 5);
    const std::size_t train_count = all_samples.size() - val_count;

    split.train_samples.assign(all_samples.begin(), all_samples.begin() + static_cast<long>(train_count));
    split.val_samples.assign(all_samples.begin() + static_cast<long>(train_count), all_samples.end());
    split.val_labels.assign(split.val_samples.size(), 0);

    for (std::size_t i = 0; i < split.val_samples.size(); ++i)
    {
        if (i % 10 != 0) continue;
        split.val_labels[i] = 1;
        Tensor& sample = split.val_samples[i];
        for (nn::Index t = 0; t < sample.size(); ++t)
        {
            sample.at(t) += 1.5f;
        }
    }

    return split;
}

} // namespace comparative_autoencoder_experiment

