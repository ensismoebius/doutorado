#include "../include/E04Dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>

#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"
#include "utility/SignalPreprocessing.hpp"

namespace e04
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

auto collect_signal_files(const E04Config& cfg, const std::string& dataset)
    -> std::vector<std::filesystem::path>
{
    namespace fs = std::filesystem;
    const fs::path root = fs::path(cfg.dataset.dataset_root);
    if (!fs::exists(root))
        throw std::runtime_error("Dataset root does not exist: " + root.string());

    std::vector<fs::path> files;
    for (const auto& entry : fs::recursive_directory_iterator(root))
    {
        if (!entry.is_regular_file()) continue;
        const std::string path_str = entry.path().string();
        const std::string ext      = entry.path().extension().string();

        if ((ext == ".csv" || ext == ".txt") &&
            (path_str.find("physionet") != std::string::npos ||
             path_str.find("PhysioNet") != std::string::npos))
        {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

auto build_split(const E04Config& cfg, const std::string& dataset) -> DatasetSplit
{
    DatasetSplit       split;
    std::vector<Tensor> all_samples;
    std::vector<int>    all_labels;

    if (dataset == "fsdd")
    {
        nn::dataLoaders::fsdd::FsddWindowDataset ds(cfg.dataset.dataset_root,
                                                     cfg.dataset.window_size);
        all_samples = ds.windows();
        all_labels  = ds.labels();
    }
    else
    {
        const auto files = collect_signal_files(cfg, dataset);
        if (files.empty())
            throw std::runtime_error("No files found for dataset token: " + dataset);

        for (const auto& file : files)
        {
            const auto signal  = nn::utility::read_csv_signal(file);
            const auto windows = to_window_tensor(signal, cfg.dataset.window_size);
            all_samples.insert(all_samples.end(), windows.begin(), windows.end());
        }
        all_labels.assign(all_samples.size(), 0);
    }

    if (all_samples.empty())
        throw std::runtime_error("No windows created for dataset token: " + dataset);

    {
        std::vector<std::size_t> idx(all_samples.size());
        std::iota(idx.begin(), idx.end(), 0u);
        std::mt19937 rng(cfg.experiment.seed != 0u ? cfg.experiment.seed : 42u);
        std::shuffle(idx.begin(), idx.end(), rng);

        std::vector<Tensor> s(all_samples.size());
        std::vector<int>    l(all_labels.size());
        for (std::size_t i = 0; i < idx.size(); ++i)
        {
            s[i] = std::move(all_samples[idx[i]]);
            l[i] = all_labels[idx[i]];
        }
        all_samples = std::move(s);
        all_labels  = std::move(l);
    }

    const std::size_t max_total =
        static_cast<std::size_t>(cfg.dataset.max_loaded_train_samples +
                                 cfg.dataset.max_validation_samples);
    if (all_samples.size() > max_total)
    {
        all_samples.resize(max_total);
        all_labels.resize(max_total);
    }

    const std::size_t val_count =
        std::min<std::size_t>(cfg.dataset.max_validation_samples, all_samples.size());
    const std::size_t train_count = all_samples.size() - val_count;

    split.train_samples.assign(all_samples.begin(),
                               all_samples.begin() + static_cast<long>(train_count));
    split.val_samples.assign(all_samples.begin() + static_cast<long>(train_count),
                             all_samples.end());
    split.val_labels.assign(all_labels.begin() + static_cast<long>(train_count),
                            all_labels.end());

    return split;
}

} // namespace e04
