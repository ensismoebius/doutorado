// FsddWindowDataset.cpp — Sliding-window dataset over FSDD WAV files.

#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"

#include <algorithm>
#include <stdexcept>

#include "data_loaders/10.5281/zenodo.1342401/loaders/FsddLoader.hpp"
#include "utility/SignalPreprocessing.hpp"

namespace nn::dataLoaders::fsdd
{

FsddWindowDataset::FsddWindowDataset(const std::filesystem::path& dataset_root, int window_size)
{
    if (window_size <= 0)
        throw std::invalid_argument("FsddWindowDataset: window_size must be > 0");

    const auto files = FsddLoader::discover(dataset_root);
    if (files.empty())
        throw std::runtime_error("FsddWindowDataset: no .wav files found under " +
                                 dataset_root.string());

    for (const auto& file : files)
    {
        const auto info   = FsddLoader::parse_filename(file.stem().string());
        const int  label  = info ? info->digit : -1;
        const nn::Tensor signal = FsddLoader::load_signal(file);

        const auto signal_len = static_cast<std::size_t>(signal.rows());
        std::size_t offset    = 0;

        while (offset < signal_len)
        {
            const std::size_t take = std::min<std::size_t>(
                signal_len - offset, static_cast<std::size_t>(window_size));

            nn::Tensor window(static_cast<nn::Index>(window_size), 1);
            for (int t = 0; t < window_size; ++t)
            {
                if (static_cast<std::size_t>(t) < take)
                    window.at(static_cast<nn::Index>(t), 0) =
                        signal.at(static_cast<nn::Index>(offset + static_cast<std::size_t>(t)), 0);
                // else: zero-pad (default-constructed Tensor is zero)
            }

            nn::utility::zscore_inplace(window);
            windows_.push_back(std::move(window));
            labels_.push_back(label);
            offset += static_cast<std::size_t>(window_size);
        }
    }
}

auto FsddWindowDataset::windows() const -> const std::vector<nn::Tensor>&
{
    return windows_;
}

auto FsddWindowDataset::labels() const -> const std::vector<int>&
{
    return labels_;
}

auto FsddWindowDataset::size() const -> std::size_t
{
    return windows_.size();
}

} // namespace nn::dataLoaders::fsdd
