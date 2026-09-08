// FsddWindowDataset.cpp — Sliding-window dataset over FSDD WAV files.

#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

#include "data_loaders/10.5281/zenodo.1342401/loaders/FsddLoader.hpp"
#include "utility/SignalPreprocessing.hpp"

namespace nn::dataLoaders::fsdd
{

FsddWindowDataset::FsddWindowDataset(const std::filesystem::path& dataset_root, int window_size)
{
    if (window_size <= 0) throw std::invalid_argument("FsddWindowDataset: window_size must be > 0");

    const auto files = FsddLoader::discover(dataset_root);
    if (files.empty())
        throw std::runtime_error(
            "FsddWindowDataset: no .wav files found under " + dataset_root.string());

    // Pass 1: parse every filename up front so a malformed stem fails the whole
    // construction rather than silently producing a speaker="" window that could
    // later leak across a split. Also builds the dense speaker_id map from the
    // sorted set of distinct speakers.
    std::vector<FsddFileInfo> infos;
    infos.reserve(files.size());
    std::set<std::string> distinct_speakers;
    for (const auto& file : files)
    {
        const auto info = FsddLoader::parse_filename(file.stem().string());
        if (!info)
            throw std::runtime_error(
                "FsddWindowDataset: filename does not parse as {digit}_{speaker}_{trial}: " +
                file.string() +
                " — leakage-safe splitting requires a speaker for every window; "
                "fix the dataset root or the offending file name.");
        infos.push_back(*info);
        distinct_speakers.insert(info->speaker);
    }

    std::map<std::string, int> speaker_id;
    {
        int next_id = 0;
        for (const auto& s : distinct_speakers) speaker_id.emplace(s, next_id++);
    }

    // Pass 2: window each recording.
    int global_window_id = 0;
    for (std::size_t rec = 0; rec < files.size(); ++rec)
    {
        const FsddFileInfo& info = infos[rec];
        const nn::Tensor signal = FsddLoader::load_signal(files[rec]);

        const auto signal_len = static_cast<std::size_t>(signal.rows());
        std::size_t offset = 0;
        int source_window_idx = 0;

        while (offset < signal_len)
        {
            const std::size_t take =
                std::min<std::size_t>(signal_len - offset, static_cast<std::size_t>(window_size));

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
            labels_.push_back(info.digit);
            metadata_.push_back(WindowMetadata{
                info.speaker,
                speaker_id.at(info.speaker),
                static_cast<int>(rec),
                global_window_id,
                source_window_idx,
                info.digit,
            });

            ++global_window_id;
            ++source_window_idx;
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

auto FsddWindowDataset::metadata() const -> const std::vector<WindowMetadata>&
{
    return metadata_;
}

auto FsddWindowDataset::size() const -> std::size_t
{
    return windows_.size();
}

} // namespace nn::dataLoaders::fsdd
