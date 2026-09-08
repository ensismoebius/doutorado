#pragma once
// FsddWindowDataset.hpp — Sliding-window dataset over all FSDD WAV files.
//
// Discovers every .wav file under dataset_root, loads each signal,
// z-score-normalises each window, and exposes flat windows + digit labels +
// per-window provenance metadata (speaker / recording identity).
//
// The metadata is what makes leakage-safe splitting possible: a caller can
// partition on speaker or recording *before* any window pooling, so windows from
// one recording never straddle a train/val/test boundary.
//
// Usage:
//   FsddWindowDataset ds("/data/fsdd", /*window_size=*/512);
//   const auto& windows = ds.windows();   // vector<nn::Tensor> each (window_size, 1)
//   const auto& labels  = ds.labels();    // vector<int>  digit 0–9 per window
//   const auto& meta    = ds.metadata();  // vector<WindowMetadata> parallel to windows()

#include <filesystem>
#include <string>
#include <vector>

#include "tensor/Tensor.hpp"

namespace nn::dataLoaders::fsdd
{

// Per-window provenance. Parallel to FsddWindowDataset::windows().
//
// speaker_id      — dense 0-based index into the sorted set of distinct speakers
//                   present under the dataset root (stable for a fixed root).
// recording_id    — dense 0-based index into FsddLoader::discover()'s sorted file
//                   list (one id per source WAV).
// window_id       — global 0-based index into windows(); the deterministic key
//                   that lets downstream paired statistics align the same window
//                   across every model.
// source_window_index — position of this window within its own recording (0-based).
struct WindowMetadata
{
    std::string speaker;
    int speaker_id;
    int recording_id;
    int window_id;
    int source_window_index;
    int digit;
};

class FsddWindowDataset
{
   public:
    // Loads all WAV files under dataset_root and slices them into non-overlapping
    // windows of window_size samples. Partial trailing windows are zero-padded.
    //
    // Throws std::invalid_argument if window_size <= 0.
    // Throws std::runtime_error if no WAV files are found, or if any filename stem
    // does not parse as "{digit}_{speaker}_{trial}" — a real FSDD root always
    // parses, and a silent speaker="" fallback would defeat leakage-safe splitting.
    explicit FsddWindowDataset(const std::filesystem::path& dataset_root, int window_size);

    [[nodiscard]] auto windows() const -> const std::vector<nn::Tensor>&;
    [[nodiscard]] auto labels() const -> const std::vector<int>&;
    [[nodiscard]] auto metadata() const -> const std::vector<WindowMetadata>&;
    [[nodiscard]] auto size() const -> std::size_t;

   private:
    std::vector<nn::Tensor> windows_;
    std::vector<int> labels_;
    std::vector<WindowMetadata> metadata_;
};

} // namespace nn::dataLoaders::fsdd
