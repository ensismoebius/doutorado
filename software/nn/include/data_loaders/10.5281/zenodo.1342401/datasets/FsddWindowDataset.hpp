#pragma once
// FsddWindowDataset.hpp — Sliding-window dataset over all FSDD WAV files.
//
// Discovers every .wav file under dataset_root, loads each signal,
// z-score-normalises each window, and exposes flat windows + digit labels.
//
// Usage:
//   FsddWindowDataset ds("/data/fsdd", /*window_size=*/512);
//   const auto& windows = ds.windows();  // vector<nn::Tensor> each (window_size, 1)
//   const auto& labels  = ds.labels();   // vector<int>  digit 0–9 per window

#include <filesystem>
#include <vector>

#include "tensor/Tensor.hpp"

namespace nn::dataLoaders::fsdd
{

class FsddWindowDataset
{
public:
    // Loads all WAV files under dataset_root and slices them into non-overlapping
    // windows of window_size samples.  Partial trailing windows are zero-padded.
    // window_size must be > 0, throws std::invalid_argument otherwise.
    explicit FsddWindowDataset(const std::filesystem::path& dataset_root, int window_size);

    [[nodiscard]] auto windows() const -> const std::vector<nn::Tensor>&;
    [[nodiscard]] auto labels()  const -> const std::vector<int>&;
    [[nodiscard]] auto size()    const -> std::size_t;

private:
    std::vector<nn::Tensor> windows_;
    std::vector<int>        labels_;
};

} // namespace nn::dataLoaders::fsdd
