#pragma once
// FsddLoader.hpp — Discover and load WAV files from an FSDD dataset root.
//
// Usage:
//   auto files  = FsddLoader::discover("/data/fsdd");
//   auto info   = FsddLoader::parse_filename("0_jackson_3");  // optional<FsddFileInfo>
//   nn::Tensor  signal = FsddLoader::load_signal(files[0]);   // (N, 1) float32 column vector

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "tensor/Tensor.hpp"
#include "data_loaders/10.5281/zenodo.1342401/loaders/FsddSample.hpp"

namespace nn::dataLoaders::fsdd
{

class FsddLoader
{
public:
    // Returns sorted list of all .wav paths under root (recursive).
    // Throws std::runtime_error if root does not exist.
    static auto discover(const std::filesystem::path& root)
        -> std::vector<std::filesystem::path>;

    // Parses a filename stem (no extension) into FsddFileInfo.
    // Returns std::nullopt for any stem that does not match {digit}_{speaker}_{trial}.
    static auto parse_filename(const std::string& stem)
        -> std::optional<FsddFileInfo>;

    // Reads a WAV file and returns a (N, 1) column-vector tensor of float32 samples.
    // Throws std::runtime_error on read failure.
    static auto load_signal(const std::filesystem::path& wav_path)
        -> nn::Tensor;
};

} // namespace nn::dataLoaders::fsdd
