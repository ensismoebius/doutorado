// FsddLoader.cpp — Implementation of FsddLoader (DOI: 10.5281/zenodo.1342401).

#include "data_loaders/10.5281/zenodo.1342401/loaders/FsddLoader.hpp"

#include <algorithm>
#include <stdexcept>

#include "wave/Wav.hpp"

namespace nn::dataLoaders::fsdd
{

auto FsddLoader::discover(const std::filesystem::path& root)
    -> std::vector<std::filesystem::path>
{
    if (!std::filesystem::exists(root))
        throw std::runtime_error("FSDD root does not exist: " + root.string());

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".wav")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

auto FsddLoader::parse_filename(const std::string& stem) -> std::optional<FsddFileInfo>
{
    // Expected: "{digit}_{speaker}_{trial}"  e.g. "0_jackson_3"
    const auto first = stem.find('_');
    if (first == std::string::npos) return std::nullopt;

    const auto last = stem.rfind('_');
    if (last == first) return std::nullopt;  // only one underscore

    FsddFileInfo info;
    try
    {
        info.digit   = std::stoi(stem.substr(0, first));
        info.speaker = stem.substr(first + 1, last - first - 1);
        info.trial   = std::stoi(stem.substr(last + 1));
    }
    catch (...)
    {
        return std::nullopt;
    }

    if (info.digit < 0 || info.digit > 9 || info.trial < 0 || info.speaker.empty())
        return std::nullopt;

    return info;
}

auto FsddLoader::load_signal(const std::filesystem::path& wav_path) -> nn::Tensor
{
    Wav wav;
    wav.read(wav_path.string());
    const auto& raw = wav.get_data();
    if (raw.empty())
        throw std::runtime_error("Empty WAV file: " + wav_path.string());

    nn::Tensor signal(static_cast<nn::Index>(raw.size()), 1);
    for (std::size_t i = 0; i < raw.size(); ++i)
        signal.at(static_cast<nn::Index>(i), 0) = static_cast<float>(raw[i]);
    return signal;
}

} // namespace nn::dataLoaders::fsdd
