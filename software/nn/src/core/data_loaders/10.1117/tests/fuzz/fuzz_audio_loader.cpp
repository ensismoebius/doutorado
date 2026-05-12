/**
 * @file src/core/dataLoaders/10.1117/tests/fuzz/fuzz_audio_loader.cpp
 * @brief Implementation for Fuzz audio loader.
 *

 */

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

#include "data_loaders/10.1117/loaders/AudioLoader.hpp"
#include "data_loaders/10.1117/schema/Metadata.hpp"

namespace
{
auto write_input_to_temp_file(const uint8_t* data, size_t size, const char* prefix) -> std::string
{
    const auto temp_dir = std::filesystem::temp_directory_path();
    const auto file_path =
        (temp_dir / (std::string(prefix) + std::to_string(::getpid()) + ".mat")).string();

    FILE* f = std::fopen(file_path.c_str(), "wb");
    if (f == nullptr)
    {
        return {};
    }
    if (size > 0)
    {
        std::fwrite(data, 1, size, f);
    }
    std::fclose(f);
    return file_path;
}
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (data == nullptr)
    {
        return 0;
    }

    const auto path = write_input_to_temp_file(data, size, "fuzz_audio_loader_");
    if (path.empty())
    {
        return 0;
    }

    try
    {
        auto [audio_tensor, stimulus, eeg_index] = nn::dataLoaders::loadAudioFromMat(path, 0);
        (void) stimulus;
        (void) eeg_index;

        const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
        const bool shape_valid =
            (audio_tensor.rows() == schema.audioSamples()) && (audio_tensor.cols() == 1);

        if (!shape_valid)
        {
            // For malformed corpus entries, parser may return early or throw.
        }
    }
    catch (const std::exception&)
    {
        // Controlled failures are expected for malformed MAT payloads.
    }
    catch (...)
    {
        // Unknown exceptions are still non-crashing outcomes from the fuzzer perspective.
    }

    std::filesystem::remove(path);
    return 0;
}
