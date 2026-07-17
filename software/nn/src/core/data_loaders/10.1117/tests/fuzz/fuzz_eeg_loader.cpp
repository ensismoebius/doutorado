/**
 * @file src/core/data_loaders/10.1117/tests/fuzz/fuzz_eeg_loader.cpp
 * @brief Implementation for Fuzz eeg loader.
 *

 */

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

#include "data_loaders/10.1117/loaders/EEGLoader.hpp"
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

    const auto path = write_input_to_temp_file(data, size, "fuzz_eeg_loader_");
    if (path.empty())
    {
        return 0;
    }

    try
    {
        auto [eeg_tensor, labels] = nn::dataLoaders::loadEEGFromMat(path, 0);
        (void) labels;

        const auto& schema = nn::dataLoaders::ImaginedSpeechSchema_10_1117;
        const bool shape_valid = (eeg_tensor.rows() == schema.eeg_channels) &&
                                 (eeg_tensor.cols() == schema.eegSamplesPerChannel());

        // The loader may accept valid inputs from corpus seeds; never crash either way.
        if (!shape_valid)
        {
            // Keep behavior side-effect free; invalid shape from malformed corpus is acceptable.
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
