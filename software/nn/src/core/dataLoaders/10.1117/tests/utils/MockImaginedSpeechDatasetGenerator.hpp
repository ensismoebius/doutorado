#pragma once

#include <cstddef>
#include <filesystem>

namespace nn::dataLoaders::test
{

struct EEGCorruptionOptions
{
    bool wrong_column_count = false;
    bool wrong_channel_count = false;
    bool missing_labels = false;
};

struct AudioCorruptionOptions
{
    bool wrong_column_count = false;
    bool wrong_signal_length = false;
    bool missing_labels = false;
};

class MockImaginedSpeechDatasetGenerator
{
   public:
    static void generateEEGMatFile(const std::filesystem::path& path, std::size_t trials);

    static void generateAudioMatFile(const std::filesystem::path& path, std::size_t trials);

    static void generateDataset(const std::filesystem::path& directory, std::size_t trials);

    static void generateCorruptedEEGMatFile(
        const std::filesystem::path& path, std::size_t trials, EEGCorruptionOptions options);

    static void generateCorruptedAudioMatFile(
        const std::filesystem::path& path, std::size_t trials, AudioCorruptionOptions options);
};

} // namespace nn::dataLoaders::test
