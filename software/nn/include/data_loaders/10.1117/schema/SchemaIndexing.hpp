/**
 * @file include/nn/dataLoaders/10.1117/schema/SchemaIndexing.hpp
 * @brief Schemaindexing.
 *
 *
 *
 * **Contract:**
 * - Public APIs should document behavior, inputs, outputs, and exceptions.
 * - Prefer RAII for resource lifecycle when applicable.
 */

#ifndef NN_DATALOADERS_10_1117_SCHEMAINDEXING_HPP
#define NN_DATALOADERS_10_1117_SCHEMAINDEXING_HPP

#include <cstddef>
#include <stdexcept>
#include <string>

#include "data_loaders/10.1117/schema/Metadata.hpp"

namespace nn::dataLoaders::schema101117
{

constexpr auto eegFeatureColumns() -> std::size_t
{
    return ImaginedSpeechSchema_10_1117.eegSignalColumns();
}

constexpr auto multimodalInputFeatureColumns() -> std::size_t
{
    return eegFeatureColumns() + ImaginedSpeechSchema_10_1117.audioSamples();
}

constexpr auto eegSignalFlatColumn(std::size_t channel, std::size_t sample) -> std::size_t
{
    return (channel * ImaginedSpeechSchema_10_1117.eegSamplesPerChannel()) + sample;
}

constexpr auto columnMajorIndex(std::size_t column, std::size_t row, std::size_t rowCount)
    -> std::size_t
{
    return (column * rowCount) + row;
}

inline auto resolveEegRowIndex(int eeg_index_label, std::size_t eeg_rows) -> std::size_t
{
    if (eeg_rows == 0)
    {
        throw std::runtime_error("Invalid EEG row count: 0");
    }

    if (eeg_index_label >= 1 && static_cast<std::size_t>(eeg_index_label) <= eeg_rows)
    {
        return static_cast<std::size_t>(eeg_index_label - 1);
    }

    if (eeg_index_label >= 0 && static_cast<std::size_t>(eeg_index_label) < eeg_rows)
    {
        return static_cast<std::size_t>(eeg_index_label);
    }

    throw std::runtime_error(
        "Audio->EEG index label is out of range: " + std::to_string(eeg_index_label));
}

} // namespace nn::dataLoaders::schema101117

#endif // NN_DATALOADERS_10_1117_SCHEMAINDEXING_HPP
