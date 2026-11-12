/**
 * @file include/data_loaders/10.1117/schema/SchemaIndexing.hpp
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

/// The EEG row of an audio row's label. The label IS the 0-based EEG row, in both sources:
/// the SQLite stores it as trial.original_row (the trial's own row; EEGSession::readRow(r)
/// reads original_row = r), and the MAT column holds the same value -- the importer used it
/// unchanged as a 0-based key, and on the real database the trials that received audio are
/// exactly the 1974 Pronounced ones and none of the 7341 Imagined (checked 2026-11-12).
/// This function used to map a label k >= 1 to k - 1 and 0 to 0, so every pair except row 0
/// read the PREVIOUS trial (.wiki/Core/DataLoaders.md, "Audio -> EEG pairing").
/// Throws for -1 (an audio-only trial: no EEG was recorded) and for a label past the rows.
inline auto resolveEegRowIndex(int eeg_index_label, std::size_t eeg_rows) -> std::size_t
{
    if (eeg_rows == 0)
    {
        throw std::runtime_error("Invalid EEG row count: 0");
    }
    if (eeg_index_label < 0)
    {
        throw std::runtime_error("Audio->EEG index label " + std::to_string(eeg_index_label) +
                                 ": the audio trial has no EEG (audio-only); a paired reader "
                                 "cannot use it");
    }
    if (static_cast<std::size_t>(eeg_index_label) >= eeg_rows)
    {
        throw std::runtime_error("Audio->EEG index label " + std::to_string(eeg_index_label) +
                                 " is past the subject's " + std::to_string(eeg_rows) +
                                 " EEG rows (labels are 0-based EEG rows)");
    }
    return static_cast<std::size_t>(eeg_index_label);
}

} // namespace nn::dataLoaders::schema101117

#endif // NN_DATALOADERS_10_1117_SCHEMAINDEXING_HPP
