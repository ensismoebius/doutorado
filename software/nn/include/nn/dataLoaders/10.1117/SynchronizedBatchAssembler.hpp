#ifndef NN_DATALOADERS_10_1117_SYNCHRONIZEDBATCHASSEMBLER_HPP
#define NN_DATALOADERS_10_1117_SYNCHRONIZEDBATCHASSEMBLER_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include "nn/dataLoaders/10.1117/AudioLoader.h"
#include "nn/dataLoaders/10.1117/EEGLoader.h"
#include "nn/dataLoaders/10.1117/SubjectDiscovery.hpp"
#include "nn/tensor/Tensor.hpp"

struct RowRequest
{
    std::size_t batch_row;
    std::size_t local_audio_row;
};

class SynchronizedBatchAssembler
{
   public:
    /**
     * Assemble a synchronized batch for grouped per-subject requests.
     *
     * For each subject this function:
     *  - sorts the per-subject `RowRequest` values by local audio row;
     *  - bulk-reads contiguous audio rows and builds per-row tasks from the
     *    flat audio buffer (stimulus, eeg index label, resolved eeg_row);
     *  - sorts the tasks by `eeg_row`, bulk-reads contiguous EEG blocks and
     *    copies EEG+audio slices into `inputs` while filling `targets` metadata.
     *
     * This implementation optimizes disk I/O by reading contiguous ranges
     * of rows from the audio and EEG MAT sessions instead of performing
     * per-row reads.
     *
     * @param grouped       Per-subject vectors of `RowRequest` describing which
     *                      batch rows and local audio rows to assemble.
     * @param subjects      Vector of `SubjectFiles` (subject metadata and row indices).
     * @param audio_sessions Per-subject `AudioMatSession` handles (opened session objects).
     * @param eeg_sessions  Per-subject `EEGMatSession` handles (opened session objects).
     * @param[out] inputs   Tensor (batch x features) that will be filled with EEG+audio rows.
     * @param[out] targets  Tensor (batch x 5) that will be filled with metadata for each row.
     * @throws std::runtime_error if a stimulus label mismatch is detected between
     *                            audio and EEG data for a matched pair.
     */
    static void assembleGrouped(
        const std::vector<std::vector<RowRequest>>& grouped,
        const std::vector<SubjectFiles>& subjects,
        const std::vector<std::unique_ptr<nn::dataLoaders::AudioMatSession>>& audio_sessions,
        const std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>>& eeg_sessions,
        nn::Tensor& inputs, nn::Tensor& targets);
};

#endif // NN_DATALOADERS_10_1117_SYNCHRONIZEDBATCHASSEMBLER_HPP