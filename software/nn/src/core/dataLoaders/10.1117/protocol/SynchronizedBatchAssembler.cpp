#include "nn/dataLoaders/10.1117/protocol/SynchronizedBatchAssembler.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "nn/dataLoaders/10.1117/schema/METADATA.hpp"
#include "nn/dataLoaders/10.1117/protocol/SamplePacking.hpp"
#include "nn/dataLoaders/10.1117/schema/SchemaIndexing.hpp"

using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
using nn::dataLoaders::schema101117::eegFeatureColumns;
using nn::dataLoaders::schema101117::resolveEegRowIndex;
using std::iota;
using std::runtime_error;
using std::size_t;
using std::sort;
using std::unique_ptr;
using std::vector;

namespace
{

constexpr std::size_t EEG_FEATURES = eegFeatureColumns();

struct BatchTask
{
    size_t batch_row;
    size_t audio_index;
    int audio_stimulus;
    int eeg_index_label;
    size_t eeg_row;
};

// Build `tasks` for a contiguous audio run: extract per-row stimulus and
// eeg-index label from the flat audio read result and resolve the EEG row
// index for later grouping. Template is used to avoid including the
// concrete `AudioRowsFlat` type in this translation unit.
template <typename AudioRowsFlatT>
static void buildTasksFromAudioRun(const AudioRowsFlatT& audio_rows_flat,
    const std::vector<RowRequest>& requests,
    size_t audio_run_pos,
    size_t audio_run_count,
    const SubjectFiles& subject,
    std::vector<BatchTask>& tasks)
{
    for (size_t audio_index = 0; audio_index < audio_run_count; ++audio_index)
    {
        const RowRequest& req = requests[audio_run_pos + audio_index];
        const int audio_stimulus = audio_rows_flat.stimuli[audio_index];
        const int eeg_index_label = audio_rows_flat.eegIndices[audio_index];
        const size_t eeg_row = resolveEegRowIndex(eeg_index_label, subject.eeg_rows);

        tasks.push_back(
            BatchTask{req.batch_row, audio_index, audio_stimulus, eeg_index_label, eeg_row});
    }
}

// Given the `tasks` built from an audio run, read contiguous EEG blocks,
// reconstruct per-channel EEG, merge and stack resampled signals,, channel-preserving
// flattened rows: it takes the raw EEG row (flattened per-channel) and the
// audio row, reconstructs the per-channel EEG matrix, calls
// `mergeAudioAndEEGSignals` to resample and stack rows, then flattens the
// stacked tensor into `inputs` at `task.batch_row`.
template <typename AudioRowsFlatT>
static void processEegBlocksForTasks(size_t subject_index,
    const SubjectFiles& subject,
    const std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>>& eeg_sessions,
    const AudioRowsFlatT& audio_rows_flat,
    std::vector<BatchTask>& tasks,
    nn::Tensor& inputs,
    nn::Tensor& targets)
{
    std::vector<std::size_t> eeg_sorted_task_indices(tasks.size());
    std::iota(eeg_sorted_task_indices.begin(), eeg_sorted_task_indices.end(), 0);

    std::sort(eeg_sorted_task_indices.begin(),
        eeg_sorted_task_indices.end(),
        [&tasks](std::size_t a, std::size_t b) { return tasks[a].eeg_row < tasks[b].eeg_row; });

    const size_t audioCols = ImaginedSpeechSchema_10_1117.audioSamples();
    const size_t eeg_channels = ImaginedSpeechSchema_10_1117.eeg_channels;
    const size_t eeg_samples_per_channel = ImaginedSpeechSchema_10_1117.eegSamplesPerChannel();
    const size_t stacked_rows = eeg_channels + 1U; // audio + channels
    const size_t stacked_cols = audioCols;

    std::size_t eeg_block_pos = 0;
    while (eeg_block_pos < eeg_sorted_task_indices.size())
    {
        std::size_t eeg_block_end = eeg_block_pos + 1;
        while (eeg_block_end < eeg_sorted_task_indices.size() &&
               tasks[eeg_sorted_task_indices[eeg_block_end]].eeg_row ==
                   tasks[eeg_sorted_task_indices[eeg_block_end - 1]].eeg_row + 1)
        {
            ++eeg_block_end;
        }

        const std::size_t eeg_run_start = tasks[eeg_sorted_task_indices[eeg_block_pos]].eeg_row;
        const std::size_t eeg_run_count = eeg_block_end - eeg_block_pos;

        const auto eeg_rows_flat =
            eeg_sessions.at(subject_index)->readRowsFlat(eeg_run_start, eeg_run_count);

        // Temporary buffers for reconstructing per-channel EEG and audio columns.
        // We reuse the same allocation per task.
        nn::Tensor eeg_matrix(eeg_channels, eeg_samples_per_channel);
        nn::Tensor audio_col(audioCols, 1);

        for (size_t j = 0; j < eeg_run_count; ++j)
        {
            const BatchTask& task = tasks[eeg_sorted_task_indices[eeg_block_pos + j]];
            const size_t audioOffset = task.audio_index * audioCols;
            const size_t eegOffset = j * EEG_FEATURES;
            const auto& eeg_labels = eeg_rows_flat.labels[j];

            const int stimulus_label = eeg_labels[1];
            if (task.audio_stimulus != stimulus_label)
            {
                throw runtime_error("Stimulus mismatch between audio and EEG in collate");
            }

            // Rebuild per-channel EEG matrix from flat signals.
            for (size_t ch = 0; ch < eeg_channels; ++ch)
            {
                const size_t channel_offset = ch * eeg_samples_per_channel;
                for (size_t s = 0; s < eeg_samples_per_channel; ++s)
                {
                    eeg_matrix.at(ch, s) = eeg_rows_flat.signals[eegOffset + channel_offset + s];
                }
            }

            // Build audio column
            for (size_t c = 0; c < audioCols; ++c)
            {
                audio_col.at(c, 0) = audio_rows_flat.samples[audioOffset + c];
            }

            // Merge & resample into stacked rows (audio first, then channels)
            const nn::Tensor stacked = mergeAudioAndEEGSignals(eeg_matrix, audio_col);

            // Flatten stacked into the single inputs row
            size_t write_index = 0;
            for (size_t r = 0; r < stacked_rows; ++r)
            {
                for (size_t c = 0; c < stacked_cols; ++c)
                {
                    inputs.at(task.batch_row, write_index++) = stacked.at(r, c);
                }
            }

            targets.at(task.batch_row, 0) = static_cast<float>(subject.subject_id);
            targets.at(task.batch_row, 1) = static_cast<float>(eeg_labels[0]);
            targets.at(task.batch_row, 2) = static_cast<float>(eeg_labels[1]);
            targets.at(task.batch_row, 3) = static_cast<float>(eeg_labels[2]);
            targets.at(task.batch_row, 4) = static_cast<float>(task.eeg_index_label);
        }

        eeg_block_pos = eeg_block_end;
    }
}

} // namespace

void SynchronizedBatchAssembler::assembleGrouped(                                         //
    const std::vector<std::vector<RowRequest>>& grouped,                                  //
    const std::vector<SubjectFiles>& subjects,                                            //
    const std::vector<std::unique_ptr<nn::dataLoaders::AudioMatSession>>& audio_sessions, //
    const std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>>& eeg_sessions,     //
    nn::Tensor& inputs,
    nn::Tensor& targets //
)
{
    for (size_t subject_index = 0; subject_index < grouped.size(); ++subject_index)
    {
        const auto& group = grouped[subject_index];
        if (group.empty()) continue;

        vector<RowRequest> requests = group;
        sort(requests.begin(),
            requests.end(),
            [](const RowRequest& a, const RowRequest& b)
            { return a.local_audio_row < b.local_audio_row; });

        size_t audio_run_pos = 0;
        while (audio_run_pos < requests.size())
        {
            size_t audio_run_end = audio_run_pos + 1;
            while (audio_run_end < requests.size() &&
                   requests[audio_run_end].local_audio_row ==
                       requests[audio_run_end - 1].local_audio_row + 1)
            {
                ++audio_run_end;
            }

            const size_t audio_run_count = audio_run_end - audio_run_pos;
            const size_t audio_run_start = requests[audio_run_pos].local_audio_row;

            const auto audio_rows_flat = audio_sessions.at(subject_index)
                                             ->readRowsFlat(      //
                                                 audio_run_start, //
                                                 audio_run_count  //
                                             );

            vector<BatchTask> audio_to_eeg_tasks;
            audio_to_eeg_tasks.reserve(audio_run_count);
            const SubjectFiles& subject = subjects.at(subject_index);

            buildTasksFromAudioRun( //
                audio_rows_flat,    //
                requests,           //
                audio_run_pos,      //
                audio_run_count,    //
                subject,            //
                audio_to_eeg_tasks  //
            );

            // Use stacked variant which reconstructs per-channel EEG, merges/resamples
            // with audio, and flattens the stacked rows directly into `inputs`.
            processEegBlocksForTasks( //
                subject_index,        //
                subject,              //
                eeg_sessions,         //
                audio_rows_flat,      //
                audio_to_eeg_tasks,   //
                inputs,               //
                targets               //
            );

            audio_run_pos = audio_run_end;
        }
    }
}
