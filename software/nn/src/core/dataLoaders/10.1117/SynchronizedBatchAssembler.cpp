#include "nn/dataLoaders/10.1117/SynchronizedBatchAssembler.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "nn/dataLoaders/10.1117/SchemaIndexing.hpp"

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
                                   const std::vector<RowRequest>& requests, size_t audio_run_pos,
                                   size_t audio_run_count, const SubjectFiles& subject,
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
// copy EEG and audio slices into `inputs` and fill `targets` accordingly.
template <typename AudioRowsFlatT>
static void processEegBlocksForTasks(
    size_t subject_index, const SubjectFiles& subject,
    const std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>>& eeg_sessions,
    const AudioRowsFlatT& audio_rows_flat, std::vector<BatchTask>& tasks, nn::Tensor& inputs,
    nn::Tensor& targets)
{
    // Permutation of task indices used to sort tasks by `eeg_row`.
    std::vector<std::size_t> order(tasks.size());
    std::iota(order.begin(), order.end(), 0);

    // Sort permutation by `eeg_row` so contiguous EEG rows can be
    // bulk-read from disk in a single call.
    std::sort(order.begin(),
              order.end(),
              [&tasks](std::size_t a, std::size_t b)
              { return tasks[a].eeg_row < tasks[b].eeg_row; });

    const size_t audioCols = ImaginedSpeechSchema_10_1117.audioSamples();

    std::size_t op = 0;
    while (op < order.size())
    {
        // Find contiguous run of eeg_row values in the sorted tasks.
        std::size_t op_end = op + 1;
        while (op_end < order.size() &&
               tasks[order[op_end]].eeg_row == tasks[order[op_end - 1]].eeg_row + 1)
        {
            ++op_end;
        }

        const std::size_t eeg_run_start = tasks[order[op]].eeg_row;
        const std::size_t eeg_run_count = op_end - op;

        const auto eeg_rows_flat =
            eeg_sessions.at(subject_index)->readRowsFlat(eeg_run_start, eeg_run_count);

        // One-row temporaries used for copying into `inputs`.
        nn::Tensor eegRow(1, EEG_FEATURES);
        nn::Tensor audioRow(1, audioCols);
        float* eegDst = eegRow.mutable_data_ptr();
        float* audioDst = audioRow.mutable_data_ptr();

        for (size_t j = 0; j < eeg_run_count; ++j)
        {
            const BatchTask& task = tasks[order[op + j]];
            const size_t audioOffset = task.audio_index * audioCols;
            const size_t eegOffset = j * EEG_FEATURES;
            const auto& eeg_labels = eeg_rows_flat.labels[j];

            const int stimulus_label = eeg_labels[1];
            if (task.audio_stimulus != stimulus_label)
            {
                throw runtime_error("Stimulus mismatch between audio and EEG in collate");
            }

            for (size_t c = 0; c < EEG_FEATURES; ++c)
            {
                eegDst[c] = eeg_rows_flat.signals[eegOffset + c];
            }

            for (size_t c = 0; c < audioCols; ++c)
            {
                audioDst[c] = audio_rows_flat.samples[audioOffset + c];
            }

            inputs.setBlock(task.batch_row, 0, eegRow);
            inputs.setBlock(task.batch_row, EEG_FEATURES, audioRow);

            targets.at(task.batch_row, 0) = static_cast<float>(subject.subject_id);
            targets.at(task.batch_row, 1) = static_cast<float>(eeg_labels[0]);
            targets.at(task.batch_row, 2) = static_cast<float>(eeg_labels[1]);
            targets.at(task.batch_row, 3) = static_cast<float>(eeg_labels[2]);
            targets.at(task.batch_row, 4) = static_cast<float>(task.eeg_index_label);
        }

        op = op_end;
    }
}

} // namespace

void SynchronizedBatchAssembler::assembleGrouped(                                         //
    const std::vector<std::vector<RowRequest>>& grouped,                                  //
    const std::vector<SubjectFiles>& subjects,                                            //
    const std::vector<std::unique_ptr<nn::dataLoaders::AudioMatSession>>& audio_sessions, //
    const std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>>& eeg_sessions,     //
    nn::Tensor& inputs, nn::Tensor& targets                                               //
)
{
    for (size_t subject_index = 0; subject_index < grouped.size(); ++subject_index)
    {
        const auto& group = grouped[subject_index];
        if (group.empty()) continue;

        // Make a local copy of requests and sort by local_audio_row so
        // we can detect contiguous runs to bulk-read audio rows.
        vector<RowRequest> requests = group;
        sort(requests.begin(),
             requests.end(),
             [](const RowRequest& a, const RowRequest& b)
             { return a.local_audio_row < b.local_audio_row; });

        size_t audio_run_pos = 0;
        while (audio_run_pos < requests.size())
        {
            // Find contiguous run of local_audio_row values in the sorted requests
            // so it can be bulk-read from disk in a single call. Note that the audio
            // run may contain non-contiguous EEG rows, which will be handled later.
            size_t audio_run_end = audio_run_pos + 1;
            while (audio_run_end < requests.size() &&
                   requests[audio_run_end].local_audio_row ==
                       requests[audio_run_end - 1].local_audio_row + 1)
            {
                ++audio_run_end;
            }

            // Load the contiguous audio rows for the audio run and build tasks for each
            // row based on the loaded audio data. Tasks will contain the information
            // needed to load and copy the matching EEG rows in the next step.
            const size_t audio_run_count = audio_run_end - audio_run_pos;
            const size_t audio_run_start = requests[audio_run_pos].local_audio_row;

            // Note: `readRowsFlat` returns a struct with flat vectors for stimuli and
            // eeg index labels, which is more efficient to build tasks from. The concrete type of
            // `audio_rows_flat` is hidden behind a template to avoid including the `AudioRowsFlat`
            // type in this translation unit, since it's only used in the implementation of
            // `buildTasksFromAudioRun` and `processEegBlocksForTasks`.
            const auto audio_rows_flat = audio_sessions.at(subject_index)
                                             ->readRowsFlat(      //
                                                 audio_run_start, //
                                                 audio_run_count  //
                                             );

            // Build tasks for the audio run and process the matching EEG blocks to fill `inputs`
            // and `targets`. This step will iterate over the tasks sorted by EEG row index, so
            // contiguous EEG rows will be bulk-read from disk and copied to the right place in
            // the batch tensors.
            vector<BatchTask> tasks;
            tasks.reserve(audio_run_count);
            const SubjectFiles& subject = subjects.at(subject_index);

            // Load the audio rows for the run and build tasks for each row based on the loaded
            // audio data. Tasks will contain the information needed to load and copy the matching
            // EEG rows in the next step.
            buildTasksFromAudioRun( //
                audio_rows_flat,    //
                requests,           //
                audio_run_pos,      //
                audio_run_count,    //
                subject,            //
                tasks               //
            );

            // Load and copy the matching EEG rows for the tasks: since tasks are sorted by EEG
            // row index, contiguous EEG rows will be bulk-read from disk and copied to the
            // right place in the batch tensors.
            processEegBlocksForTasks( //
                subject_index,        //
                subject,              //
                eeg_sessions,         //
                audio_rows_flat,      //
                tasks,                //
                inputs,               //
                targets               //
            );

            // Move to the next audio run.
            audio_run_pos = audio_run_end;
        }
    }
}