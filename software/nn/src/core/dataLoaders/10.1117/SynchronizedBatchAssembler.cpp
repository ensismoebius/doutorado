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

} // namespace

void SynchronizedBatchAssembler::assembleGrouped(                               //
    const vector<vector<RowRequest>>& grouped,                                  //
    const vector<SubjectFiles>& subjects,                                       //
    const vector<unique_ptr<nn::dataLoaders::AudioMatSession>>& audio_sessions, //
    const vector<unique_ptr<nn::dataLoaders::EEGMatSession>>& eeg_sessions,     //
    nn::Tensor& inputs,                                                         //
    nn::Tensor& targets                                                         //
)
{
    for (size_t subject_index = 0; subject_index < grouped.size(); ++subject_index)
    {
        if (grouped[subject_index].empty())
        {
            continue;
        }

        auto requests = grouped[subject_index];

        // Sort requests by `local_audio_row` so we can read contiguous runs
        // of audio rows in a single bulk I/O call (`readRowsFlat`). This
        // improves locality and reduces the number of I/O/syscalls when
        // assembling the batch.
        std::sort(requests.begin(),
                  requests.end(),
                  [](const RowRequest& a, const RowRequest& b)
                  { return a.local_audio_row < b.local_audio_row; });

        std::size_t pos = 0;
        while (pos < requests.size())
        {
            // Find the end of the current run of contiguous `local_audio_row`
            // values (e.g. rows like N, N+1, N+2...). By detecting these runs
            // we can call `readRowsFlat(run_start_row, run_count)` once for
            // the whole block instead of per-row, reducing I/O overhead and
            // improving cache/locality when assembling the batch.
            std::size_t run_end = pos + 1;
            while (run_end < requests.size() &&
                   requests[run_end].local_audio_row == requests[run_end - 1].local_audio_row + 1)
            {
                ++run_end;
            }

            // Starting audio row index for this contiguous run.
            const size_t run_start_row = requests[pos].local_audio_row;
            // Number of consecutive audio rows included in the contiguous
            // run (used as the `count` argument for the bulk read).
            const size_t run_count = run_end - pos;
            // Bulk-read audio rows [run_start_row, run_start_row + run_count)
            // into `audio_rows_flat`. The returned object contains per-row
            // `samples`, `stimuli` and `eegIndices` arranged in row order so
            // we can build `tasks` without performing per-row I/O.
            const auto audio_rows_flat =
                audio_sessions.at(subject_index)->readRowsFlat(run_start_row, run_count);

            vector<BatchTask> tasks;
            tasks.reserve(run_count);
            const SubjectFiles& subject = subjects.at(subject_index);

            // Construct one `BatchTask` per audio row in the contiguous run.
            // `batch_row`: which row in the final `inputs`/`targets` this
            // sample should be written to.
            // `audio_index`: index into `audio_rows_flat` for this sample; a
            // small integer used to compute offsets when copying samples
            // out of the flat audio buffer without additional seeks.
            // `audio_stimulus`: stimulus label read from the audio row; used
            // later to sanity-check that the matched EEG row has the same
            // stimulus label.
            // `eeg_index_label`: raw EEG index value stored alongside the
            // audio row in the source file (file-specific encoding preserved
            // here so the original reference can be reconstructed if needed).
            // `eeg_row`: resolved EEG row index (result of
            // `resolveEegRowIndex`) used to group and bulk-read EEG rows.
            for (size_t audio_index = 0; audio_index < run_count; ++audio_index)
            {
                // Snapshot the request for this audio row (contains
                // `batch_row` where the final sample should be placed).
                const RowRequest& req = requests[pos + audio_index];
                // Stimulus label read from the contiguous audio read buffer
                // for this audio_index; used to validate alignment with EEG.
                const int audio_stimulus = audio_rows_flat.stimuli[audio_index];
                // Raw EEG index value stored alongside the audio row in the
                // source file; this is a file-specific encoding.
                const int eeg_index_label = audio_rows_flat.eegIndices[audio_index];
                // Resolve the raw EEG index label to a concrete EEG row
                // index within `subject.eeg_rows` (used for grouping reads).
                const size_t eeg_row = resolveEegRowIndex(eeg_index_label, subject.eeg_rows);

                // Create a BatchTask mapping: (batch_row, audio_index,
                // audio_stimulus, eeg_index_label, eeg_row). `audio_index`
                // is an offset into `audio_rows_flat` for later copying.
                tasks.push_back(BatchTask{
                    req.batch_row, audio_index, audio_stimulus, eeg_index_label, eeg_row});
            }

            vector<size_t> order(tasks.size());
            iota(order.begin(), order.end(), 0);

            // Reorder `tasks` by `eeg_row` and operate on the index permutation
            // (`order`) so we can read contiguous EEG rows in bulk via
            // `eeg_sessions.at(...)->readRowsFlat(...)`. After reading the
            // contiguous EEG block we place each audio+EEG pair into the
            // correct `batch_row` slot in `inputs` (preserving synchronization).
            std::sort(order.begin(),
                      order.end(),
                      [&tasks](std::size_t a, std::size_t b)
                      { return tasks[a].eeg_row < tasks[b].eeg_row; });

            std::size_t op = 0;
            while (op < order.size())
            {
                // Find a contiguous run of `eeg_row` values in `tasks` (after
                // sorting by `eeg_row`) so we can read that EEG block with a
                // single bulk call to `readRowsFlat(eeg_run_start, eeg_run_count)`.
                // This reduces I/O/syscall overhead and preserves ordering for
                // efficient placement into `inputs` using the `order` permutation.
                std::size_t op_end = op + 1;
                while (op_end < order.size() &&
                       tasks[order[op_end]].eeg_row == tasks[order[op_end - 1]].eeg_row + 1)
                {
                    ++op_end;
                }

                // Starting EEG row index for the contiguous block to read.
                const std::size_t eeg_run_start = tasks[order[op]].eeg_row;
                // Number of consecutive EEG rows included in this contiguous
                // block (used as the `count` argument for the bulk read).
                const std::size_t eeg_run_count = op_end - op;
                // Bulk-read EEG rows [eeg_run_start, eeg_run_start +
                // eeg_run_count) into `eeg_rows_flat`. The returned structure
                // contains the per-row `signals` and `labels` in row order.
                const auto eeg_rows_flat =
                    eeg_sessions.at(subject_index)->readRowsFlat(eeg_run_start, eeg_run_count);

                // One-row tensor used as a destination for copying a single
                // EEG row out of the bulk-read EEG buffer.
                nn::Tensor eegRow(1, EEG_FEATURES);
                // One-row tensor used as a destination for copying the
                // matched audio sample slice out of the bulk-read audio
                // buffer.
                nn::Tensor audioRow(1, ImaginedSpeechSchema_10_1117.audioSamples());
                // Raw pointer to `eegRow` storage for the tight inner-loop
                // copy (avoids accessor overhead inside the per-sample loop).
                float* eegDst = eegRow.mutable_data_ptr();
                // Raw pointer to `audioRow` storage for the tight inner-loop
                // copy (avoids accessor overhead inside the per-sample loop).
                float* audioDst = audioRow.mutable_data_ptr();
                // Number of audio columns per row; used to compute offsets
                // into the flat `audio_rows_flat.samples` buffer.
                const size_t audioCols = ImaginedSpeechSchema_10_1117.audioSamples();

                // Iterate each row in the just-read EEG block, copy the
                // corresponding EEG features and matched audio slice into
                // the correct `batch_row` in `inputs`, and populate
                // `targets` metadata for downstream consumers.
                for (size_t j = 0; j < eeg_run_count; ++j)
                {
                    // Map the sorted permutation back to the original task
                    // so we know target `batch_row` and `audio_index`.
                    const BatchTask& task = tasks[order[op + j]];
                    // Offset (in samples) into the flat audio buffer for this
                    // task's audio slice: `audio_index * audioSamples`.
                    const size_t audioOffset =
                        task.audio_index * ImaginedSpeechSchema_10_1117.audioSamples();
                    // Offset (in features) into the flat EEG signals buffer
                    // for the j-th row within the currently read EEG block.
                    const size_t eegOffset = j * EEG_FEATURES;
                    // Reference to the labels array for the current EEG row
                    // (contains metadata used below for validation/targets).
                    const auto& eeg_labels = eeg_rows_flat.labels[j];

                    // Extract the stimulus label from EEG labels (index 1)
                    // and validate it matches the audio stimulus for this
                    // task to ensure proper synchronization.
                    const int stimulus_label = eeg_labels[1];
                    if (task.audio_stimulus != stimulus_label)
                    {
                        throw runtime_error("Stimulus mismatch between audio and EEG in collate");
                    }

                    // Copy EEG feature vector from the flat buffer into the
                    // one-row temporary (`eegRow`) via the raw pointer.
                    for (size_t c = 0; c < EEG_FEATURES; ++c)
                    {
                        eegDst[c] = eeg_rows_flat.signals[eegOffset + c];
                    }

                    // Copy audio samples for this task from the flat audio
                    // buffer into the one-row temporary (`audioRow`).
                    for (size_t c = 0; c < audioCols; ++c)
                    {
                        audioDst[c] = audio_rows_flat.samples[audioOffset + c];
                    }

                    // Place the populated EEG one-row tensor into `inputs`
                    // at column 0 for the target `batch_row`.
                    inputs.setBlock(task.batch_row, 0, eegRow);

                    // Place the populated audio one-row tensor into `inputs`
                    // starting at column `EEG_FEATURES` so EEG+audio are
                    // concatenated in the row.
                    inputs.setBlock(task.batch_row, EEG_FEATURES, audioRow);

                    // Populate `targets` metadata for this batch row:
                    // [subject_id, eeg_labels[0], eeg_labels[1], eeg_labels[2], original eeg index
                    // label]
                    targets.at(task.batch_row, 0) = static_cast<float>(subject.subject_id);
                    targets.at(task.batch_row, 1) = static_cast<float>(eeg_labels[0]);
                    targets.at(task.batch_row, 2) = static_cast<float>(eeg_labels[1]);
                    targets.at(task.batch_row, 3) = static_cast<float>(eeg_labels[2]);
                    targets.at(task.batch_row, 4) = static_cast<float>(task.eeg_index_label);
                }
                // Advance to the next permutation block.
                op = op_end;
            }

            pos = run_end;
        }
    }
}