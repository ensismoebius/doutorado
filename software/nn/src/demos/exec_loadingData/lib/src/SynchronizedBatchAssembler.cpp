#include "../include/SynchronizedBatchAssembler.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "nn/dataLoaders/10.1117/SchemaIndexing.hpp"

using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
using nn::dataLoaders::schema101117::eegFeatureColumns;
using nn::dataLoaders::schema101117::resolveEegRowIndex;

namespace
{

constexpr std::size_t EEG_FEATURES = eegFeatureColumns();

struct BatchTask
{
    std::size_t batch_row;
    std::size_t audio_index;
    int audio_stimulus;
    int eeg_index_label;
    std::size_t eeg_row;
};

} // namespace

void SynchronizedBatchAssembler::assembleGrouped(
    const std::vector<std::vector<RowRequest>>& grouped, const std::vector<SubjectFiles>& subjects,
    const std::vector<std::unique_ptr<nn::dataLoaders::AudioMatSession>>& audio_sessions,
    const std::vector<std::unique_ptr<nn::dataLoaders::EEGMatSession>>& eeg_sessions,
    nn::Tensor& inputs, nn::Tensor& targets)
{
    for (std::size_t subject_index = 0; subject_index < grouped.size(); ++subject_index)
    {
        if (grouped[subject_index].empty())
        {
            continue;
        }

        auto requests = grouped[subject_index];
        std::sort(requests.begin(),
                  requests.end(),
                  [](const RowRequest& a, const RowRequest& b)
                  { return a.local_audio_row < b.local_audio_row; });

        std::size_t pos = 0;
        while (pos < requests.size())
        {
            std::size_t run_end = pos + 1;
            while (run_end < requests.size() &&
                   requests[run_end].local_audio_row == requests[run_end - 1].local_audio_row + 1)
            {
                ++run_end;
            }

            const std::size_t run_start_row = requests[pos].local_audio_row;
            const std::size_t run_count = run_end - pos;
            const auto audio_rows_flat =
                audio_sessions.at(subject_index)->readRowsFlat(run_start_row, run_count);

            std::vector<BatchTask> tasks;
            tasks.reserve(run_count);
            const SubjectFiles& subject = subjects.at(subject_index);
            for (std::size_t k = 0; k < run_count; ++k)
            {
                const RowRequest& req = requests[pos + k];
                const int audio_stimulus = audio_rows_flat.stimuli[k];
                const int eeg_index_label = audio_rows_flat.eegIndices[k];
                const std::size_t eeg_row = resolveEegRowIndex(eeg_index_label, subject.eeg_rows);
                tasks.push_back(
                    BatchTask{req.batch_row, k, audio_stimulus, eeg_index_label, eeg_row});
            }

            std::vector<std::size_t> order(tasks.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(),
                      order.end(),
                      [&tasks](std::size_t a, std::size_t b)
                      { return tasks[a].eeg_row < tasks[b].eeg_row; });

            std::size_t op = 0;
            while (op < order.size())
            {
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

                nn::Tensor eegRow(1, EEG_FEATURES);
                nn::Tensor audioRow(1, ImaginedSpeechSchema_10_1117.audioSamples());
                float* eegDst = eegRow.mutable_data_ptr();
                float* audioDst = audioRow.mutable_data_ptr();
                const std::size_t audioCols = ImaginedSpeechSchema_10_1117.audioSamples();

                for (std::size_t j = 0; j < eeg_run_count; ++j)
                {
                    const BatchTask& task = tasks[order[op + j]];
                    const std::size_t audioOffset =
                        task.audio_index * ImaginedSpeechSchema_10_1117.audioSamples();
                    const std::size_t eegOffset = j * EEG_FEATURES;
                    const auto& eeg_labels = eeg_rows_flat.labels[j];

                    const int stimulus_label = eeg_labels[1];
                    if (task.audio_stimulus != stimulus_label)
                    {
                        throw std::runtime_error(
                            "Stimulus mismatch between audio and EEG in collate");
                    }

                    for (std::size_t c = 0; c < EEG_FEATURES; ++c)
                    {
                        eegDst[c] = eeg_rows_flat.signals[eegOffset + c];
                    }
                    inputs.setBlock(task.batch_row, 0, eegRow);

                    for (std::size_t c = 0; c < audioCols; ++c)
                    {
                        audioDst[c] = audio_rows_flat.samples[audioOffset + c];
                    }
                    inputs.setBlock(task.batch_row, EEG_FEATURES, audioRow);

                    targets.at(task.batch_row, 0) = static_cast<float>(subject.subject_id);
                    targets.at(task.batch_row, 1) = static_cast<float>(eeg_labels[0]);
                    targets.at(task.batch_row, 2) = static_cast<float>(eeg_labels[1]);
                    targets.at(task.batch_row, 3) = static_cast<float>(eeg_labels[2]);
                    targets.at(task.batch_row, 4) = static_cast<float>(task.eeg_index_label);
                }
                op = op_end;
            }

            pos = run_end;
        }
    }
}
