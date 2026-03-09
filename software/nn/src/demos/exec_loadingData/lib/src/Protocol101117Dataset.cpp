#include "../include/Protocol101117Dataset.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "nn/dataLoaders/10.1117/SchemaIndexing.hpp"

using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
using nn::dataLoaders::schema101117::eegFeatureColumns;
using nn::dataLoaders::schema101117::multimodalInputFeatureColumns;
using nn::dataLoaders::schema101117::resolveEegRowIndex;
using std::size_t;

namespace
{

constexpr size_t EEG_FEATURES = eegFeatureColumns();
constexpr size_t INPUT_FEATURES = multimodalInputFeatureColumns();

auto makeInputTensor(const nn::Tensor& eeg, const nn::Tensor& audio) -> nn::Tensor
{
    if (eeg.rows() != ImaginedSpeechSchema_10_1117.eeg_channels ||
        eeg.cols() != ImaginedSpeechSchema_10_1117.eegSamplesPerChannel())
    {
        throw std::runtime_error("Unexpected EEG shape. Expected [6x4096].");
    }

    if (audio.rows() != ImaginedSpeechSchema_10_1117.audioSamples() || audio.cols() != 1)
    {
        throw std::runtime_error("Unexpected Audio shape. Expected [176400x1].");
    }

    nn::Tensor input(1, INPUT_FEATURES);

    size_t col = 0;
    for (size_t ch = 0; ch < ImaginedSpeechSchema_10_1117.eeg_channels; ++ch)
    {
        for (size_t s = 0; s < ImaginedSpeechSchema_10_1117.eegSamplesPerChannel(); ++s)
        {
            input.at(0, col++) = eeg.at(ch, s);
        }
    }

    for (size_t i = 0; i < ImaginedSpeechSchema_10_1117.audioSamples(); ++i)
    {
        input.at(0, col++) = audio.at(i, 0);
    }

    return input;
}

auto makeTargetTensor(int subject_id, const std::array<int, 3>& eeg_labels, int eeg_index_label)
    -> nn::Tensor
{
    nn::Tensor target(1, 5);
    target.at(0, 0) = static_cast<float>(subject_id);
    target.at(0, 1) = static_cast<float>(eeg_labels[0]);
    target.at(0, 2) = static_cast<float>(eeg_labels[1]);
    target.at(0, 3) = static_cast<float>(eeg_labels[2]);
    target.at(0, 4) = static_cast<float>(eeg_index_label);
    return target;
}

} // namespace

Protocol101117Dataset::Protocol101117Dataset(std::vector<SubjectFiles> subjects)
    : subjects_(std::move(subjects))
{
    prefix_audio_row_offsets_.reserve(subjects_.size() + 1);
    prefix_audio_row_offsets_.emplace_back(0);

    for (const auto& subject : subjects_)
    {
        const size_t next = prefix_audio_row_offsets_.back() + subject.audio_rows;
        prefix_audio_row_offsets_.emplace_back(next);
    }

    audio_sessions_.resize(subjects_.size());
    eeg_sessions_.resize(subjects_.size());
}

[[nodiscard]] auto Protocol101117Dataset::size() const -> size_t
{
    return prefix_audio_row_offsets_.empty() ? 0 : prefix_audio_row_offsets_.back();
}

[[nodiscard]] auto Protocol101117Dataset::get_item(size_t idx) const -> Batch
{
    if (idx >= size())
    {
        throw std::out_of_range("Dataset index out of range: " + std::to_string(idx));
    }

    const auto upper_it = std::upper_bound( //
        prefix_audio_row_offsets_.begin(),  //
        prefix_audio_row_offsets_.end(),    //
        idx                                 //
    );

    const size_t subject_index = static_cast<size_t>( //
        std::distance(                                //
            prefix_audio_row_offsets_.begin(),        //
            upper_it) -
        1 //
    );

    const size_t audio_row = idx - prefix_audio_row_offsets_[subject_index];
    return loadSampleByLocalIndex(subject_index, audio_row);
}

[[nodiscard]] auto Protocol101117Dataset::collate(const std::vector<std::size_t>& indices) const
    -> Batch
{
    if (indices.empty())
    {
        return Dataset::collate(indices);
    }

    nn::Tensor inputs(indices.size(), INPUT_FEATURES);
    nn::Tensor targets(indices.size(), 5);

    struct RowRequest
    {
        size_t batch_row;
        size_t local_audio_row;
    };

    std::vector<std::vector<RowRequest>> grouped(subjects_.size());
    for (size_t row = 0; row < indices.size(); ++row)
    {
        const size_t idx = indices[row];
        if (idx >= size())
        {
            throw std::out_of_range("Dataset index out of range in collate: " +
                                    std::to_string(idx));
        }

        const auto upper_it = std::upper_bound(
            prefix_audio_row_offsets_.begin(), prefix_audio_row_offsets_.end(), idx);
        const size_t subject_index =
            static_cast<size_t>(std::distance(prefix_audio_row_offsets_.begin(), upper_it) - 1);
        grouped[subject_index].push_back(
            RowRequest{row, idx - prefix_audio_row_offsets_[subject_index]});
    }

    for (size_t subject_index = 0; subject_index < grouped.size(); ++subject_index)
    {
        if (grouped[subject_index].empty())
        {
            continue;
        }

        ensureSessions(subject_index);
        auto& requests = grouped[subject_index];
        std::sort(requests.begin(),
                  requests.end(),
                  [](const RowRequest& a, const RowRequest& b)
                  { return a.local_audio_row < b.local_audio_row; });

        size_t pos = 0;
        while (pos < requests.size())
        {
            size_t run_end = pos + 1;
            while (run_end < requests.size() &&
                   requests[run_end].local_audio_row == requests[run_end - 1].local_audio_row + 1)
            {
                ++run_end;
            }

            const size_t run_start_row = requests[pos].local_audio_row;
            const size_t run_count = run_end - pos;
            const auto audio_rows_flat =
                audio_sessions_.at(subject_index)->readRowsFlat(run_start_row, run_count);

            struct BatchTask
            {
                size_t batch_row;
                size_t audio_index;
                int audio_stimulus;
                int eeg_index_label;
                size_t eeg_row;
            };

            std::vector<BatchTask> tasks;
            tasks.reserve(run_count);
            const SubjectFiles& subject = subjects_.at(subject_index);
            for (size_t k = 0; k < run_count; ++k)
            {
                const RowRequest& req = requests[pos + k];
                const int audio_stimulus = audio_rows_flat.stimuli[k];
                const int eeg_index_label = audio_rows_flat.eegIndices[k];
                const size_t eeg_row = resolveEegRowIndex(eeg_index_label, subject.eeg_rows);
                tasks.push_back(
                    BatchTask{req.batch_row, k, audio_stimulus, eeg_index_label, eeg_row});
            }

            std::vector<size_t> order(tasks.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(),
                      order.end(),
                      [&tasks](size_t a, size_t b) { return tasks[a].eeg_row < tasks[b].eeg_row; });

            size_t op = 0;
            while (op < order.size())
            {
                size_t op_end = op + 1;
                while (op_end < order.size() &&
                       tasks[order[op_end]].eeg_row == tasks[order[op_end - 1]].eeg_row + 1)
                {
                    ++op_end;
                }

                const size_t eeg_run_start = tasks[order[op]].eeg_row;
                const size_t eeg_run_count = op_end - op;
                const auto eeg_rows_flat =
                    eeg_sessions_.at(subject_index)->readRowsFlat(eeg_run_start, eeg_run_count);

                nn::Tensor eegRow(1, EEG_FEATURES);
                nn::Tensor audioRow(1, ImaginedSpeechSchema_10_1117.audioSamples());
                float* eegDst = eegRow.mutable_data_ptr();
                float* audioDst = audioRow.mutable_data_ptr();
                const size_t audioCols = ImaginedSpeechSchema_10_1117.audioSamples();

                for (size_t j = 0; j < eeg_run_count; ++j)
                {
                    const BatchTask& task = tasks[order[op + j]];
                    const size_t audioOffset =
                        task.audio_index * ImaginedSpeechSchema_10_1117.audioSamples();
                    const size_t eegOffset = j * EEG_FEATURES;
                    const auto& eeg_labels = eeg_rows_flat.labels[j];

                    const int stimulus_label = eeg_labels[1];
                    if (task.audio_stimulus != stimulus_label)
                    {
                        throw std::runtime_error(
                            "Stimulus mismatch between audio and EEG in collate");
                    }

                    for (size_t c = 0; c < EEG_FEATURES; ++c)
                    {
                        eegDst[c] = eeg_rows_flat.signals[eegOffset + c];
                    }
                    inputs.setBlock(task.batch_row, 0, eegRow);

                    for (size_t c = 0; c < audioCols; ++c)
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

    return {.inputs = std::move(inputs), .targets = std::move(targets)};
}

[[nodiscard]] auto Protocol101117Dataset::subjects() const -> const std::vector<SubjectFiles>&
{
    return subjects_;
}

auto Protocol101117Dataset::loadSampleByLocalIndex(size_t subject_index, size_t audio_row) const
    -> Batch
{
    const SubjectFiles& subject = subjects_.at(subject_index);
    ensureSessions(subject_index);

    const auto [audio_tensor, audio_stimulus, eeg_index_label] =
        audio_sessions_.at(subject_index)->readRow(audio_row);

    const size_t eeg_row = resolveEegRowIndex(eeg_index_label, subject.eeg_rows);
    const auto [    //
        eeg_tensor, //
        eeg_labels  //
    ] = eeg_sessions_.at(subject_index)->readRow(eeg_row);

    const int stimulus_label = eeg_labels[1];
    if (audio_stimulus != stimulus_label)
    {
        throw std::runtime_error("Stimulus mismatch between audio and EEG for subject " +
                                 subject.subject_name + " at audio row " +
                                 std::to_string(audio_row));
    }

    nn::Tensor input = makeInputTensor(eeg_tensor, audio_tensor);
    nn::Tensor target = makeTargetTensor(subject.subject_id, eeg_labels, eeg_index_label);
    return {.inputs = std::move(input), .targets = std::move(target)};
}

void Protocol101117Dataset::ensureSessions(size_t subject_index) const
{
    if (audio_sessions_.at(subject_index) && eeg_sessions_.at(subject_index))
    {
        return;
    }

    const SubjectFiles& subject = subjects_.at(subject_index);
    if (!audio_sessions_.at(subject_index))
    {
        audio_sessions_.at(subject_index) =
            std::make_unique<nn::dataLoaders::AudioMatSession>(subject.audio_mat_path);
    }
    if (!eeg_sessions_.at(subject_index))
    {
        eeg_sessions_.at(subject_index) =
            std::make_unique<nn::dataLoaders::EEGMatSession>(subject.eeg_mat_path);
    }
}
