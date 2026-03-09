#include "../include/Protocol101117Dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../include/SamplePacking.hpp"
#include "../include/SynchronizedBatchAssembler.hpp"
#include "nn/dataLoaders/10.1117/METADATA.hpp"
#include "nn/dataLoaders/10.1117/SchemaIndexing.hpp"

using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
using nn::dataLoaders::schema101117::multimodalInputFeatureColumns;
using nn::dataLoaders::schema101117::resolveEegRowIndex;
using std::size_t;

namespace
{

constexpr size_t INPUT_FEATURES = multimodalInputFeatureColumns();

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
    }

    SynchronizedBatchAssembler::assembleGrouped(
        grouped, subjects_, audio_sessions_, eeg_sessions_, inputs, targets);

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

    nn::Tensor input = buildInputTensor(eeg_tensor, audio_tensor);
    nn::Tensor target = buildTargetTensor(subject.subject_id, eeg_labels, eeg_index_label);
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
