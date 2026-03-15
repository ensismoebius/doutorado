#include "nn/dataLoaders/10.1117/Protocol101117Dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nn/dataLoaders/10.1117/SamplePacking.hpp"
#include "nn/dataLoaders/10.1117/SchemaIndexing.hpp"
#include "nn/dataLoaders/10.1117/SynchronizedBatchAssembler.hpp"

using nn::dataLoaders::schema101117::eegFeatureColumns;
using nn::dataLoaders::schema101117::multimodalInputFeatureColumns;
using nn::dataLoaders::schema101117::resolveEegRowIndex;
using std::size_t;

namespace
{

constexpr size_t INPUT_FEATURES = multimodalInputFeatureColumns();
constexpr size_t EEG_FEATURES = eegFeatureColumns();

} // namespace

Protocol101117Dataset::Protocol101117Dataset(std::vector<SubjectFiles> subjects,
                                             bool concatenate_modalities)
    : subjects_(std::move(subjects)), concatenate_modalities_(concatenate_modalities)
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

void Protocol101117Dataset::set_concatenate_modalities(bool concatenate_modalities)
{
    concatenate_modalities_ = concatenate_modalities;
}

[[nodiscard]] auto Protocol101117Dataset::concatenate_modalities() const -> bool
{
    return concatenate_modalities_;
}

[[nodiscard]] auto Protocol101117Dataset::get_sample(size_t idx,
                                                     std::optional<bool> concatenate_override) const
    -> Protocol101117Sample
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

    Protocol101117Sample sample;
    sample.audio = nn::Tensor(1, audio_tensor.rows());
    for (size_t i = 0; i < static_cast<size_t>(audio_tensor.rows()); ++i)
    {
        sample.audio.at(0, i) = audio_tensor.at(i, 0);
    }

    sample.eeg = nn::Tensor(1, eeg_tensor.rows() * eeg_tensor.cols());
    size_t eeg_col = 0;
    for (size_t r = 0; r < static_cast<size_t>(eeg_tensor.rows()); ++r)
    {
        for (size_t c = 0; c < static_cast<size_t>(eeg_tensor.cols()); ++c)
        {
            sample.eeg.at(0, eeg_col++) = eeg_tensor.at(r, c);
        }
    }

    sample.targets = buildTargetTensor(subject.subject_id, eeg_labels, eeg_index_label);
    sample.concatenated = concatenate_override.value_or(concatenate_modalities_);

    if (sample.concatenated)
    {
        sample.inputs = buildInputTensor(eeg_tensor, audio_tensor);
    }
    else
    {
        // In separated mode, keep EEG as primary input and expose audio in `sample.audio`.
        sample.inputs = sample.eeg;
    }

    return sample;
}

[[nodiscard]] auto Protocol101117Dataset::size() const -> size_t
{
    return prefix_audio_row_offsets_.empty() ? 0 : prefix_audio_row_offsets_.back();
}

[[nodiscard]] auto Protocol101117Dataset::get_item(size_t idx) const -> Batch
{
    auto sample = get_sample(idx);
    return {.inputs = std::move(sample.inputs), .targets = std::move(sample.targets)};
}

[[nodiscard]] auto Protocol101117Dataset::collate(const std::vector<std::size_t>& indices) const
    -> Batch
{
    if (indices.empty())
    {
        return Dataset::collate(indices);
    }

    // The assembler always builds concatenated EEG+audio rows. If this dataset
    // is configured for separated output, we slice only EEG columns afterwards
    // without re-reading files.
    nn::Tensor assembled_inputs(indices.size(), INPUT_FEATURES);
    nn::Tensor inputs;
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
        grouped, subjects_, audio_sessions_, eeg_sessions_, assembled_inputs, targets);

    if (concatenate_modalities_)
    {
        inputs = std::move(assembled_inputs);
    }
    else
    {
        inputs = nn::Tensor(indices.size(), EEG_FEATURES);
        for (size_t row = 0; row < indices.size(); ++row)
        {
            for (size_t col = 0; col < EEG_FEATURES; ++col)
            {
                inputs.at(row, col) = assembled_inputs.at(row, col);
            }
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