#include "nn/dataLoaders/10.1117/Protocol101117Dataset.hpp"

#include <algorithm>
#include <array>
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
constexpr size_t AUDIO_FEATURES = nn::dataLoaders::ImaginedSpeechSchema_10_1117.audioSamples();
constexpr size_t STACKED_CONCAT_FEATURES = EEG_FEATURES * 2U;

struct RawSynchronizedSample
{
    nn::Tensor audio_tensor;
    nn::Tensor eeg_tensor;
    std::array<int, 3> eeg_labels{};
    int eeg_index_label = 0;
    const SubjectFiles* subject = nullptr;
};

auto locateSubjectAndAudioRow(const std::vector<std::size_t>& prefix_audio_row_offsets,
                              std::size_t idx) -> std::pair<std::size_t, std::size_t>
{
    const auto upper_it = std::upper_bound( //
        prefix_audio_row_offsets.begin(),   //
        prefix_audio_row_offsets.end(),     //
        idx                                 //
    );

    const size_t subject_index = static_cast<size_t>(                 //
        std::distance(prefix_audio_row_offsets.begin(), upper_it) - 1 //
    );

    const size_t audio_row = idx - prefix_audio_row_offsets[subject_index];
    return {subject_index, audio_row};
}

void flattenAudioToRow(const nn::Tensor& audio_column_tensor, nn::Tensor& audio_row_tensor)
{
    for (size_t i = 0; i < static_cast<size_t>(audio_column_tensor.rows()); ++i)
    {
        audio_row_tensor.at(0, i) = audio_column_tensor.at(i, 0);
    }
}

void flattenEegToRow(const nn::Tensor& eeg_matrix_tensor, nn::Tensor& eeg_row_tensor)
{
    size_t eeg_col = 0;
    for (size_t r = 0; r < static_cast<size_t>(eeg_matrix_tensor.rows()); ++r)
    {
        for (size_t c = 0; c < static_cast<size_t>(eeg_matrix_tensor.cols()); ++c)
        {
            eeg_row_tensor.at(0, eeg_col++) = eeg_matrix_tensor.at(r, c);
        }
    }
}

void copyEegColumnsFromConcatenated(const nn::Tensor& concatenated_inputs,
                                    nn::Tensor& eeg_only_inputs)
{
    for (size_t row = 0; row < static_cast<size_t>(concatenated_inputs.rows()); ++row)
    {
        for (size_t col = 0; col < EEG_FEATURES; ++col)
        {
            eeg_only_inputs.at(row, col) = concatenated_inputs.at(row, col);
        }
    }
}

void copyAudioColumnsFromConcatenated(const nn::Tensor& concatenated_inputs,
                                      nn::Tensor& audio_only_inputs)
{
    for (size_t row = 0; row < static_cast<size_t>(concatenated_inputs.rows()); ++row)
    {
        for (size_t col = 0; col < AUDIO_FEATURES; ++col)
        {
            audio_only_inputs.at(row, col) = concatenated_inputs.at(row, EEG_FEATURES + col);
        }
    }
}

auto readSynchronizedSampleFromSessions(const SubjectFiles& subject,
                                        const nn::dataLoaders::AudioMatSession& audio_session,
                                        const nn::dataLoaders::EEGMatSession& eeg_session,
                                        size_t audio_row) -> RawSynchronizedSample
{
    const auto [audio_tensor, audio_stimulus, eeg_index_label] = audio_session.readRow(audio_row);

    const size_t eeg_row = resolveEegRowIndex(eeg_index_label, subject.eeg_rows);
    const auto [    //
        eeg_tensor, //
        eeg_labels  //
    ] = eeg_session.readRow(eeg_row);

    const int stimulus_label = eeg_labels[1];
    if (audio_stimulus != stimulus_label)
    {
        throw std::runtime_error("Stimulus mismatch between audio and EEG for subject " +
                                 subject.subject_name + " at audio row " +
                                 std::to_string(audio_row));
    }

    return {.audio_tensor = std::move(audio_tensor),
            .eeg_tensor = std::move(eeg_tensor),
            .eeg_labels = eeg_labels,
            .eeg_index_label = eeg_index_label,
            .subject = &subject};
}

} // namespace

Protocol101117Dataset::Protocol101117Dataset(std::vector<SubjectFiles> subjects,
                                             Protocol101117InputMode input_mode)
    : subjects_(std::move(subjects)), input_mode_(input_mode)
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

void Protocol101117Dataset::set_input_mode(Protocol101117InputMode input_mode)
{
    input_mode_ = input_mode;
}

[[nodiscard]] auto Protocol101117Dataset::input_mode() const -> Protocol101117InputMode
{
    return input_mode_;
}

[[nodiscard]] auto Protocol101117Dataset::get_sample(
    size_t idx, std::optional<Protocol101117InputMode> input_mode_override) const
    -> Protocol101117Sample
{
    if (idx >= size())
    {
        throw std::out_of_range("Dataset index out of range: " + std::to_string(idx));
    }

    const auto [subject_index, audio_row] =
        locateSubjectAndAudioRow(prefix_audio_row_offsets_, idx);
    ensureSessions(subject_index);

    const SubjectFiles& subject = subjects_.at(subject_index);
    const auto& audio_session = *audio_sessions_.at(subject_index);
    const auto& eeg_session = *eeg_sessions_.at(subject_index);
    const RawSynchronizedSample raw =
        readSynchronizedSampleFromSessions(subject, audio_session, eeg_session, audio_row);

    Protocol101117Sample sample;
    sample.audio = nn::Tensor(1, raw.audio_tensor.rows());
    flattenAudioToRow(raw.audio_tensor, sample.audio);

    sample.eeg = nn::Tensor(1, raw.eeg_tensor.rows() * raw.eeg_tensor.cols());
    flattenEegToRow(raw.eeg_tensor, sample.eeg);

    sample.targets =
        buildTargetTensor(raw.subject->subject_id, raw.eeg_labels, raw.eeg_index_label);
    sample.input_mode = input_mode_override.value_or(input_mode_);

    if (sample.input_mode == Protocol101117InputMode::Concatenated)
    {
        sample.inputs = buildInputTensorFromFlattenedRows(sample.eeg, sample.audio);
    }
    else if (sample.input_mode == Protocol101117InputMode::EegOnly)
    {
        // In separated mode, keep EEG as primary input and expose audio in `sample.audio`.
        sample.inputs = sample.eeg;
    }
    else
    {
        sample.inputs = sample.audio;
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

        const auto [subject_index, audio_row] =
            locateSubjectAndAudioRow(prefix_audio_row_offsets_, idx);
        grouped[subject_index].push_back(RowRequest{row, audio_row});
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

    if (input_mode_ == Protocol101117InputMode::Concatenated)
    {
        inputs = nn::Tensor(indices.size(), STACKED_CONCAT_FEATURES);

        nn::Tensor eeg_row(1, EEG_FEATURES);
        nn::Tensor audio_row(1, AUDIO_FEATURES);
        for (size_t row = 0; row < indices.size(); ++row)
        {
            for (size_t col = 0; col < EEG_FEATURES; ++col)
            {
                eeg_row.at(0, col) = assembled_inputs.at(row, col);
            }

            for (size_t col = 0; col < AUDIO_FEATURES; ++col)
            {
                audio_row.at(0, col) = assembled_inputs.at(row, EEG_FEATURES + col);
            }

            const nn::Tensor stacked = buildInputTensorFromFlattenedRows(eeg_row, audio_row);
            inputs.setBlock(row, 0, stacked);
        }
    }
    else if (input_mode_ == Protocol101117InputMode::EegOnly)
    {
        inputs = nn::Tensor(indices.size(), EEG_FEATURES);
        copyEegColumnsFromConcatenated(assembled_inputs, inputs);
    }
    else
    {
        inputs = nn::Tensor(indices.size(), AUDIO_FEATURES);
        copyAudioColumnsFromConcatenated(assembled_inputs, inputs);
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
    ensureSessions(subject_index);
    const SubjectFiles& subject = subjects_.at(subject_index);
    const auto& audio_session = *audio_sessions_.at(subject_index);
    const auto& eeg_session = *eeg_sessions_.at(subject_index);
    const RawSynchronizedSample raw =
        readSynchronizedSampleFromSessions(subject, audio_session, eeg_session, audio_row);

    nn::Tensor input = buildInputTensor(raw.eeg_tensor, raw.audio_tensor);
    nn::Tensor target =
        buildTargetTensor(raw.subject->subject_id, raw.eeg_labels, raw.eeg_index_label);
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