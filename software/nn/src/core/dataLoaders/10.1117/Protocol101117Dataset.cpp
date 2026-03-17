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
constexpr size_t STACKED_CONCAT_ROWS =
    nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels + 1U;
constexpr size_t STACKED_CONCAT_COLS = AUDIO_FEATURES;
constexpr size_t STACKED_CONCAT_FEATURES = STACKED_CONCAT_ROWS * STACKED_CONCAT_COLS;

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

Protocol101117Dataset::Protocol101117Dataset(
    std::vector<SubjectFiles> subjects, Protocol101117InputMode input_mode)
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

[[nodiscard]] auto Protocol101117Dataset::get_sample(size_t idx,
    std::optional<Protocol101117InputMode> input_mode_override) const -> Protocol101117Sample
{
    if (idx >= size())
    {
        throw std::out_of_range("Dataset index out of range: " + std::to_string(idx));
    }

    const auto [subject_index, audio_row] =
        locateSubjectAndAudioRow(prefix_audio_row_offsets_, idx);

    ensureSubjectMatSessionsInitialized(subject_index);

    const SubjectFiles& subject = subjects_.at(subject_index);
    const auto& audio_session = *audio_sessions_.at(subject_index);
    const auto& eeg_session = *eeg_sessions_.at(subject_index);

    const RawSynchronizedSample raw =
        readSynchronizedSampleFromSessions(subject, audio_session, eeg_session, audio_row);

    Protocol101117Sample sample;
    sample.audio = buildLegacyAudioRow(raw.audio_tensor);
    sample.eeg = buildLegacyEegRow(raw.eeg_tensor);

    sample.targets =
        buildTargetTensor(raw.subject->subject_id, raw.eeg_labels, raw.eeg_index_label);
    sample.input_mode = input_mode_override.value_or(input_mode_);

    if (sample.input_mode == Protocol101117InputMode::Concatenated)
    {
        sample.inputs = buildStackedInputTensorFromRaw(raw.eeg_tensor, raw.audio_tensor);
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
            throw std::out_of_range(
                "Dataset index out of range in collate: " + std::to_string(idx));
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
        ensureSubjectMatSessionsInitialized(subject_index);
    }

    SynchronizedBatchAssembler::assembleGrouped(
        grouped, subjects_, audio_sessions_, eeg_sessions_, assembled_inputs, targets);

    if (input_mode_ == Protocol101117InputMode::Concatenated)
    {
        inputs = nn::Tensor(indices.size(), STACKED_CONCAT_FEATURES);

        const size_t eeg_channels = nn::dataLoaders::ImaginedSpeechSchema_10_1117.eeg_channels;
        const size_t eeg_samples_per_channel =
            nn::dataLoaders::ImaginedSpeechSchema_10_1117.eegSamplesPerChannel();

        nn::Tensor eeg_matrix(eeg_channels, eeg_samples_per_channel);
        nn::Tensor audio_column(AUDIO_FEATURES, 1);

        for (size_t row = 0; row < indices.size(); ++row)
        {
            for (size_t ch = 0; ch < eeg_channels; ++ch)
            {
                const size_t eeg_row_offset = ch * eeg_samples_per_channel;
                for (size_t sample_idx = 0; sample_idx < eeg_samples_per_channel; ++sample_idx)
                {
                    eeg_matrix.at(ch, sample_idx) =
                        assembled_inputs.at(row, eeg_row_offset + sample_idx);
                }
            }

            for (size_t col = 0; col < AUDIO_FEATURES; ++col)
            {
                audio_column.at(col, 0) = assembled_inputs.at(row, EEG_FEATURES + col);
            }

            const nn::Tensor stacked = buildStackedInputTensorFromRaw(eeg_matrix, audio_column);
            for (size_t stacked_row = 0; stacked_row < STACKED_CONCAT_ROWS; ++stacked_row)
            {
                for (size_t col = 0; col < STACKED_CONCAT_COLS; ++col)
                {
                    const size_t flat_index = stacked_row * STACKED_CONCAT_COLS + col;
                    inputs.at(row, flat_index) = stacked.at(stacked_row, col);
                }
            }
        }
    }
    else if (input_mode_ == Protocol101117InputMode::EegOnly)
    {
        inputs = extractEegFromAssembledRows(assembled_inputs);
    }
    else
    {
        inputs = extractAudioFromAssembledRows(assembled_inputs);
    }

    return {.inputs = std::move(inputs), .targets = std::move(targets)};
}

[[nodiscard]] auto Protocol101117Dataset::subjects() const -> const std::vector<SubjectFiles>&
{
    return subjects_;
}

/**
 * Ensure the MAT session objects for the given subject are opened and
 * initialized (lazy initialization).
 *
 * This function creates `AudioMatSession` and/or `EEGMatSession` instances
 * for the subject at `subject_index` if they are not already present in the
 * mutable session caches. It is intended to be called by reader paths
 * (e.g. `get_sample` and `collate`) before accessing `audio_sessions_`/
 * `eeg_sessions_`.
 *
 * Note: this performs I/O (opens .mat files) and may throw on file errors.
 * It is NOT thread-safe; callers must synchronize externally if used from
 * multiple threads.
 */
void Protocol101117Dataset::ensureSubjectMatSessionsInitialized(size_t subject_index) const
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