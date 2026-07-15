/**
 * @file AudioWindowDataset.cpp
 * @brief Implementation of AudioWindowDataset.
 *
 * Audio tensor from `AudioSession::readRow` has shape (audio_samples, 1),
 * i.e., a column vector with 176 400 rows for the 10.1117 dataset.
 *
 * Window extraction:
 *   out[0, t] = audio[window_start + t, 0]   for t in [0, window_size).
 */

#include "data_loaders/10.1117/datasets/windowed/AudioWindowDataset.hpp"

#include <stdexcept>
#include <string>

#include "data_loaders/10.1117/datasets/windowed/WindowingDatasetPrinter.hpp"
#include "data_loaders/10.1117/schema/Metadata.hpp"

using nn::dataLoaders::ImaginedSpeechSchema_10_1117;

namespace
{

// Schema-derived constant (compile-time).
constexpr int kAudioSamples =
    static_cast<int>(ImaginedSpeechSchema_10_1117.audioSamples()); // 176 400

/// Copy audio samples [window_start, window_start + window_size) from the
/// column-vector `audio` into row `out_row` of `out`.
void extract_audio_window(
    const nn::Tensor& audio, int window_start, int window_size, nn::Tensor& out, int out_row)
{
    for (int t = 0; t < window_size; ++t)
    {
        out.at(out_row, t) = audio.at(window_start + t, 0);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

AudioWindowDataset::AudioWindowDataset(
    std::vector<SubjectFiles> subjects, nn::windowing::WindowSpec spec)
    : subjects_(std::move(subjects)), spec_(spec)
{
    spec_.validate();

    windows_per_row_ = spec_.num_windows(kAudioSamples);
    if (windows_per_row_ == 0)
    {
        throw std::invalid_argument(
            "AudioWindowDataset: window_size exceeds audio signal length (" +
            std::to_string(kAudioSamples) + " samples)");
    }

    for (std::size_t s = 0; s < subjects_.size(); ++s)
    {
        const auto& sub = subjects_[s];
        for (std::size_t r = 0; r < sub.audio_rows; ++r)
        {
            for (int w = 0; w < windows_per_row_; ++w)
            {
                const int start = w * spec_.hop_size();
                index_table_.push_back({.subject_idx = s, .row_idx = r, .window_start = start});
            }
        }
    }

    audio_sessions_.resize(subjects_.size());
}

// ---------------------------------------------------------------------------
// size()
// ---------------------------------------------------------------------------

auto AudioWindowDataset::size() const -> std::size_t
{
    return index_table_.size();
}

// ---------------------------------------------------------------------------
// Lazy session init
// ---------------------------------------------------------------------------

void AudioWindowDataset::ensure_session(std::size_t subject_idx) const
{
    if (!audio_sessions_[subject_idx])
    {
        audio_sessions_[subject_idx] =
            std::make_unique<nn::dataLoaders::AudioSession>(subjects_[subject_idx].audio_path);
    }
}

// ---------------------------------------------------------------------------
// get_item()
// ---------------------------------------------------------------------------

auto AudioWindowDataset::get_item(std::size_t idx) const -> Batch
{
    if (idx >= index_table_.size())
    {
        throw std::out_of_range("AudioWindowDataset::get_item: index " + std::to_string(idx) +
                                " out of range [0, " + std::to_string(index_table_.size()) + ")");
    }

    const WindowIndex& wi = index_table_[idx];
    ensure_session(wi.subject_idx);

    const auto& session = *audio_sessions_[wi.subject_idx];
    const auto [audio_tensor, stimulus, eeg_index_label] = session.readRow(wi.row_idx);

    nn::Tensor inputs(1, spec_.window_size);
    extract_audio_window(audio_tensor, wi.window_start, spec_.window_size, inputs, 0);

    nn::Tensor targets(1, 2);
    targets.at(0, 0) = static_cast<float>(stimulus);
    targets.at(0, 1) = static_cast<float>(eeg_index_label);

    return {.inputs = std::move(inputs), .targets = std::move(targets)};
}

// ---------------------------------------------------------------------------
// collate_into()
// ---------------------------------------------------------------------------

void AudioWindowDataset::collate_into(const std::vector<std::size_t>& indices, Batch& batch) const
{
    if (indices.empty())
    {
        batch.inputs = nn::Tensor(0, spec_.window_size);
        batch.targets = nn::Tensor(0, 2);
        return;
    }

    const auto batch_size = static_cast<nn::Index>(indices.size());
    const auto input_cols = static_cast<nn::Index>(spec_.window_size);
    constexpr nn::Index target_cols = 2;

    if (batch.inputs.rows() != batch_size || batch.inputs.cols() != input_cols)
    {
        batch.inputs = nn::Tensor(batch_size, input_cols);
    }
    if (batch.targets.rows() != batch_size || batch.targets.cols() != target_cols)
    {
        batch.targets = nn::Tensor(batch_size, target_cols);
    }

    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        const WindowIndex& wi = index_table_[indices[i]];
        ensure_session(wi.subject_idx);

        const auto& session = *audio_sessions_[wi.subject_idx];
        const auto [audio_tensor, stimulus, eeg_index_label] = session.readRow(wi.row_idx);

        extract_audio_window(
            audio_tensor, wi.window_start, spec_.window_size, batch.inputs, static_cast<int>(i));

        batch.targets.at(static_cast<nn::Index>(i), 0) = static_cast<float>(stimulus);
        batch.targets.at(static_cast<nn::Index>(i), 1) = static_cast<float>(eeg_index_label);
    }
}

void AudioWindowDataset::print(IDatasetPrinter& printer) const
{
    auto* windowing_printer = dynamic_cast<WindowingDatasetPrinter*>(&printer);
    if (windowing_printer)
    {
        windowing_printer->print_audio_window(*this);
        return;
    }

    printer.print_generic(*this);
}
