/**
 * @file FusedWindowDataset.cpp
 * @brief Implementation of FusedWindowDataset (Option 3 — fused EEG + audio pipeline).
 *
 * For each audio row in each subject:
 *  1. Resolve the matching EEG row at get_item time via `eeg_index_label` in the
 *     audio record (avoids expensive eager I/O in the constructor).
 *  2. Extract window `k` from both modalities.
 *  3. Concatenate channel-major EEG window with audio window.
 *
 * Input features layout per sample (all floats, row-major):
 *   [eeg_ch0_t0 .. eeg_ch0_tW-1 | eeg_ch1_t0 .. | ... | eeg_ch5_tW-1 | aud_t0 .. aud_tM-1]
 *   where W = eeg_window_size, M = audio_window_size.
 */

#include "nn/dataLoaders/10.1117/windowing/FusedWindowDataset.hpp"

#include <stdexcept>
#include <string>
#include <tuple>

#include "nn/dataLoaders/10.1117/schema/METADATA.hpp"
#include "nn/dataLoaders/10.1117/schema/SchemaIndexing.hpp"

using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
using nn::dataLoaders::schema101117::resolveEegRowIndex;

namespace
{

constexpr int kEegChannels = static_cast<int>(ImaginedSpeechSchema_10_1117.eeg_channels);
constexpr int kEegSamplesPC = static_cast<int>(ImaginedSpeechSchema_10_1117.eegSamplesPerChannel());
constexpr int kAudioSamples = static_cast<int>(ImaginedSpeechSchema_10_1117.audioSamples());

/// Write one EEG window (channel-major) into `out` at the specified row,
/// starting at column `col_offset`.
void write_eeg_window(const nn::Tensor& eeg,
    int window_start,
    int eeg_ws,
    nn::Tensor& out,
    int out_row,
    int col_offset)
{
    // Bounds checks to avoid silent buffer overruns that can corrupt nearby stack
    const int out_rows = static_cast<int>(out.rows());
    const int out_cols = static_cast<int>(out.cols());
    const int eeg_cols = static_cast<int>(eeg.cols());

    if (out_row < 0 || out_row >= out_rows)
        throw std::out_of_range("write_eeg_window: out_row out of range");
    if (window_start < 0 || (window_start + eeg_ws) > eeg_cols)
        throw std::out_of_range("write_eeg_window: eeg window out of range");
    if (col_offset < 0 || (col_offset + kEegChannels * eeg_ws) > out_cols)
        throw std::out_of_range("write_eeg_window: output columns out of range");

    for (int c = 0; c < kEegChannels; ++c)
    {
        for (int t = 0; t < eeg_ws; ++t)
        {
            out.at(out_row, col_offset + c * eeg_ws + t) = eeg.at(c, window_start + t);
        }
    }
}

/// Write one audio window into `out` at the specified row,
/// starting at column `col_offset`.
void write_audio_window(const nn::Tensor& audio,
    int window_start,
    int audio_ws,
    nn::Tensor& out,
    int out_row,
    int col_offset)
{
    const int out_rows = static_cast<int>(out.rows());
    const int out_cols = static_cast<int>(out.cols());
    const int audio_rows = static_cast<int>(audio.rows());

    if (out_row < 0 || out_row >= out_rows)
        throw std::out_of_range("write_audio_window: out_row out of range");
    if (window_start < 0 || (window_start + audio_ws) > audio_rows)
        throw std::out_of_range("write_audio_window: audio window out of range");
    if (col_offset < 0 || (col_offset + audio_ws) > out_cols)
        throw std::out_of_range("write_audio_window: output columns out of range");

    for (int t = 0; t < audio_ws; ++t)
    {
        out.at(out_row, col_offset + t) = audio.at(window_start + t, 0);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

FusedWindowDataset::FusedWindowDataset(std::vector<SubjectFiles> subjects,
    nn::windowing::WindowSpec eeg_spec,
    nn::windowing::WindowSpec audio_spec)
    : subjects_(std::move(subjects)), eeg_spec_(eeg_spec), audio_spec_(audio_spec)
{
    eeg_spec_.validate();
    audio_spec_.validate();

    const int eeg_n = eeg_spec_.num_windows(kEegSamplesPC);
    const int audio_n = audio_spec_.num_windows(kAudioSamples);

    if (eeg_n == 0 || audio_n == 0)
    {
        throw std::invalid_argument(
            "FusedWindowDataset: window_size too large for signal length "
            "(eeg_windows=" +
            std::to_string(eeg_n) + ", audio_windows=" + std::to_string(audio_n) + ")");
    }

    windows_per_pair_ = std::min(eeg_n, audio_n);
    input_features_ = kEegChannels * eeg_spec_.window_size + audio_spec_.window_size;

    // Build flat index table: one entry per (subject, audio_row, window_k).
    for (std::size_t s = 0; s < subjects_.size(); ++s)
    {
        const std::size_t n_audio_rows = subjects_[s].audio_rows;
        for (std::size_t r = 0; r < n_audio_rows; ++r)
        {
            for (int k = 0; k < windows_per_pair_; ++k)
            {
                index_table_.push_back({.subject_idx = s, .audio_row = r, .window_k = k});
            }
        }
    }

    audio_sessions_.resize(subjects_.size());
    eeg_sessions_.resize(subjects_.size());
}

// ---------------------------------------------------------------------------
// size()
// ---------------------------------------------------------------------------

auto FusedWindowDataset::size() const -> std::size_t
{
    return index_table_.size();
}

// ---------------------------------------------------------------------------
// Lazy session init
// ---------------------------------------------------------------------------

void FusedWindowDataset::ensure_sessions(std::size_t subject_idx) const
{
    const auto& sub = subjects_[subject_idx];
    if (!audio_sessions_[subject_idx])
    {
        audio_sessions_[subject_idx] =
            std::make_unique<nn::dataLoaders::AudioMatSession>(sub.audio_mat_path);
    }
    if (!eeg_sessions_[subject_idx])
    {
        eeg_sessions_[subject_idx] =
            std::make_unique<nn::dataLoaders::EEGMatSession>(sub.eeg_mat_path);
    }
}

// ---------------------------------------------------------------------------
// get_item()
// ---------------------------------------------------------------------------

auto FusedWindowDataset::get_item(std::size_t idx) const -> Batch
{
    if (idx >= index_table_.size())
    {
        throw std::out_of_range("FusedWindowDataset::get_item: index " + std::to_string(idx) +
                                " out of range [0, " + std::to_string(index_table_.size()) + ")");
    }

    const WindowIndex& wi = index_table_[idx];
    ensure_sessions(wi.subject_idx);

    const auto& audio_session = *audio_sessions_[wi.subject_idx];
    const auto& eeg_session = *eeg_sessions_[wi.subject_idx];
    const auto& sub = subjects_[wi.subject_idx];

    // Read audio row — also retrieves eeg_index_label for cross-modal sync.
    auto audio_row_tuple = audio_session.readRow(wi.audio_row);
    nn::Tensor audio_tensor = std::move(std::get<0>(audio_row_tuple));
    const int stimulus = std::get<1>(audio_row_tuple);
    const int eeg_index_label = std::get<2>(audio_row_tuple);

    // Resolve matching EEG row.
    const std::size_t eeg_row = resolveEegRowIndex(eeg_index_label, sub.eeg_rows);
    auto eeg_row_tuple = eeg_session.readRow(eeg_row);
    nn::Tensor eeg_tensor = std::move(std::get<0>(eeg_row_tuple));
    const auto eeg_labels = std::get<1>(eeg_row_tuple);

    // Compute sample offsets for window k.
    const int eeg_start = wi.window_k * eeg_spec_.hop_size();
    const int audio_start = wi.window_k * audio_spec_.hop_size();

    // Build fused input: [EEG_flat | audio_flat].
    nn::Tensor inputs(1, input_features_);
    write_eeg_window(eeg_tensor, eeg_start, eeg_spec_.window_size, inputs, 0, 0);
    write_audio_window(audio_tensor,
        audio_start,
        audio_spec_.window_size,
        inputs,
        0,
        kEegChannels * eeg_spec_.window_size);

    // Targets: [subject_id, eeg_lbl[0], eeg_lbl[1], eeg_lbl[2], eeg_index].
    nn::Tensor targets(1, 5);
    targets.at(0, 0) = static_cast<float>(sub.subject_id);
    targets.at(0, 1) = static_cast<float>(eeg_labels[0]);
    targets.at(0, 2) = static_cast<float>(eeg_labels[1]);
    targets.at(0, 3) = static_cast<float>(eeg_labels[2]);
    targets.at(0, 4) = static_cast<float>(eeg_index_label);

    return {.inputs = std::move(inputs), .targets = std::move(targets)};
}

// ---------------------------------------------------------------------------
// collate_into()
// ---------------------------------------------------------------------------

void FusedWindowDataset::collate_into(const std::vector<std::size_t>& indices, Batch& batch) const
{
    if (indices.empty())
    {
        batch.inputs = nn::Tensor(0, input_features_);
        batch.targets = nn::Tensor(0, 5);
        return;
    }

    const auto batch_size = static_cast<nn::Index>(indices.size());
    const auto input_cols = static_cast<nn::Index>(input_features_);
    constexpr nn::Index target_cols = 5;

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
        ensure_sessions(wi.subject_idx);

        const auto& audio_session = *audio_sessions_[wi.subject_idx];
        const auto& eeg_session = *eeg_sessions_[wi.subject_idx];
        const auto& sub = subjects_[wi.subject_idx];

        auto audio_row_tuple = audio_session.readRow(wi.audio_row);
        nn::Tensor audio_tensor = std::move(std::get<0>(audio_row_tuple));
        const int stimulus = std::get<1>(audio_row_tuple);
        const int eeg_index_label = std::get<2>(audio_row_tuple);

        const std::size_t eeg_row = resolveEegRowIndex(eeg_index_label, sub.eeg_rows);
        auto eeg_row_tuple = eeg_session.readRow(eeg_row);
        nn::Tensor eeg_tensor = std::move(std::get<0>(eeg_row_tuple));
        const auto eeg_labels = std::get<1>(eeg_row_tuple);

        const int eeg_start = wi.window_k * eeg_spec_.hop_size();
        const int audio_start = wi.window_k * audio_spec_.hop_size();
        const int row = static_cast<int>(i);

        write_eeg_window(eeg_tensor, eeg_start, eeg_spec_.window_size, batch.inputs, row, 0);
        write_audio_window(audio_tensor,
            audio_start,
            audio_spec_.window_size,
            batch.inputs,
            row,
            kEegChannels * eeg_spec_.window_size);

        batch.targets.at(row, 0) = static_cast<float>(sub.subject_id);
        batch.targets.at(row, 1) = static_cast<float>(eeg_labels[0]);
        batch.targets.at(row, 2) = static_cast<float>(eeg_labels[1]);
        batch.targets.at(row, 3) = static_cast<float>(eeg_labels[2]);
        batch.targets.at(row, 4) = static_cast<float>(eeg_index_label);
    }
}
