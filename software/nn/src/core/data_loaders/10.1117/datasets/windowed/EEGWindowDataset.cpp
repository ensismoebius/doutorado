/**
 * @file EEGWindowDataset.cpp
 * @brief Implementation of EEGWindowDataset.
 *
 * EEG tensor from `EEGMatSession::readRow` has shape (eeg_channels, eeg_samples).
 * For the 10.1117 dataset: (6, 4096).
 *
 * Window extraction (channel-major output):
 *   out[0, c * window_size + t] = eeg[c, window_start + t]
 *   for c in [0, eeg_channels), t in [0, window_size).
 */

#include "data_loaders/10.1117/datasets/windowed/EEGWindowDataset.hpp"

#include <stdexcept>
#include <string>

#include "data_loaders/10.1117/datasets/windowed/WindowingDatasetPrinter.hpp"
#include "data_loaders/10.1117/schema/Metadata.hpp"
#include "windowing/WindowingEngine.hpp"

using nn::dataLoaders::ImaginedSpeechSchema_10_1117;
using nn::windowing::compute_windows;

namespace
{

// Schema-derived constants (compile-time).
constexpr int kEegChannels = static_cast<int>(ImaginedSpeechSchema_10_1117.eeg_channels);
constexpr int kEegSamplesPerChannel =
    static_cast<int>(ImaginedSpeechSchema_10_1117.eegSamplesPerChannel());

/// Extract one windowed EEG sample into a pre-allocated output tensor.
/// @param eeg          Full EEG row tensor, shape (eeg_channels, eeg_samples_per_channel).
/// @param window_start First time sample to include (inclusive).
/// @param window_size  Number of time samples.
/// @param out          Pre-allocated output tensor, shape (1, eeg_channels * window_size).
void extract_eeg_window(const nn::Tensor& eeg,
    int window_start,
    int window_size,
    int offset_col, // column offset within `out` (for batch filling)
    nn::Tensor& out,
    int out_row)
{
    for (int c = 0; c < kEegChannels; ++c)
    {
        for (int t = 0; t < window_size; ++t)
        {
            out.at(out_row, offset_col + c * window_size + t) = eeg.at(c, window_start + t);
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

EEGWindowDataset::EEGWindowDataset(
    std::vector<SubjectFiles> subjects, nn::windowing::WindowSpec spec)
    : subjects_(std::move(subjects)), spec_(spec)
{
    spec_.validate();

    // Pre-compute number of windows per row from schema constant.
    windows_per_row_ = spec_.num_windows(kEegSamplesPerChannel);
    if (windows_per_row_ == 0)
    {
        throw std::invalid_argument("EEGWindowDataset: window_size exceeds EEG signal length (" +
                                    std::to_string(kEegSamplesPerChannel) + " samples)");
    }

    // Build flat index table.
    for (std::size_t s = 0; s < subjects_.size(); ++s)
    {
        const auto& sub = subjects_[s];
        for (std::size_t r = 0; r < sub.eeg_rows; ++r)
        {
            for (int w = 0; w < windows_per_row_; ++w)
            {
                const int start = w * spec_.hop_size();
                index_table_.push_back({.subject_idx = s, .row_idx = r, .window_start = start});
            }
        }
    }

    // Reserve session slots (lazy init).
    eeg_sessions_.resize(subjects_.size());
}

// ---------------------------------------------------------------------------
// size()
// ---------------------------------------------------------------------------

auto EEGWindowDataset::size() const -> std::size_t
{
    return index_table_.size();
}

// ---------------------------------------------------------------------------
// Lazy session init
// ---------------------------------------------------------------------------

void EEGWindowDataset::ensure_session(std::size_t subject_idx) const
{
    if (!eeg_sessions_[subject_idx])
    {
        eeg_sessions_[subject_idx] =
            std::make_unique<nn::dataLoaders::EEGMatSession>(subjects_[subject_idx].eeg_mat_path);
    }
}

// ---------------------------------------------------------------------------
// get_item()
// ---------------------------------------------------------------------------

auto EEGWindowDataset::get_item(std::size_t idx) const -> Batch
{
    if (idx >= index_table_.size())
    {
        throw std::out_of_range("EEGWindowDataset::get_item: index " + std::to_string(idx) +
                                " out of range [0, " + std::to_string(index_table_.size()) + ")");
    }

    const WindowIndex& wi = index_table_[idx];
    ensure_session(wi.subject_idx);

    const auto& session = *eeg_sessions_[wi.subject_idx];
    const auto [eeg_tensor, eeg_labels] = session.readRow(wi.row_idx);

    const int ws = spec_.window_size;
    const int out_cols = kEegChannels * ws;

    nn::Tensor inputs(1, out_cols);
    extract_eeg_window(eeg_tensor, wi.window_start, ws, 0, inputs, 0);

    // targets: [subject_id, lbl[0], lbl[1] (stimulus), lbl[2] (blink)]
    nn::Tensor targets(1, 3);
    targets.at(0, 0) = static_cast<float>(eeg_labels[0]);
    targets.at(0, 1) = static_cast<float>(eeg_labels[1]);
    targets.at(0, 2) = static_cast<float>(eeg_labels[2]);

    return {.inputs = std::move(inputs), .targets = std::move(targets)};
}

// ---------------------------------------------------------------------------
// collate_into()
// ---------------------------------------------------------------------------
// Overrides the default Dataset::collate_into to pre-allocate the batch tensor
// once and fill it in-place, avoiding per-call allocation inside the loop.

void EEGWindowDataset::collate_into(const std::vector<std::size_t>& indices, Batch& batch) const
{
    if (indices.empty())
    {
        batch.inputs = nn::Tensor(0, kEegChannels * spec_.window_size);
        batch.targets = nn::Tensor(0, 3);
        return;
    }

    const auto batch_size = static_cast<nn::Index>(indices.size());
    const auto input_cols = static_cast<nn::Index>(kEegChannels * spec_.window_size);
    constexpr nn::Index target_cols = 3;

    // Resize output tensors only when dimensions change (avoids malloc on
    // consecutive calls from the DataLoaderIterator's pre-allocated Batch).
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
        const std::size_t idx = indices[i];
        const WindowIndex& wi = index_table_[idx];
        ensure_session(wi.subject_idx);

        const auto& session = *eeg_sessions_[wi.subject_idx];
        const auto [eeg_tensor, eeg_labels] = session.readRow(wi.row_idx);

        extract_eeg_window(
            eeg_tensor, wi.window_start, spec_.window_size, 0, batch.inputs, static_cast<int>(i));

        batch.targets.at(static_cast<nn::Index>(i), 0) = static_cast<float>(eeg_labels[0]);
        batch.targets.at(static_cast<nn::Index>(i), 1) = static_cast<float>(eeg_labels[1]);
        batch.targets.at(static_cast<nn::Index>(i), 2) = static_cast<float>(eeg_labels[2]);
    }
}

void EEGWindowDataset::print(IDatasetPrinter& printer) const
{
    auto* windowing_printer = dynamic_cast<WindowingDatasetPrinter*>(&printer);
    if (windowing_printer)
    {
        windowing_printer->print_eeg_window(*this);
        return;
    }

    printer.print_generic(*this);
}
