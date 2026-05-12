/**
 * @file src/core/dataLoaders/10.1117/datasets/windowed/WindowingDatasetPrinter.cpp
 * @brief Implementation of WindowingDatasetPrinter.
 */

#include "data_loaders/10.1117/datasets/windowed/WindowingDatasetPrinter.hpp"

#include <iostream>

#include "data_loaders/10.1117/datasets/windowed/AudioWindowDataset.hpp"
#include "data_loaders/10.1117/datasets/windowed/EEGWindowDataset.hpp"
#include "data_loaders/10.1117/datasets/windowed/FusedWindowDataset.hpp"

using std::cout;

WindowingDatasetPrinter::WindowingDatasetPrinter(const std::string& context) : context_(context) {}

void WindowingDatasetPrinter::print_generic(const Dataset& dataset)
{
    cout << "Dataset initialized with " << dataset.size() << " total samples.";
    if (!context_.empty())
    {
        cout << " (" << context_ << ")";
    }
    cout << '\n';
}

void WindowingDatasetPrinter::print_audio_window(const AudioWindowDataset& ds)
{
    const auto& spec = ds.spec();
    cout << "Audio window dataset summary";
    if (!context_.empty())
    {
        cout << " (" << context_ << ")";
    }
    cout << '\n';
    cout << "  window_size: " << spec.window_size << '\n';
    cout << "  overlap: " << spec.overlap << '\n';
    cout << "  hop_size: " << spec.hop_size() << '\n';
    cout << "  windows_per_row: " << ds.windows_per_row() << '\n';
    cout << "  total windows: " << ds.size() << '\n';

    if (ds.windows_per_row() > 0)
    {
        cout << "  estimated source rows: "
             << (ds.size() / static_cast<std::size_t>(ds.windows_per_row())) << '\n';
    }
}

void WindowingDatasetPrinter::print_eeg_window(const EEGWindowDataset& ds)
{
    const auto& spec = ds.spec();
    cout << "EEG window dataset summary";
    if (!context_.empty())
    {
        cout << " (" << context_ << ")";
    }
    cout << '\n';
    cout << "  window_size: " << spec.window_size << '\n';
    cout << "  overlap: " << spec.overlap << '\n';
    cout << "  hop_size: " << spec.hop_size() << '\n';
    cout << "  windows_per_row: " << ds.windows_per_row() << '\n';
    cout << "  total windows: " << ds.size() << '\n';

    if (ds.windows_per_row() > 0)
    {
        cout << "  estimated source rows: "
             << (ds.size() / static_cast<std::size_t>(ds.windows_per_row())) << '\n';
    }
}

void WindowingDatasetPrinter::print_fused_window(const FusedWindowDataset& ds)
{
    const auto& eeg_spec = ds.eeg_spec();
    const auto& audio_spec = ds.audio_spec();

    cout << "Fused window dataset summary";
    if (!context_.empty())
    {
        cout << " (" << context_ << ")";
    }
    cout << '\n';
    cout << "  eeg_window_size: " << eeg_spec.window_size
         << " | eeg_hop_size: " << eeg_spec.hop_size() << " | eeg_overlap: " << eeg_spec.overlap
         << '\n';
    cout << "  audio_window_size: " << audio_spec.window_size
         << " | audio_hop_size: " << audio_spec.hop_size()
         << " | audio_overlap: " << audio_spec.overlap << '\n';
    cout << "  windows_per_pair: " << ds.windows_per_pair() << '\n';
    cout << "  input_features: " << ds.input_features() << '\n';
    cout << "  total fused windows: " << ds.size() << '\n';

    if (ds.windows_per_pair() > 0)
    {
        cout << "  estimated source audio rows: "
             << (ds.size() / static_cast<std::size_t>(ds.windows_per_pair())) << '\n';
    }
}
