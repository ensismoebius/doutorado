/**
 * @file src/experiments/03/lib/src/dataset_info.cpp
 * @brief Helpers to summarize dataset statistics for the Experiment03 demos.
 */

#include "nn/dataLoaders/10.1117/dataset_info.hpp"

#include <iomanip>
#include <iostream>

#include "nn/dataLoaders/10.1117/datasets/raw/Dataset101117.hpp"
#include "nn/dataLoaders/10.1117/datasets/windowed/AudioWindowDataset.hpp"
#include "nn/dataLoaders/10.1117/datasets/windowed/EEGWindowDataset.hpp"
#include "nn/dataLoaders/10.1117/datasets/windowed/FusedWindowDataset.hpp"

using std::cout;
using std::endl;

void printDatasetSummary(const Dataset101117& dataset, const std::string& dataset_root)
{
    cout << "Dataset root: " << dataset_root << '\n';

    const auto& subjects = dataset.subjects();
    cout << "Subjects discovered: " << subjects.size() << '\n';

    // Determine column widths and compute AudioWithEEG using a lightweight estimate
    size_t subj_w = std::string("Subject").size();
    size_t audio_with_eeg_w = std::string("AudioWithEEG").size();
    size_t eeg_w = std::string("EEGRows").size();

    std::vector<size_t> audio_with_eeg_counts;
    audio_with_eeg_counts.reserve(subjects.size());

    size_t total_audio_with_eeg = 0;

    for (const auto& s : subjects)
    {
        subj_w = std::max(subj_w, s.subject_name.size());
        eeg_w = std::max(eeg_w, std::to_string(s.eeg_rows).size());

        // Lightweight conservative estimate: audio rows that can have matching EEG
        const size_t audio_with_eeg = std::min(s.audio_rows, s.eeg_rows);
        audio_with_eeg_counts.push_back(audio_with_eeg);
        audio_with_eeg_w = std::max(audio_with_eeg_w, std::to_string(audio_with_eeg).size());
        total_audio_with_eeg += audio_with_eeg;
    }

    // Header (without AudioRows column)
    cout << std::left << std::setw(static_cast<int>(subj_w) + 2) << "Subject" << std::right
         << std::setw(static_cast<int>(audio_with_eeg_w) + 2) << "AudioWithEEG"
         << std::setw(static_cast<int>(eeg_w) + 2) << "EEGRows" << '\n';

    // Separator
    cout << std::string(subj_w + audio_with_eeg_w + eeg_w + 6, '-') << '\n';

    // Rows
    for (size_t i = 0; i < subjects.size(); ++i)
    {
        const auto& s = subjects[i];
        const size_t audio_with_eeg = audio_with_eeg_counts[i];
        cout << std::left << std::setw(static_cast<int>(subj_w) + 2) << s.subject_name << std::right
             << std::setw(static_cast<int>(audio_with_eeg_w) + 2) << audio_with_eeg
             << std::setw(static_cast<int>(eeg_w) + 2) << s.eeg_rows << '\n';
    }

    cout << '\n' << "Total synchronized samples: " << dataset.size() << '\n';
    cout << "Total audio rows with matching EEG (estimate): " << total_audio_with_eeg << "\n\n";
}

// ============================================================================
// Dataset101117Printer Implementation
// ============================================================================

Dataset101117Printer::Dataset101117Printer(const std::string& dataset_root)
    : dataset_root_(dataset_root)
{
}

void Dataset101117Printer::print_generic(const Dataset& dataset)
{
    // Default generic printer: print basic dataset info
    cout << "Generic dataset with " << dataset.size() << " samples." << '\n';
}

void Dataset101117Printer::print_protocol101117(const Dataset101117& dataset)
{
    printDatasetSummary(dataset, dataset_root_);
}

// ============================================================================
// WindowingDatasetPrinter Implementation
// ============================================================================

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
