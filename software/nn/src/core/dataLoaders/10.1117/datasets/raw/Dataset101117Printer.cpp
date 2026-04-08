/**
 * @file src/core/dataLoaders/10.1117/datasets/raw/Dataset101117Printer.cpp
 * @brief Implementation of Dataset101117Printer.
 */

#include "nn/dataLoaders/10.1117/datasets/raw/Dataset101117Printer.hpp"

#include <iomanip>
#include <iostream>

#include "nn/dataLoaders/10.1117/datasets/raw/Dataset101117.hpp"

using std::cout;

namespace
{

void print_dataset_summary(const Dataset101117& dataset, const std::string& dataset_root)
{
    cout << "Dataset root: " << dataset_root << '\n';

    const auto& subjects = dataset.subjects();
    cout << "Subjects discovered: " << subjects.size() << '\n';

    // Determine column widths and compute AudioWithEEG using a lightweight estimate.
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

        // Lightweight conservative estimate: audio rows that can have matching EEG.
        const size_t audio_with_eeg = std::min(s.audio_rows, s.eeg_rows);
        audio_with_eeg_counts.push_back(audio_with_eeg);
        audio_with_eeg_w = std::max(audio_with_eeg_w, std::to_string(audio_with_eeg).size());
        total_audio_with_eeg += audio_with_eeg;
    }

    // Header (without AudioRows column).
    cout << std::left << std::setw(static_cast<int>(subj_w) + 2) << "Subject" << std::right
         << std::setw(static_cast<int>(audio_with_eeg_w) + 2) << "AudioWithEEG"
         << std::setw(static_cast<int>(eeg_w) + 2) << "EEGRows" << '\n';

    // Separator.
    cout << std::string(subj_w + audio_with_eeg_w + eeg_w + 6, '-') << '\n';

    // Rows.
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

} // namespace

Dataset101117Printer::Dataset101117Printer(const std::string& dataset_root)
    : dataset_root_(dataset_root)
{
}

void Dataset101117Printer::print_generic(const Dataset& dataset)
{
    cout << "Generic dataset with " << dataset.size() << " samples." << '\n';
}

void Dataset101117Printer::print_protocol101117(const Dataset101117& dataset)
{
    print_dataset_summary(dataset, dataset_root_);
}
