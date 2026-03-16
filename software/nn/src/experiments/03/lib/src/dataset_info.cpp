#include "dataset_info.hpp"

#include <iomanip>
#include <iostream>

#include "nn/dataLoaders/10.1117/Protocol101117Dataset.hpp"

using std::cout;
using std::endl;

void printDatasetSummary(const Protocol101117Dataset& dataset, const std::string& dataset_root)
{
    cout << "Dataset root: " << dataset_root << '\n';

    const auto& subjects = dataset.subjects();
    cout << "Subjects discovered: " << subjects.size() << '\n';

    // Determine column widths
    size_t subj_w = std::string("Subject").size();
    size_t audio_w = std::string("AudioRows").size();
    size_t eeg_w = std::string("EEGRows").size();
    for (const auto& s : subjects)
    {
        subj_w = std::max(subj_w, s.subject_name.size());
        audio_w = std::max(audio_w, std::to_string(s.audio_rows).size());
        eeg_w = std::max(eeg_w, std::to_string(s.eeg_rows).size());
    }

    // Header
    cout << std::left << std::setw(static_cast<int>(subj_w) + 2) << "Subject" << std::right
         << std::setw(static_cast<int>(audio_w) + 2) << "AudioRows"
         << std::setw(static_cast<int>(eeg_w) + 2) << "EEGRows" << '\n';

    // Separator
    cout << std::string(subj_w + audio_w + eeg_w + 6, '-') << '\n';

    // Rows
    for (const auto& s : subjects)
    {
        cout << std::left << std::setw(static_cast<int>(subj_w) + 2) << s.subject_name << std::right
             << std::setw(static_cast<int>(audio_w) + 2) << s.audio_rows
             << std::setw(static_cast<int>(eeg_w) + 2) << s.eeg_rows << '\n';
    }

    cout << '\n' << "Total synchronized samples: " << dataset.size() << "\n\n";
}
