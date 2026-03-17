#include "nn/dataLoaders/10.1117/SubjectDiscovery.hpp"

#include <algorithm>
#include <filesystem>
#include <nn/dataLoaders/mat_file.hpp>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nn/dataLoaders/10.1117/NAMES.hpp"
#include "nn/dataLoaders/mat_file_utils.hpp"

using std::regex;
using std::vector;

using matioCpp::utils::countMatRows;

using nn::dataLoaders::AUDIO_MAT_FILE_SUFFIX;
using nn::dataLoaders::AUDIO_MAT_VARIABLE_NAME;
using nn::dataLoaders::EEG_MAT_FILE_SUFFIX;
using nn::dataLoaders::EEG_MAT_VARIABLE_NAME;

auto discoverSubjects(                       //
    const std::string& root_dir,             //
    const std::string& subject_regex_pattern //
    ) -> std::vector<SubjectFiles>
{
    regex selection_pattern(subject_regex_pattern);
    vector<SubjectFiles> subjects;

    namespace fs = std::filesystem;
    fs::path root_path(root_dir);

    if (!fs::exists(root_path) || !fs::is_directory(root_path))
    {
        throw std::runtime_error(
            "Dataset root does not "
            "exist or is not a directory: " +
            root_dir);
    }

    for (const auto& entry : fs::directory_iterator(root_path))
    {
        if (!entry.is_directory())
        {
            continue;
        }

        const std::string dir_name = entry.path().filename().string();

        std::smatch regex_groups_matches;
        if (!std::regex_match(dir_name, regex_groups_matches, selection_pattern))
        {
            continue;
        }

        const int subject_id = std::stoi(regex_groups_matches[1].str());

        const fs::path eeg_path = entry.path() / (dir_name + EEG_MAT_FILE_SUFFIX);
        const fs::path audio_path = entry.path() / (dir_name + AUDIO_MAT_FILE_SUFFIX);

        if (!fs::exists(eeg_path) || !fs::exists(audio_path))
        {
            continue;
        }

        SubjectFiles info{};
        info.subject_id = subject_id;
        info.subject_name = dir_name;
        info.eeg_mat_path = eeg_path.string();
        info.audio_mat_path = audio_path.string();
        info.eeg_rows = countMatRows(info.eeg_mat_path, EEG_MAT_VARIABLE_NAME);
        info.audio_rows = countMatRows(info.audio_mat_path, AUDIO_MAT_VARIABLE_NAME);

        subjects.emplace_back(std::move(info));
    }

    if (subjects.empty())
    {
        throw std::runtime_error(
            "No valid subject directories found. "
            "Expected S01/S01_EEG.mat and S01/S01_Audio.mat");
    }

    std::sort(subjects.begin(),
        subjects.end(),
        [](const SubjectFiles& a, const SubjectFiles& b) { return a.subject_id < b.subject_id; });

    return subjects;
}