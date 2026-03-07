#include "../include/subject_discovery.hpp"

#include <algorithm>
#include <filesystem>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

auto discoverSubjects(const std::string& root_dir, const std::string& subject_regex_pattern)
    -> std::vector<SubjectFiles>
{
    namespace fs = std::filesystem;

    fs::path root_path(root_dir);
    if (!fs::exists(root_path) || !fs::is_directory(root_path))
    {
        throw std::runtime_error("Dataset root does not exist or is not a directory: " + root_dir);
    }

    std::vector<SubjectFiles> subjects;
    std::regex selection_pattern(subject_regex_pattern);

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

        const fs::path eeg_path = entry.path() / (dir_name + "_EEG.mat");
        const fs::path audio_path = entry.path() / (dir_name + "_Audio.mat");

        if (!fs::exists(eeg_path) || !fs::exists(audio_path))
        {
            continue;
        }

        SubjectFiles info{};
        info.subject_id = subject_id;
        info.subject_name = dir_name;
        info.eeg_mat_path = eeg_path.string();
        info.audio_mat_path = audio_path.string();
        info.eeg_rows = countMatRows(info.eeg_mat_path, "EEG");
        info.audio_rows = countMatRows(info.audio_mat_path, "Audio");

        subjects.push_back(std::move(info));
    }

    if (subjects.empty())
    {
        throw std::runtime_error(
            "No valid subject directories found. "
            "Expected S01/S01_EEG.mat and S01/S01_Audio.mat");
    }

    std::sort(subjects.begin(),
              subjects.end(),
              [](const SubjectFiles& a, const SubjectFiles& b)
              { return a.subject_id < b.subject_id; });

    return subjects;
}