#include "nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp"

#include <algorithm>
#include <filesystem>
#include <nn/dataLoaders/mat_file.hpp>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nn/dataLoaders/10.1117/schema/NAMES.hpp"
#include "nn/dataLoaders/mat_file_utils.hpp"

using std::regex;
using std::vector;

using matioCpp::utils::countMatRows;

using nn::dataLoaders::kAudioMatFileSuffix;
using nn::dataLoaders::kAudioMatVariableName;
using nn::dataLoaders::kEegMatFileSuffix;
using nn::dataLoaders::kEegMatVariableName;

auto discoverSubjects(                       //
    const std::string& root_dir,             //
    const std::string& subject_regex_pattern, //
    bool use_shards                           //
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

        const fs::path eeg_mat = entry.path() / (dir_name + kEegMatFileSuffix);
        const fs::path audio_mat = entry.path() / (dir_name + kAudioMatFileSuffix);
        const fs::path shard_index = entry.path() / (dir_name + std::string("_shards.json"));

        bool eeg_ok = fs::exists(eeg_mat);
        bool audio_ok = fs::exists(audio_mat);
        bool eeg_is_shard = false;
        bool audio_is_shard = false;

        fs::path eeg_path = eeg_mat;
        fs::path audio_path = audio_mat;

        if (use_shards && fs::exists(shard_index))
        {
            if (!eeg_ok)
            {
                eeg_ok = true;
                eeg_is_shard = true;
                eeg_path = shard_index;
            }
            if (!audio_ok)
            {
                audio_ok = true;
                audio_is_shard = true;
                audio_path = shard_index;
            }
        }

        if (!eeg_ok || !audio_ok)
        {
            continue;
        }

        SubjectFiles info{};
        info.subject_id = subject_id;
        info.subject_name = dir_name;
        info.eeg_mat_path = eeg_path.string();
        info.audio_mat_path = audio_path.string();

        try
        {
            info.eeg_rows = eeg_is_shard
                                ? matioCpp::utils::countShardRows(info.eeg_mat_path, "eeg")
                                : countMatRows(info.eeg_mat_path, kEegMatVariableName);
            info.audio_rows = audio_is_shard
                                  ? matioCpp::utils::countShardRows(info.audio_mat_path, "audio")
                                  : countMatRows(info.audio_mat_path, kAudioMatVariableName);
        }
        catch (const std::exception&)
        {
            continue;
        }

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