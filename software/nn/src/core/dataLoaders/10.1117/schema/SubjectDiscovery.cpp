#include "nn/dataLoaders/10.1117/schema/SubjectDiscovery.hpp"

#include <sqlite3.h>

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

auto discoverSubjects(                        //
    const std::string& root_dir,              //
    const std::string& subject_regex_pattern, //
    bool use_shards                           //
    ) -> std::vector<SubjectFiles>
{
    regex selection_pattern(subject_regex_pattern);
    vector<SubjectFiles> subjects;

    namespace fs = std::filesystem;
    fs::path root_path(root_dir);

    if (!fs::exists(root_path))
    {
        throw std::runtime_error("Dataset root does not exist: " + root_dir);
    }

    // If a single sqlite DB file is provided, enumerate subjects from DB.
    if (fs::is_regular_file(root_path) && root_path.extension() == ".sqlite")
    {
        sqlite3* db = nullptr;
        if (sqlite3_open_v2(root_path.string().c_str(), &db, SQLITE_OPEN_READONLY, nullptr) !=
            SQLITE_OK)
        {
            if (db) sqlite3_close(db);
            throw std::runtime_error("Failed to open sqlite DB: " + root_dir);
        }

        const char* q = "SELECT id, subject_name FROM subject ORDER BY id ASC";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, q, -1, &stmt, nullptr) != SQLITE_OK)
        {
            sqlite3_close(db);
            throw std::runtime_error("Failed to prepare subject query");
        }

        while (sqlite3_step(stmt) == SQLITE_ROW)
        {
            SubjectFiles info{};
            info.subject_id = sqlite3_column_int(stmt, 0);
            const unsigned char* name = sqlite3_column_text(stmt, 1);
            info.subject_name =
                name ? reinterpret_cast<const char*>(name) : std::to_string(info.subject_id);
            info.eeg_mat_path = root_path.string();
            info.audio_mat_path = root_path.string();

            // Count eeg rows (trials with original_row not null) and audio rows
            sqlite3_stmt* cstmt = nullptr;
            const char* ceeg =
                "SELECT COUNT(*) FROM trial WHERE subject_id = ? AND original_row IS NOT NULL";
            if (sqlite3_prepare_v2(db, ceeg, -1, &cstmt, nullptr) == SQLITE_OK)
            {
                sqlite3_bind_int(cstmt, 1, info.subject_id);
                if (sqlite3_step(cstmt) == SQLITE_ROW)
                {
                    info.eeg_rows = static_cast<size_t>(sqlite3_column_int64(cstmt, 0));
                }
                sqlite3_finalize(cstmt);
            }
            const char* caudio =
                "SELECT COUNT(*) FROM audio_samples a JOIN trial t ON a.trial_id = t.id WHERE "
                "t.subject_id = ?";
            cstmt = nullptr;
            if (sqlite3_prepare_v2(db, caudio, -1, &cstmt, nullptr) == SQLITE_OK)
            {
                sqlite3_bind_int(cstmt, 1, info.subject_id);
                if (sqlite3_step(cstmt) == SQLITE_ROW)
                {
                    info.audio_rows = static_cast<size_t>(sqlite3_column_int64(cstmt, 0));
                }
                sqlite3_finalize(cstmt);
            }

            subjects.emplace_back(std::move(info));
        }

        sqlite3_finalize(stmt);
        sqlite3_close(db);

        if (subjects.empty())
        {
            throw std::runtime_error("No subjects found in sqlite DB: " + root_dir);
        }

        return subjects;
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
            info.eeg_rows = eeg_is_shard ? matioCpp::utils::countShardRows(info.eeg_mat_path, "eeg")
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