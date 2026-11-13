/**
 * @file src/core/data_loaders/10.1117/schema/SubjectDiscovery.cpp
 * @brief Implementation for Subjectdiscovery.
 *

 */

#include "data_loaders/10.1117/schema/SubjectDiscovery.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <data_loaders/mat_io/mat_file.hpp>
#include <filesystem>
#include <map>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "data_loaders/10.1117/schema/Names.hpp"
#include "data_loaders/10.1117/schema/SqliteDataset.hpp"
#include "data_loaders/mat_io/mat_file_utils.hpp"
#include "utility/path_expand.hpp"

using std::regex;
using std::vector;

using matioCpp::utils::countMatRows;

using nn::dataLoaders::kAudioMatFileSuffix;
using nn::dataLoaders::kAudioMatVariableName;
using nn::dataLoaders::kEegMatFileSuffix;
using nn::dataLoaders::kEegMatVariableName;
using nn::dataLoaders::openSqliteDatasetReadOnly;
using nn::dataLoaders::sqliteStepRow;

namespace
{

// COUNT(*) of one per-subject query. A query that cannot be prepared or stepped
// throws: it used to leave the count at 0, a subject that looked empty.
std::size_t countForSubject(sqlite3* db, const char* sql, int subject_id)
{
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        throw std::runtime_error(
            std::string("SubjectDiscovery: cannot prepare \"") + sql + "\": " + sqlite3_errmsg(db));
    sqlite3_bind_int(stmt, 1, subject_id);
    bool has_row = false;
    try
    {
        has_row = sqliteStepRow(db, stmt, sql);
    }
    catch (...)
    {
        sqlite3_finalize(stmt);
        throw;
    }
    if (!has_row)
    {
        sqlite3_finalize(stmt);
        throw std::runtime_error(std::string("SubjectDiscovery: \"") + sql + "\" returned no row");
    }
    const auto n = static_cast<std::size_t>(sqlite3_column_int64(stmt, 0));
    sqlite3_finalize(stmt);
    return n;
}

// Fills eeg_rows/audio_rows on `info` from the sqlite DB's trial/audio_samples tables.
void fillSubjectRowCountsFromSqlite(sqlite3* db, SubjectFiles& info)
{
    info.eeg_rows = countForSubject(db,
        "SELECT COUNT(*) FROM trial WHERE subject_id = ? AND original_row IS NOT NULL",
        info.subject_id);
    info.audio_rows = countForSubject(db,
        "SELECT COUNT(*) FROM audio_samples a JOIN trial t ON a.trial_id = t.id WHERE "
        "t.subject_id = ?",
        info.subject_id);
}

// Handles the "root_dir is a single sqlite DB file" case: enumerate subjects from the
// `subject` table and fill in their row counts.
auto discoverSubjectsFromSqlite(const std::string& root_dir,
    const std::filesystem::path& root_path,
    const regex& selection_pattern) -> vector<SubjectFiles>
{
    sqlite3* db = openSqliteDatasetReadOnly(root_path);

    const char* q = "SELECT id, subject_name FROM subject ORDER BY id ASC";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, q, -1, &stmt, nullptr) != SQLITE_OK)
    {
        const std::string err = sqlite3_errmsg(db);
        sqlite3_close(db);
        throw std::runtime_error(
            "SubjectDiscovery: cannot read the subject table of " + root_dir + ": " + err);
    }

    vector<SubjectFiles> subjects;
    // sqliteStepRow, not `== SQLITE_ROW`: an error mid-table used to end the loop like the
    // last row did, and the run went on with the subjects read so far.
    auto next_row = [&]()
    {
        try
        {
            return sqliteStepRow(db, stmt, q);
        }
        catch (...)
        {
            sqlite3_finalize(stmt);
            sqlite3_close(db);
            throw;
        }
    };
    while (next_row())
    {
        SubjectFiles info{};
        info.subject_id = sqlite3_column_int(stmt, 0);
        const unsigned char* name = sqlite3_column_text(stmt, 1);
        // A NULL name used to become the id; a name outside the caller's subject pattern
        // used to be accepted although the pattern was only ever applied to directories.
        if (name == nullptr)
        {
            sqlite3_finalize(stmt);
            sqlite3_close(db);
            throw std::runtime_error("SubjectDiscovery: subject " +
                                     std::to_string(info.subject_id) + " has no name in " +
                                     root_dir);
        }
        info.subject_name = reinterpret_cast<const char*>(name);
        if (!std::regex_match(info.subject_name, selection_pattern))
        {
            sqlite3_finalize(stmt);
            sqlite3_close(db);
            throw std::runtime_error("SubjectDiscovery: subject \"" + info.subject_name + "\" in " +
                                     root_dir + " does not match the subject pattern");
        }
        info.eeg_path = root_path.string();
        info.audio_path = root_path.string();

        // Count eeg rows (trials with original_row not null) and audio rows
        try
        {
            fillSubjectRowCountsFromSqlite(db, info);
        }
        catch (...)
        {
            sqlite3_finalize(stmt);
            sqlite3_close(db);
            throw;
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

// Matches one directory entry against the subject regex and returns its SubjectFiles.
// Returns nullopt only for entries that are not subjects — a non-directory or a name
// outside the pattern. A subject directory with a missing or unreadable EEG/audio file
// throws: it used to be skipped, so the dataset silently lost a speaker.
auto discoverSubjectFromDirEntry(const std::filesystem::directory_entry& entry,
    const regex& selection_pattern) -> std::optional<SubjectFiles>
{
    namespace fs = std::filesystem;

    if (!entry.is_directory())
    {
        return std::nullopt;
    }

    const std::string dir_name = entry.path().filename().string();

    std::smatch regex_groups_matches;
    if (!std::regex_match(dir_name, regex_groups_matches, selection_pattern))
    {
        return std::nullopt;
    }

    const int subject_id = std::stoi(regex_groups_matches[1].str());

    const fs::path eeg_mat = entry.path() / (dir_name + kEegMatFileSuffix());
    const fs::path audio_mat = entry.path() / (dir_name + kAudioMatFileSuffix());
    const fs::path shard_index = entry.path() / (dir_name + std::string("_shards.json"));

    bool eeg_ok = fs::exists(eeg_mat);
    bool audio_ok = fs::exists(audio_mat);
    bool eeg_is_shard = false;
    bool audio_is_shard = false;

    fs::path eeg_path = eeg_mat;
    fs::path audio_path = audio_mat;

    if (fs::exists(shard_index))
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
        throw std::runtime_error("SubjectDiscovery: subject directory " + entry.path().string() +
                                 " has no " + (eeg_ok ? "audio" : "EEG") +
                                 " file (.mat or shard index)");
    }

    SubjectFiles info{};
    info.subject_id = subject_id;
    info.subject_name = dir_name;
    info.eeg_path = eeg_path.string();
    info.audio_path = audio_path.string();

    try
    {
        info.eeg_rows = eeg_is_shard ? matioCpp::utils::countShardRows(info.eeg_path, "eeg")
                                     : countMatRows(info.eeg_path, kEegMatVariableName());
        info.audio_rows = audio_is_shard ? matioCpp::utils::countShardRows(info.audio_path, "audio")
                                         : countMatRows(info.audio_path, kAudioMatVariableName());
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("SubjectDiscovery: cannot read the files of subject directory " +
                                 entry.path().string() + ": " + e.what());
    }

    return info;
}

// Handles the "root_dir is a directory of per-subject folders" case: scan, filter and
// sort by subject_id.
auto discoverSubjectsFromDirectory(
    const std::filesystem::path& root_path, const regex& selection_pattern) -> vector<SubjectFiles>
{
    namespace fs = std::filesystem;

    vector<SubjectFiles> subjects;
    for (const auto& entry : fs::directory_iterator(root_path))
    {
        auto info = discoverSubjectFromDirEntry(entry, selection_pattern);
        if (info) subjects.emplace_back(std::move(*info));
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

} // namespace

auto discoverSubjects(                       //
    const std::string& root_dir,             //
    const std::string& subject_regex_pattern //
    ) -> std::vector<SubjectFiles>
{
    regex selection_pattern(subject_regex_pattern);

    namespace fs = std::filesystem;
    fs::path root_path(nn::utility::expand_home(root_dir));

    if (!fs::exists(root_path))
    {
        throw std::runtime_error("Dataset root does not exist: " + root_dir);
    }

    // If a single sqlite DB file is provided, enumerate subjects from DB.
    if (fs::is_regular_file(root_path) && root_path.extension() == ".sqlite")
    {
        return discoverSubjectsFromSqlite(root_dir, root_path, selection_pattern);
    }

    return discoverSubjectsFromDirectory(root_path, selection_pattern);
}

auto discoverStimulusNames(const std::string& root_dir) -> std::map<int, std::string>
{
    namespace fs = std::filesystem;
    const fs::path root_path(nn::utility::expand_home(root_dir));
    if (!(fs::is_regular_file(root_path) && root_path.extension() == ".sqlite"))
        throw std::runtime_error(
            "SubjectDiscovery: stimulus names come from the stimulus table of "
            "a .sqlite dataset; " +
            root_dir + " is not one");

    sqlite3* db = openSqliteDatasetReadOnly(root_path);
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, "SELECT id, name FROM stimulus ORDER BY id", -1, &stmt, nullptr) !=
        SQLITE_OK)
    {
        const std::string err = sqlite3_errmsg(db);
        sqlite3_close(db);
        throw std::runtime_error(
            "SubjectDiscovery: cannot read the stimulus table of " + root_dir + ": " + err);
    }
    std::map<int, std::string> names;
    int rc = SQLITE_ROW;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW)
    {
        const int id = sqlite3_column_int(stmt, 0);
        const unsigned char* name = sqlite3_column_text(stmt, 1);
        if (name == nullptr || *name == '\0')
        {
            sqlite3_finalize(stmt);
            sqlite3_close(db);
            throw std::runtime_error(
                "SubjectDiscovery: stimulus " + std::to_string(id) + " has no name in " + root_dir);
        }
        names.emplace(id, reinterpret_cast<const char*>(name));
    }
    const std::string err = sqlite3_errmsg(db);
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    if (rc != SQLITE_DONE)
        throw std::runtime_error(
            "SubjectDiscovery: reading the stimulus table of " + root_dir + " failed: " + err);
    if (names.empty())
        throw std::runtime_error(
            "SubjectDiscovery: the stimulus table of " + root_dir + " is empty");
    return names;
}
