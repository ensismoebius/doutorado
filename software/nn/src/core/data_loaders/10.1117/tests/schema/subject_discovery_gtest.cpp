/**
 * @file subject_discovery_gtest.cpp
 * @brief Exercises subject discovery success and defensive branches.
 */

#include <gtest/gtest.h>
#include <sqlite3.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "data_loaders/10.1117/schema/SubjectDiscovery.hpp"

namespace
{
class SubjectDiscoveryTest : public ::testing::Test
{
   protected:
    std::filesystem::path tmp_root_;

    void SetUp() override
    {
        tmp_root_ = std::filesystem::temp_directory_path() /
                    ("subject_discovery_" + std::to_string(getpid()));
        std::filesystem::remove_all(tmp_root_);
        std::filesystem::create_directories(tmp_root_);
    }

    void TearDown() override
    {
        std::filesystem::remove_all(tmp_root_);
    }

    static void execOrThrow(sqlite3* db, const char* sql)
    {
        char* err = nullptr;
        const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err);
        if (rc != SQLITE_OK)
        {
            const std::string message = err ? err : "sqlite exec failed";
            if (err) sqlite3_free(err);
            throw std::runtime_error(message);
        }
    }

    static auto createSqliteDb(const std::filesystem::path& path) -> sqlite3*
    {
        sqlite3* db = nullptr;
        const int rc = sqlite3_open_v2(
            path.string().c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
        if (rc != SQLITE_OK)
        {
            if (db) sqlite3_close(db);
            throw std::runtime_error("sqlite open failed");
        }
        return db;
    }

    // The trial/audio_samples tables the per-subject row counts are read from.
    static void createCountTables(sqlite3* db)
    {
        execOrThrow(db,
            "CREATE TABLE trial(id INTEGER PRIMARY KEY, subject_id INTEGER, stimulus_id INTEGER, "
            "original_row INTEGER);");
        execOrThrow(db,
            "CREATE TABLE audio_samples(id INTEGER PRIMARY KEY, trial_id INTEGER, samples BLOB, "
            "audio_row INTEGER);");
    }
};
} // namespace

TEST_F(SubjectDiscoveryTest, ThrowsWhenDatasetRootDoesNotExist)
{
    const auto missing_path = tmp_root_ / "missing_root";
    EXPECT_THROW((void) discoverSubjects(missing_path.string(), "^S(\\d+)$"), std::runtime_error);
}

TEST_F(SubjectDiscoveryTest, ThrowsWhenDirectoryContainsNoValidSubjectPairs)
{
    std::filesystem::create_directories(tmp_root_ / "invalid_subject");
    EXPECT_THROW((void) discoverSubjects(tmp_root_.string(), "^S(\\d+)$"), std::runtime_error);
}

TEST_F(SubjectDiscoveryTest, ShardIndexFallbackStillThrowsWhenShardIndexIsInvalid)
{
    const auto subject_dir = tmp_root_ / "S01";
    std::filesystem::create_directories(subject_dir);

    std::ofstream shard_index(subject_dir / "S01_shards.json");
    shard_index << "{\"invalid\":true}";
    shard_index.close();

    EXPECT_THROW((void) discoverSubjects(tmp_root_.string(), "^S(\\d+)$"), std::runtime_error);
}

TEST_F(SubjectDiscoveryTest, SqliteThrowsWhenSubjectTableIsMissing)
{
    const auto db_path = tmp_root_ / "dataset.sqlite";
    sqlite3* db = createSqliteDb(db_path);
    sqlite3_close(db);

    EXPECT_THROW((void) discoverSubjects(db_path.string(), "^S(\\d+)$"), std::runtime_error);
}

TEST_F(SubjectDiscoveryTest, SqliteThrowsWhenNoSubjectsArePresent)
{
    const auto db_path = tmp_root_ / "dataset.sqlite";
    sqlite3* db = createSqliteDb(db_path);
    execOrThrow(db, "CREATE TABLE subject(id INTEGER PRIMARY KEY, subject_name TEXT);");
    sqlite3_close(db);

    EXPECT_THROW((void) discoverSubjects(db_path.string(), "^S(\\d+)$"), std::runtime_error);
}

// The cases below were all accepted until 2026-11-12: a NULL name became the id, a
// count query that could not run left the count at 0, and the subject pattern was never
// applied to a database's subjects.
TEST_F(SubjectDiscoveryTest, SqliteRefusesNullSubjectName)
{
    const auto db_path = tmp_root_ / "dataset.sqlite";
    sqlite3* db = createSqliteDb(db_path);
    execOrThrow(db, "CREATE TABLE subject(id INTEGER PRIMARY KEY, subject_name TEXT);");
    execOrThrow(db, "INSERT INTO subject(id, subject_name) VALUES (42, NULL);");
    createCountTables(db);
    sqlite3_close(db);

    EXPECT_THROW((void) discoverSubjects(db_path.string(), "^S(\\d+)$"), std::runtime_error);
}

TEST_F(SubjectDiscoveryTest, SqliteRefusesMissingCountTables)
{
    const auto db_path = tmp_root_ / "dataset.sqlite";
    sqlite3* db = createSqliteDb(db_path);
    execOrThrow(db, "CREATE TABLE subject(id INTEGER PRIMARY KEY, subject_name TEXT);");
    execOrThrow(db, "INSERT INTO subject(id, subject_name) VALUES (1, 'S01');");
    sqlite3_close(db);

    EXPECT_THROW((void) discoverSubjects(db_path.string(), "^S(\\d+)$"), std::runtime_error);
}

TEST_F(SubjectDiscoveryTest, SqliteRefusesSubjectOutsideThePattern)
{
    const auto db_path = tmp_root_ / "dataset.sqlite";
    sqlite3* db = createSqliteDb(db_path);
    execOrThrow(db, "CREATE TABLE subject(id INTEGER PRIMARY KEY, subject_name TEXT);");
    execOrThrow(db, "INSERT INTO subject(id, subject_name) VALUES (1, 'subject-one');");
    createCountTables(db);
    sqlite3_close(db);

    EXPECT_THROW((void) discoverSubjects(db_path.string(), "^S(\\d+)$"), std::runtime_error);
}

TEST_F(SubjectDiscoveryTest, SqliteCountsRowsPerSubject)
{
    const auto db_path = tmp_root_ / "dataset.sqlite";
    sqlite3* db = createSqliteDb(db_path);
    execOrThrow(db, "CREATE TABLE subject(id INTEGER PRIMARY KEY, subject_name TEXT);");
    execOrThrow(db, "INSERT INTO subject(id, subject_name) VALUES (1, 'S01');");
    createCountTables(db);
    execOrThrow(db,
        "INSERT INTO trial(id, subject_id, stimulus_id, original_row) VALUES (1, 1, 1, 1), "
        "(2, 1, 2, 2), (3, 1, 2, NULL);");
    execOrThrow(db, "INSERT INTO audio_samples(trial_id, audio_row) VALUES (1, 0);");
    sqlite3_close(db);

    const auto subjects = discoverSubjects(db_path.string(), "^S(\\d+)$");
    ASSERT_EQ(subjects.size(), 1U);
    EXPECT_EQ(subjects[0].subject_name, "S01");
    EXPECT_EQ(subjects[0].eeg_rows, 2U);
    EXPECT_EQ(subjects[0].audio_rows, 1U);
}

TEST_F(SubjectDiscoveryTest, StimulusNamesComeFromTheStimulusTable)
{
    const auto db_path = tmp_root_ / "dataset.sqlite";
    sqlite3* db = createSqliteDb(db_path);
    execOrThrow(db, "CREATE TABLE stimulus(id INTEGER PRIMARY KEY, name TEXT UNIQUE);");
    execOrThrow(db, "INSERT INTO stimulus(id, name) VALUES (1, 'A'), (5, 'Atras'), (11, 'U');");
    sqlite3_close(db);

    const auto names = discoverStimulusNames(db_path.string());
    ASSERT_EQ(names.size(), 3U);
    EXPECT_EQ(names.at(5), "Atras");
    EXPECT_EQ(names.at(11), "U");
}

TEST_F(SubjectDiscoveryTest, StimulusNamesRefuseNullNameAndNonSqliteRoot)
{
    const auto db_path = tmp_root_ / "dataset.sqlite";
    sqlite3* db = createSqliteDb(db_path);
    execOrThrow(db, "CREATE TABLE stimulus(id INTEGER PRIMARY KEY, name TEXT);");
    execOrThrow(db, "INSERT INTO stimulus(id, name) VALUES (1, NULL);");
    sqlite3_close(db);

    EXPECT_THROW((void) discoverStimulusNames(db_path.string()), std::runtime_error);
    EXPECT_THROW((void) discoverStimulusNames(tmp_root_.string()), std::runtime_error);
}
