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

TEST_F(SubjectDiscoveryTest, SqliteAllowsNullSubjectNameAndMissingCountTables)
{
    const auto db_path = tmp_root_ / "dataset.sqlite";
    sqlite3* db = createSqliteDb(db_path);
    execOrThrow(db, "CREATE TABLE subject(id INTEGER PRIMARY KEY, subject_name TEXT);");
    execOrThrow(db, "INSERT INTO subject(id, subject_name) VALUES (42, NULL);");
    sqlite3_close(db);

    const auto subjects = discoverSubjects(db_path.string(), "^S(\\d+)$");

    ASSERT_EQ(subjects.size(), 1U);
    EXPECT_EQ(subjects[0].subject_id, 42);
    EXPECT_EQ(subjects[0].subject_name, "42");
    EXPECT_EQ(subjects[0].eeg_path, db_path.string());
    EXPECT_EQ(subjects[0].audio_path, db_path.string());
    EXPECT_EQ(subjects[0].eeg_rows, 0U);
    EXPECT_EQ(subjects[0].audio_rows, 0U);
}
