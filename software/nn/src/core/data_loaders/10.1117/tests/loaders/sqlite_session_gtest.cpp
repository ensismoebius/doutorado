/**
 * @file src/core/data_loaders/10.1117/tests/loaders/sqlite_session_gtest.cpp
 * @brief Implementation for Sqlite session gtest.
 *

 */

// sqlite_session_gtest.cpp
// Verify SQL-backed session reads match raw sqlite blobs for a few trials.

#include <gtest/gtest.h>
#include <sqlite3.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "data_loaders/10.1117/loaders/AudioLoader.hpp"
#include "data_loaders/10.1117/loaders/EEGLoader.hpp"
#include "data_loaders/10.1117/schema/Metadata.hpp"
#include "data_loaders/10.1117/schema/SqliteDataset.hpp"
#include "test_utils/SqliteTestHelpers.hpp"

using namespace nn::dataLoaders;

// Helper: create a temporary mock sqlite DB with two subjects and example
// trials: audio-only, eeg-only, and both. Returns file path and first subject id.
// Use shared helpers in `nn::testing` instead of inline DB creation.

TEST(SqliteSession, AudioSessionMatchesBlobs)
{
    int subject_id = -1;
    std::string db_path = nn::testing::create_mock_imagined_db(subject_id,
        ImaginedSpeechSchema_10_1117.audioSamples(),
        ImaginedSpeechSchema_10_1117.eegSamplesPerChannel());
    // open for direct checks
    sqlite3* db = nullptr;
    ASSERT_EQ(SQLITE_OK, sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr));

    // instantiate session for the subject
    AudioSession session(db_path, subject_id);

    // get up to 2 audio rows for this subject
    sqlite3_stmt* q = nullptr;
    ASSERT_EQ(SQLITE_OK,
        sqlite3_prepare_v2(db,
            "SELECT a.audio_row, a.samples FROM audio_samples a JOIN trial t ON a.trial_id = t.id "
            "WHERE t.subject_id = ? LIMIT 2",
            -1,
            &q,
            nullptr));
    ASSERT_EQ(SQLITE_OK, sqlite3_bind_int(q, 1, subject_id));

    while (sqlite3_step(q) == SQLITE_ROW)
    {
        int audio_row = sqlite3_column_int(q, 0);
        const void* blob = sqlite3_column_blob(q, 1);
        int bytes = sqlite3_column_bytes(q, 1);
        const size_t expected_bytes = ImaginedSpeechSchema_10_1117.audioSamples() * sizeof(double);
        ASSERT_EQ(static_cast<size_t>(bytes), expected_bytes);

        // read via session
        auto tup = session.readRow(static_cast<size_t>(audio_row));
        const nn::Tensor& audio_tensor = std::get<0>(tup);
        ASSERT_EQ(audio_tensor.rows(),
            static_cast<nn::Index>(ImaginedSpeechSchema_10_1117.audioSamples()));

        // copy blob locally before any sqlite finalize
        std::vector<double> src(audio_tensor.rows());
        memcpy(src.data(), blob, bytes);
        const float* dst = audio_tensor.data_ptr();
        // compare first 64 samples and last 64 samples
        const size_t check_n = 64;
        for (size_t i = 0; i < check_n; ++i)
        {
            ASSERT_NEAR(static_cast<double>(dst[i]), src[i], 1e-5);
        }
        const size_t n = ImaginedSpeechSchema_10_1117.audioSamples();
        for (size_t i = 0; i < check_n; ++i)
        {
            ASSERT_NEAR(static_cast<double>(dst[n - 1 - i]), src[n - 1 - i], 1e-5);
        }
    }
    sqlite3_finalize(q);
    sqlite3_close(db);
    sqlite3_shutdown();
    std::filesystem::remove(db_path);
}

TEST(SqliteSession, EEGSessionMatchesBlobs)
{
    int subject_id = -1;
    std::string db_path = nn::testing::create_mock_imagined_db(subject_id,
        ImaginedSpeechSchema_10_1117.audioSamples(),
        ImaginedSpeechSchema_10_1117.eegSamplesPerChannel());
    sqlite3* db = nullptr;
    ASSERT_EQ(SQLITE_OK, sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr));

    EEGSession session(db_path, subject_id);

    // select up to 2 trials that have original_row
    sqlite3_stmt* q = nullptr;
    ASSERT_EQ(SQLITE_OK,
        sqlite3_prepare_v2(db,
            "SELECT id, original_row FROM trial WHERE subject_id = ? AND original_row IS NOT NULL "
            "LIMIT 2",
            -1,
            &q,
            nullptr));
    ASSERT_EQ(SQLITE_OK, sqlite3_bind_int(q, 1, subject_id));

    while (sqlite3_step(q) == SQLITE_ROW)
    {
        int trial_id = sqlite3_column_int(q, 0);
        int original_row = sqlite3_column_int(q, 1);

        // get eeg_samples blobs for this trial
        sqlite3_stmt* e = nullptr;
        ASSERT_EQ(SQLITE_OK,
            sqlite3_prepare_v2(db,
                "SELECT F3, F4, C3, C4, P3, P4, blink FROM eeg_samples WHERE trial_id = ? LIMIT 1",
                -1,
                &e,
                nullptr));
        ASSERT_EQ(SQLITE_OK, sqlite3_bind_int(e, 1, trial_id));
        ASSERT_EQ(SQLITE_ROW, sqlite3_step(e));

        std::vector<std::vector<double>> src_ch_data(6);
        for (int ch = 0; ch < 6; ++ch)
        {
            const void* blob_ptr = sqlite3_column_blob(e, ch);
            int b = sqlite3_column_bytes(e, ch);
            const size_t expected =
                ImaginedSpeechSchema_10_1117.eegSamplesPerChannel() * sizeof(double);
            ASSERT_EQ(static_cast<size_t>(b), expected);
            const double* d = reinterpret_cast<const double*>(blob_ptr);
            src_ch_data[ch].assign(d, d + ImaginedSpeechSchema_10_1117.eegSamplesPerChannel());
        }
        int blink = sqlite3_column_int(e, 6);
        sqlite3_finalize(e);

        // read via session using original_row
        auto tup = session.readRow(static_cast<size_t>(original_row));
        const nn::Tensor& eeg_tensor = std::get<0>(tup);
        const auto labels = std::get<1>(tup);
        ASSERT_EQ(labels[2], blink);

        // compare first 32 samples of each channel
        const size_t check_n = 32;
        for (int ch = 0; ch < 6; ++ch)
        {
            const auto& src = src_ch_data[ch];
            for (size_t i = 0; i < check_n; ++i)
            {
                ASSERT_NEAR(static_cast<double>(eeg_tensor.at(ch, i)), src[i], 1e-6);
            }
        }
    }
    sqlite3_finalize(q);
    sqlite3_close(db);
    sqlite3_shutdown();
    std::filesystem::remove(db_path);
}

// ── openSqliteDatasetReadOnly: the immutable open used by every 10.1117 SQLite reader ──────────
// Why immutable, and why a non-empty -wal is refused: SqliteDataset.hpp.

namespace
{
// A WAL-mode database with one table and `rows` rows, written through `writer` (left open by
// the caller when it wants the transactions to stay in the -wal file).
void write_wal_db(sqlite3*& writer, const std::string& path, int rows)
{
    ASSERT_EQ(SQLITE_OK, sqlite3_open(path.c_str(), &writer));
    ASSERT_EQ(
        SQLITE_OK, sqlite3_exec(writer, "PRAGMA journal_mode=WAL;", nullptr, nullptr, nullptr));
    ASSERT_EQ(
        SQLITE_OK, sqlite3_exec(writer, "PRAGMA wal_autocheckpoint=0;", nullptr, nullptr, nullptr));
    ASSERT_EQ(
        SQLITE_OK, sqlite3_exec(writer, "CREATE TABLE t(x INTEGER);", nullptr, nullptr, nullptr));
    for (int i = 0; i < rows; ++i)
        ASSERT_EQ(SQLITE_OK,
            sqlite3_exec(writer, "INSERT INTO t VALUES (1);", nullptr, nullptr, nullptr));
}

auto count_rows(sqlite3* db) -> int
{
    sqlite3_stmt* q = nullptr;
    EXPECT_EQ(SQLITE_OK, sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM t", -1, &q, nullptr));
    EXPECT_TRUE(sqliteStepRow(db, q, "count"));
    const int n = sqlite3_column_int(q, 0);
    sqlite3_finalize(q);
    return n;
}
} // namespace

TEST(SqliteDatasetOpen, RefusesTransactionsStillInTheWriteAheadLog)
{
    const auto dir = std::filesystem::temp_directory_path() / "nn_sqlite_dataset_wal";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "db.sqlite").string();

    sqlite3* writer = nullptr;
    write_wal_db(writer, path, 3);
    ASSERT_GT(std::filesystem::file_size(path + "-wal"), 0U);
    try
    {
        sqlite3* db = openSqliteDatasetReadOnly(path);
        sqlite3_close(db);
        ADD_FAILURE() << "a non-empty -wal was opened immutable; its 3 rows would be invisible";
    }
    catch (const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("wal_checkpoint"), std::string::npos) << e.what();
    }

    // Closing the last connection checkpoints the log into the file: now it opens, and the
    // rows are there.
    ASSERT_EQ(SQLITE_OK, sqlite3_close(writer));
    sqlite3* db = openSqliteDatasetReadOnly(path);
    EXPECT_EQ(3, count_rows(db));
    sqlite3_close(db);
    std::filesystem::remove_all(dir);
}

TEST(SqliteDatasetOpen, FileNameWithUriCharactersIsReadLiterally)
{
    // '?', '#' and '%' mean something inside a file: URI; the name must still be this file.
    const auto dir = std::filesystem::temp_directory_path() / "nn sqlite?dataset#50%";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "db.sqlite").string();
    sqlite3* writer = nullptr;
    write_wal_db(writer, path, 2);
    ASSERT_EQ(SQLITE_OK, sqlite3_close(writer));

    sqlite3* db = openSqliteDatasetReadOnly(path);
    EXPECT_EQ(2, count_rows(db));
    sqlite3_close(db);
    std::filesystem::remove_all(dir);
}

TEST(SqliteDatasetOpen, MissingFileThrowsNamingIt)
{
    const std::string path = "/nonexistent_dir_for_nn_tests/db.sqlite";
    try
    {
        sqlite3_close(openSqliteDatasetReadOnly(path));
        ADD_FAILURE() << "a missing dataset file was opened";
    }
    catch (const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find(path), std::string::npos) << e.what();
    }
}

TEST(SqliteDatasetOpen, StepErrorIsAnExceptionNotTheEndOfTheRows)
{
    // A write on an immutable (read-only) handle fails in sqlite3_step. A `== SQLITE_ROW` loop
    // would read that failure as "no more rows".
    const auto dir = std::filesystem::temp_directory_path() / "nn_sqlite_dataset_step";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "db.sqlite").string();
    sqlite3* writer = nullptr;
    write_wal_db(writer, path, 1);
    ASSERT_EQ(SQLITE_OK, sqlite3_close(writer));

    sqlite3* db = openSqliteDatasetReadOnly(path);
    sqlite3_stmt* q = nullptr;
    ASSERT_EQ(SQLITE_OK, sqlite3_prepare_v2(db, "INSERT INTO t VALUES (2)", -1, &q, nullptr));
    EXPECT_THROW(sqliteStepRow(db, q, "insert"), std::runtime_error);
    sqlite3_finalize(q);
    sqlite3_close(db);
    std::filesystem::remove_all(dir);
}

TEST(SqliteDatasetOpen, SqliteSessionOnAMissingDatabaseThrowsInsteadOfTryingMat)
{
    // A .sqlite path used to fall through to the MAT reader when it could not be opened.
    EXPECT_THROW(AudioSession("/nonexistent_dir_for_nn_tests/db.sqlite", 1), std::runtime_error);
    EXPECT_THROW(EEGSession("/nonexistent_dir_for_nn_tests/db.sqlite", 1), std::runtime_error);
    try
    {
        AudioSession s("/nonexistent_dir_for_nn_tests/db.sqlite", 1);
    }
    catch (const std::runtime_error& e)
    {
        EXPECT_NE(std::string(e.what()).find("SQLite dataset"), std::string::npos) << e.what();
    }
}
