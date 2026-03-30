#include <gtest/gtest.h>
#include <sqlite3.h>

#include <filesystem>

#include "nn/dataLoaders/IBatchSource.hpp"
#include "nn/dataLoaders/SqliteBatchSource.hpp"
#include "nn/utility/batching.hpp"

using namespace nn::dataLoaders;

static void create_test_db(const std::string& path)
{
    sqlite3* db = nullptr;
    if (sqlite3_open(path.c_str(), &db) != SQLITE_OK) throw std::runtime_error("open db");

    const char* sql = R"SQL(
CREATE TABLE trial(id INTEGER PRIMARY KEY);
CREATE TABLE eeg_samples(id INTEGER PRIMARY KEY, trial_id INTEGER, F3 BLOB, F4 BLOB, C3 BLOB, C4 BLOB, P3 BLOB, P4 BLOB, blink INTEGER);
CREATE TABLE audio_samples(id INTEGER PRIMARY KEY, trial_id INTEGER, samples BLOB, audio_row INTEGER);
INSERT INTO trial(id) VALUES(1);
)SQL";
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK)
    {
        std::string e = err ? err : "exec error";
        sqlite3_free(err);
        sqlite3_close(db);
        throw std::runtime_error(e);
    }

    // Insert short EEG per-channel (2 samples) for 6 channels -> 12 floats
    std::vector<float> eeg_channel = {1.0f, 2.0f};
    std::vector<float> eeg_blob;
    for (int ch = 0; ch < 6; ++ch)
        eeg_blob.insert(eeg_blob.end(), eeg_channel.begin(), eeg_channel.end());

    // Insert audio short (3 samples)
    std::vector<float> audio_blob = {0.1f, 0.2f, 0.3f};

    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db,
        "INSERT INTO eeg_samples(trial_id, F3, F4, C3, C4, P3, P4, blink) VALUES(?,?,?,?,?,?,?,0);",
        -1,
        &st,
        nullptr);
    sqlite3_bind_int(st, 1, 1);
    for (int c = 0; c < 6; ++c)
    {
        sqlite3_bind_blob(st,
            2 + c,
            eeg_channel.data(),
            static_cast<int>(eeg_channel.size() * sizeof(float)),
            SQLITE_TRANSIENT);
    }
    if (sqlite3_step(st) != SQLITE_DONE)
    {
        sqlite3_finalize(st);
        sqlite3_close(db);
        throw std::runtime_error("insert eeg");
    }
    sqlite3_finalize(st);

    sqlite3_prepare_v2(db,
        "INSERT INTO audio_samples(trial_id, samples, audio_row) VALUES(?,?,0);",
        -1,
        &st,
        nullptr);
    sqlite3_bind_int(st, 1, 1);
    sqlite3_bind_blob(st,
        2,
        audio_blob.data(),
        static_cast<int>(audio_blob.size() * sizeof(float)),
        SQLITE_TRANSIENT);
    if (sqlite3_step(st) != SQLITE_DONE)
    {
        sqlite3_finalize(st);
        sqlite3_close(db);
        throw std::runtime_error("insert audio");
    }
    sqlite3_finalize(st);

    sqlite3_close(db);
}

TEST(SqliteBatchSourceWindowing, PaddingRepeatLastSample)
{
    const std::string tmpdir = std::filesystem::temp_directory_path() / "sqlite_test_db";
    std::filesystem::create_directories(tmpdir);
    const std::string dbpath = tmpdir + "/database.sqlite";
    create_test_db(dbpath);

    // Sanity-check the DB we just created: ensure tables exist and have rows.
    {
        sqlite3* checkdb = nullptr;
        ASSERT_EQ(SQLITE_OK, sqlite3_open(dbpath.c_str(), &checkdb));
        sqlite3_stmt* st = nullptr;
        sqlite3_prepare_v2(checkdb, "SELECT COUNT(*) FROM trial;", -1, &st, nullptr);
        int rc = sqlite3_step(st);
        ASSERT_EQ(rc, SQLITE_ROW);
        int trials = sqlite3_column_int(st, 0);
        sqlite3_finalize(st);
        // Check join count
        sqlite3_prepare_v2(checkdb,
            "SELECT COUNT(DISTINCT t.id) FROM trial t INNER JOIN audio_samples a ON a.trial_id = "
            "t.id INNER JOIN eeg_samples e ON e.trial_id = t.id;",
            -1,
            &st,
            nullptr);
        rc = sqlite3_step(st);
        ASSERT_EQ(rc, SQLITE_ROW);
        int joined = sqlite3_column_int(st, 0);
        sqlite3_finalize(st);
        sqlite3_close(checkdb);
        ASSERT_GT(trials, 0);
        ASSERT_GT(joined, 0);
    }

    // underlying source is null (we expect SqliteBatchSource to produce batches)
    std::unique_ptr<IBatchSource> underlying = nullptr;
    nn::windowing::WindowSpec eeg_win{.window_size = 4, .overlap = 0.5f, .sample_rate = 1024};
    nn::windowing::WindowSpec audio_win{.window_size = 4, .overlap = 0.5f, .sample_rate = 44100};

    SqliteBatchSource src(tmpdir,
        std::move(underlying),
        2,
        nn::dataLoaders::SqliteDatasetType::FusedWindow,
        eeg_win,
        audio_win,
        Protocol101117InputMode::Concatenated);

    Batch b;
    bool ok = src.next(b);
    EXPECT_TRUE(ok);
    EXPECT_EQ(b.inputs.rows(), 2);
    EXPECT_EQ(b.targets.rows(), 2);
    EXPECT_GT(b.inputs.cols(), 0);
    EXPECT_GT(b.targets.cols(), 0);

    // Clean up
    std::filesystem::remove(dbpath);
    std::filesystem::remove(tmpdir);
}
