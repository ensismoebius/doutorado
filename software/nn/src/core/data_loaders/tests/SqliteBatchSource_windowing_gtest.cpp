/**
 * @file src/core/data_loaders/tests/SqliteBatchSource_windowing_gtest.cpp
 * @brief Implementation for Sqlitebatchsource windowing gtest.
 *

 */

#include <gtest/gtest.h>
#include <sqlite3.h>

#include <filesystem>

#include "data_loaders/sources/SqliteBatchSource.hpp"
#include "test_utils/SqliteTestHelpers.hpp"
#include "utility/batching.hpp"

using namespace nn::dataLoaders;

// Use shared test helpers to create the small protocol DB for this test.

TEST(SqliteBatchSourceWindowing, PaddingRepeatLastSample)
{
    const std::string tmpdir = nn::testing::make_temp_db_path_unique("sqlite_batch_win_test");
    nn::testing::create_simple_protocol_db(tmpdir, 2, 3);
    const std::string dbpath = (std::filesystem::path(tmpdir) / "database.sqlite").string();

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
        ASSERT_EQ(trials, 1);
        ASSERT_EQ(joined, 1);
    }

    nn::windowing::WindowSpec eeg_win{.window_size = 4, .overlap = 0.5f, .sample_rate = 1024};
    nn::windowing::WindowSpec audio_win{.window_size = 4, .overlap = 0.5f, .sample_rate = 44100};

    {
        SqliteBatchSource src(tmpdir,
            2,
            nn::dataLoaders::SqliteDatasetType::FusedWindow,
            eeg_win,
            audio_win,
            Protocol101117InputMode::Concatenated);

        Batch b; //
        bool ok = src.next(b);
        EXPECT_TRUE(ok);
        EXPECT_EQ(b.inputs.rows(), 1);
        EXPECT_EQ(b.targets.rows(), 1);
        EXPECT_EQ(b.inputs.cols(), 28);
        EXPECT_EQ(b.targets.cols(), 28);
    }

    // Clean up recursively in case SQLite leaves aux files (journal/wal/shm).
    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
    EXPECT_FALSE(std::filesystem::exists(tmpdir));
}

TEST(SqliteBatchSourceWindowing, EmitsAllWindowsAcrossSuccessiveBatches)
{
    const std::string tmpdir =
        nn::testing::make_temp_db_path_unique("sqlite_batch_window_sequence_test");
    nn::testing::create_simple_protocol_db(tmpdir, 8, 8);

    nn::windowing::WindowSpec eeg_win{.window_size = 4, .overlap = 0.5f, .sample_rate = 1024};
    nn::windowing::WindowSpec audio_win{.window_size = 4, .overlap = 0.5f, .sample_rate = 44100};

    SqliteBatchSource src(tmpdir,
        2,
        nn::dataLoaders::SqliteDatasetType::FusedWindow,
        eeg_win,
        audio_win,
        Protocol101117InputMode::Concatenated);

    Batch first_batch;
    ASSERT_TRUE(src.next(first_batch));
    EXPECT_EQ(first_batch.inputs.rows(), 2);
    EXPECT_EQ(first_batch.inputs.cols(), 28);

    Batch second_batch;
    ASSERT_TRUE(src.next(second_batch));
    EXPECT_EQ(second_batch.inputs.rows(), 1);
    EXPECT_EQ(second_batch.inputs.cols(), 28);

    const float first_window_start = first_batch.inputs.at(0, 0);
    const float second_window_start = first_batch.inputs.at(1, 0);
    const float third_window_start = second_batch.inputs.at(0, 0);
    EXPECT_NEAR(first_window_start, 0.0f, 1e-6f);
    EXPECT_NEAR(second_window_start, 2.0f, 1e-6f);
    EXPECT_NEAR(third_window_start, 4.0f, 1e-6f);

    Batch third_batch;
    EXPECT_FALSE(src.next(third_batch));

    std::error_code ec;
    std::filesystem::remove_all(tmpdir, ec);
    EXPECT_FALSE(std::filesystem::exists(tmpdir));
}
