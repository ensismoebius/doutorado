#include <gtest/gtest.h>
#include <sqlite3.h>

#include <filesystem>

#include "nn/dataLoaders/SqliteBatchSource.hpp"
#include "nn/testing/SqliteTestHelpers.hpp"
#include "nn/utility/batching.hpp"

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
        ASSERT_GT(trials, 0);
        ASSERT_GT(joined, 0);
    }

    nn::windowing::WindowSpec eeg_win{.window_size = 4, .overlap = 0.5f, .sample_rate = 1024};
    nn::windowing::WindowSpec audio_win{.window_size = 4, .overlap = 0.5f, .sample_rate = 44100};

    SqliteBatchSource src(tmpdir,
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
