/**
 * @file src/core/dataLoaders/10.1117/tests/loaders/sqlite_session_gtest.cpp
 * @brief Implementation for Sqlite session gtest.
 *

 */

// sqlite_session_gtest.cpp
// Verify SQL-backed session reads match raw sqlite blobs for a few trials.

#include <gtest/gtest.h>
#include <sqlite3.h>

#include <filesystem>
#include <string>
#include <vector>

#include "dataLoaders/10.1117/loaders/AudioLoader.h"
#include "dataLoaders/10.1117/loaders/EEGLoader.h"
#include "dataLoaders/10.1117/schema/METADATA.hpp"
#include "testing/SqliteTestHelpers.hpp"

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
    AudioMatSession session(db_path, subject_id);

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

    EEGMatSession session(db_path, subject_id);

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
