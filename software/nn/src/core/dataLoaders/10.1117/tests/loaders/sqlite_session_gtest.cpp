// sqlite_session_gtest.cpp
// Verify SQL-backed session reads match raw sqlite blobs for a few trials.

#include <gtest/gtest.h>
#include <sqlite3.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "nn/dataLoaders/10.1117/loaders/AudioLoader.h"
#include "nn/dataLoaders/10.1117/loaders/EEGLoader.h"
#include "nn/dataLoaders/10.1117/schema/METADATA.hpp"

using namespace nn::dataLoaders;

// Helper: create a temporary mock sqlite DB with two subjects and example
// trials: audio-only, eeg-only, and both. Returns file path and first subject id.
static std::string create_mock_db(int& out_subject_id)
{
    namespace fs = std::filesystem;
    const auto tmp = fs::temp_directory_path();
    // create unique filename using pid+timestamp
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::string uid = std::to_string(getpid()) + "_" + std::to_string(now);
    std::string db_filename = "mock_imagined_" + uid + ".sqlite";
    std::string db_path = (tmp / fs::path(db_filename)).string();
    // create empty file
    std::ofstream ofs(db_path);
    if (!ofs) throw std::runtime_error("failed to create temp db file");
    ofs.close();

    sqlite3* db = nullptr;
    int rc =
        sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
    if (rc != SQLITE_OK)
    {
        throw std::runtime_error("sqlite3_open_v2 failed: " + std::to_string(rc));
    }

    const char* schema = R"SQL(
CREATE TABLE subject(id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE trial(id INTEGER PRIMARY KEY, subject_id INTEGER, original_row INTEGER, modality_id INTEGER, stimulus_id INTEGER);
CREATE TABLE audio_samples(id INTEGER PRIMARY KEY, trial_id INTEGER, audio_row INTEGER, samples BLOB);
CREATE TABLE eeg_samples(id INTEGER PRIMARY KEY, trial_id INTEGER, F3 BLOB, F4 BLOB, C3 BLOB, C4 BLOB, P3 BLOB, P4 BLOB, blink INTEGER);
)SQL";
    char* errmsg = nullptr;
    rc = sqlite3_exec(db, schema, nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK)
    {
        std::string msg = errmsg ? errmsg : "sqlite3_exec failed";
        if (errmsg) sqlite3_free(errmsg);
        sqlite3_close(db);
        throw std::runtime_error(msg);
    }

    // insert two subjects
    sqlite3_stmt* ins = nullptr;
    rc = sqlite3_prepare_v2(db, "INSERT INTO subject(name) VALUES(?)", -1, &ins, nullptr);
    if (rc != SQLITE_OK)
    {
        sqlite3_close(db);
        throw std::runtime_error("prepare failed");
    }
    rc = sqlite3_bind_text(ins, 1, "subj_1", -1, SQLITE_STATIC);
    if (rc != SQLITE_OK)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("bind failed");
    }
    rc = sqlite3_step(ins);
    if (rc != SQLITE_DONE)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("step failed");
    }
    sqlite3_finalize(ins);

    rc = sqlite3_prepare_v2(db, "INSERT INTO subject(name) VALUES(?)", -1, &ins, nullptr);
    if (rc != SQLITE_OK)
    {
        sqlite3_close(db);
        throw std::runtime_error("prepare failed");
    }
    rc = sqlite3_bind_text(ins, 1, "subj_2", -1, SQLITE_STATIC);
    if (rc != SQLITE_OK)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("bind failed");
    }
    rc = sqlite3_step(ins);
    if (rc != SQLITE_DONE)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("step failed");
    }
    sqlite3_finalize(ins);

    // get first subject id
    sqlite3_stmt* q = nullptr;
    rc = sqlite3_prepare_v2(db, "SELECT id FROM subject ORDER BY id LIMIT 1", -1, &q, nullptr);
    if (rc != SQLITE_OK)
    {
        sqlite3_close(db);
        throw std::runtime_error("prepare failed");
    }
    rc = sqlite3_step(q);
    if (rc != SQLITE_ROW)
    {
        sqlite3_finalize(q);
        sqlite3_close(db);
        throw std::runtime_error("no subject row");
    }
    out_subject_id = sqlite3_column_int(q, 0);
    sqlite3_finalize(q);

    // Prepare inserts for trial, audio_samples, eeg_samples
    rc = sqlite3_prepare_v2(
        db, "INSERT INTO trial(subject_id, original_row) VALUES(?, ?)", -1, &ins, nullptr);
    if (rc != SQLITE_OK)
    {
        sqlite3_close(db);
        throw std::runtime_error("prepare failed");
    }

    // Trial A: audio-only (original_row NULL)
    rc = sqlite3_bind_int(ins, 1, out_subject_id);
    if (rc != SQLITE_OK)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("bind failed");
    }
    rc = sqlite3_bind_null(ins, 2);
    if (rc != SQLITE_OK)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("bind failed");
    }
    rc = sqlite3_step(ins);
    if (rc != SQLITE_DONE)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("step failed");
    }
    sqlite3_reset(ins);

    // Trial B: eeg-only (original_row = 0)
    rc = sqlite3_bind_int(ins, 1, out_subject_id);
    if (rc != SQLITE_OK)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("bind failed");
    }
    rc = sqlite3_bind_int(ins, 2, 0);
    if (rc != SQLITE_OK)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("bind failed");
    }
    rc = sqlite3_step(ins);
    if (rc != SQLITE_DONE)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("step failed");
    }
    sqlite3_reset(ins);

    // Trial C: both (original_row = 1)
    rc = sqlite3_bind_int(ins, 1, out_subject_id);
    if (rc != SQLITE_OK)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("bind failed");
    }
    rc = sqlite3_bind_int(ins, 2, 1);
    if (rc != SQLITE_OK)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("bind failed");
    }
    rc = sqlite3_step(ins);
    if (rc != SQLITE_DONE)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("step failed");
    }
    sqlite3_finalize(ins);

    // Fetch trial ids
    rc = sqlite3_prepare_v2(
        db, "SELECT id, original_row FROM trial WHERE subject_id = ? ORDER BY id", -1, &q, nullptr);
    if (rc != SQLITE_OK)
    {
        sqlite3_close(db);
        throw std::runtime_error("prepare failed");
    }
    rc = sqlite3_bind_int(q, 1, out_subject_id);
    if (rc != SQLITE_OK)
    {
        sqlite3_finalize(q);
        sqlite3_close(db);
        throw std::runtime_error("bind failed");
    }
    std::vector<int> trial_ids;
    while ((rc = sqlite3_step(q)) == SQLITE_ROW)
    {
        trial_ids.push_back(sqlite3_column_int(q, 0));
    }
    sqlite3_finalize(q);

    // Insert audio_samples for trial A and trial C
    rc = sqlite3_prepare_v2(db,
        "INSERT INTO audio_samples(trial_id, audio_row, samples) VALUES(?, ?, ?)",
        -1,
        &ins,
        nullptr);
    if (rc != SQLITE_OK)
    {
        sqlite3_close(db);
        throw std::runtime_error("prepare failed");
    }
    // create audio blob (full length)
    const size_t audio_n = ImaginedSpeechSchema_10_1117.audioSamples();
    std::vector<double> audio_buf(audio_n);
    for (size_t i = 0; i < audio_n; ++i) audio_buf[i] = static_cast<double>(i) * 0.001;

    // Trial A -> audio_row 10
    rc = sqlite3_bind_int(ins, 1, trial_ids[0]);
    rc = sqlite3_bind_int(ins, 2, 10);
    rc = sqlite3_bind_blob(
        ins, 3, audio_buf.data(), static_cast<int>(audio_n * sizeof(double)), SQLITE_TRANSIENT);
    rc = sqlite3_step(ins);
    if (rc != SQLITE_DONE)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("step failed");
    }
    sqlite3_reset(ins);

    // Trial C -> audio_row 11
    rc = sqlite3_bind_int(ins, 1, trial_ids[2]);
    rc = sqlite3_bind_int(ins, 2, 11);
    rc = sqlite3_bind_blob(
        ins, 3, audio_buf.data(), static_cast<int>(audio_n * sizeof(double)), SQLITE_TRANSIENT);
    rc = sqlite3_step(ins);
    if (rc != SQLITE_DONE)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("step failed");
    }
    sqlite3_finalize(ins);

    // Insert eeg_samples for trial B and C (one row each)
    rc = sqlite3_prepare_v2(db,
        "INSERT INTO eeg_samples(trial_id, F3, F4, C3, C4, P3, P4, blink) VALUES(?, ?, ?, ?, ?, ?, "
        "?, ?)",
        -1,
        &ins,
        nullptr);
    if (rc != SQLITE_OK)
    {
        sqlite3_close(db);
        throw std::runtime_error("prepare failed");
    }
    const size_t eeg_n = ImaginedSpeechSchema_10_1117.eegSamplesPerChannel();
    std::vector<double> chbuf(eeg_n);
    for (size_t i = 0; i < eeg_n; ++i) chbuf[i] = static_cast<double>(i) * 0.0001;

    // Trial B (eeg-only)
    rc = sqlite3_bind_int(ins, 1, trial_ids[1]);
    for (int c = 0; c < 6; ++c)
    {
        rc = sqlite3_bind_blob(
            ins, 2 + c, chbuf.data(), static_cast<int>(eeg_n * sizeof(double)), SQLITE_TRANSIENT);
        if (rc != SQLITE_OK)
        {
            sqlite3_finalize(ins);
            sqlite3_close(db);
            throw std::runtime_error("bind_blob failed");
        }
    }
    rc = sqlite3_bind_int(ins, 8, 0);
    rc = sqlite3_step(ins);
    if (rc != SQLITE_DONE)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("step failed");
    }
    sqlite3_reset(ins);

    // Trial C (both)
    rc = sqlite3_bind_int(ins, 1, trial_ids[2]);
    for (int c = 0; c < 6; ++c)
    {
        rc = sqlite3_bind_blob(
            ins, 2 + c, chbuf.data(), static_cast<int>(eeg_n * sizeof(double)), SQLITE_TRANSIENT);
        if (rc != SQLITE_OK)
        {
            sqlite3_finalize(ins);
            sqlite3_close(db);
            throw std::runtime_error("bind_blob failed");
        }
    }
    rc = sqlite3_bind_int(ins, 8, 1);
    rc = sqlite3_step(ins);
    if (rc != SQLITE_DONE)
    {
        sqlite3_finalize(ins);
        sqlite3_close(db);
        throw std::runtime_error("step failed");
    }
    sqlite3_finalize(ins);

    sqlite3_close(db);
    return db_path;
}

TEST(SqliteSession, AudioSessionMatchesBlobs)
{
    int subject_id = -1;
    std::string db_path = create_mock_db(subject_id);
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
    std::string db_path = create_mock_db(subject_id);
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
