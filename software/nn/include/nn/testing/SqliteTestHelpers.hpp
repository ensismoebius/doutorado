// Shared test helpers for creating small SQLite protocol DBs for tests.
#ifndef NN_TESTING_SQLITE_TEST_HELPERS_HPP
#define NN_TESTING_SQLITE_TEST_HELPERS_HPP

#include <sqlite3.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nn::testing
{
inline std::string make_temp_db_path_unique(const std::string& prefix = "nn_test_db")
{
    auto tmp = std::filesystem::temp_directory_path();
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    auto pid = static_cast<long>(getpid());
    std::string name = prefix + "_" + std::to_string(pid) + "_" + std::to_string(now);
    tmp /= name;
    std::filesystem::create_directories(tmp);
    return tmp.string();
}

inline void create_simple_protocol_db(
    const std::string& db_root, std::size_t eeg_len, std::size_t audio_len)
{
    const std::string db_path = (std::filesystem::path(db_root) / "database.sqlite").string();
    sqlite3* db = nullptr;
    if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) throw std::runtime_error("sqlite open");

    const char* create_trial = "CREATE TABLE IF NOT EXISTS trial (id INTEGER PRIMARY KEY);";
    sqlite3_exec(db, create_trial, nullptr, nullptr, nullptr);

    const char* create_eeg =
        "CREATE TABLE IF NOT EXISTS eeg_samples (id INTEGER PRIMARY KEY AUTOINCREMENT, trial_id "
        "INTEGER, F3 BLOB, F4 BLOB, C3 BLOB, C4 BLOB, P3 BLOB, P4 BLOB, blink BLOB);";
    sqlite3_exec(db, create_eeg, nullptr, nullptr, nullptr);

    const char* create_audio =
        "CREATE TABLE IF NOT EXISTS audio_samples (id INTEGER PRIMARY KEY AUTOINCREMENT, trial_id "
        "INTEGER, samples BLOB, audio_row INTEGER);";
    sqlite3_exec(db, create_audio, nullptr, nullptr, nullptr);

    sqlite3_exec(db, "INSERT INTO trial (id) VALUES (1);", nullptr, nullptr, nullptr);

    const char* eeg_sql =
        "INSERT INTO eeg_samples (trial_id, F3, F4, C3, C4, P3, P4, blink) VALUES (1, "
        "?,?,?,?,?,?,?);";
    sqlite3_stmt* eeg_stmt = nullptr;
    if (sqlite3_prepare_v2(db, eeg_sql, -1, &eeg_stmt, nullptr) != SQLITE_OK)
        throw std::runtime_error("prepare eeg");

    std::vector<float> channel(eeg_len);
    for (std::size_t i = 0; i < eeg_len; ++i) channel[i] = static_cast<float>(i);

    for (int col = 1; col <= 7; ++col)
    {
        sqlite3_bind_blob(eeg_stmt,
            col,
            channel.data(),
            static_cast<int>(channel.size() * sizeof(float)),
            SQLITE_TRANSIENT);
    }
    sqlite3_step(eeg_stmt);
    sqlite3_finalize(eeg_stmt);

    const char* audio_sql =
        "INSERT INTO audio_samples (trial_id, samples, audio_row) VALUES (1, ?, 0);";
    sqlite3_stmt* aud_stmt = nullptr;
    if (sqlite3_prepare_v2(db, audio_sql, -1, &aud_stmt, nullptr) != SQLITE_OK)
        throw std::runtime_error("prepare audio");
    std::vector<float> audio(audio_len);
    for (std::size_t i = 0; i < audio_len; ++i) audio[i] = static_cast<float>(i);
    sqlite3_bind_blob(aud_stmt,
        1,
        audio.data(),
        static_cast<int>(audio.size() * sizeof(float)),
        SQLITE_TRANSIENT);
    sqlite3_step(aud_stmt);
    sqlite3_finalize(aud_stmt);

    sqlite3_close(db);
}

// Create a mock imagined-speech sqlite DB used by tests that need subjects,
// trials, EEG and audio blobs. Returns the path to the created DB and sets
// `out_subject_id` to the first subject id. The caller must remove the file
// when finished. `audio_n` and `eeg_n` are the number of samples per-row
// (audio) and per-channel (eeg) respectively.
inline std::string create_mock_imagined_db(
    int& out_subject_id, std::size_t audio_n, std::size_t eeg_n)
{
    namespace fs = std::filesystem;
    const auto tmp = fs::temp_directory_path();
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
    rc = sqlite3_step(ins);
    sqlite3_finalize(ins);

    rc = sqlite3_prepare_v2(db, "INSERT INTO subject(name) VALUES(?)", -1, &ins, nullptr);
    if (rc != SQLITE_OK)
    {
        sqlite3_close(db);
        throw std::runtime_error("prepare failed");
    }
    rc = sqlite3_bind_text(ins, 1, "subj_2", -1, SQLITE_STATIC);
    rc = sqlite3_step(ins);
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

    // prepare inserts for trial
    rc = sqlite3_prepare_v2(
        db, "INSERT INTO trial(subject_id, original_row) VALUES(?, ?)", -1, &ins, nullptr);
    if (rc != SQLITE_OK)
    {
        sqlite3_close(db);
        throw std::runtime_error("prepare failed");
    }

    // Trial A: audio-only (original_row NULL)
    rc = sqlite3_bind_int(ins, 1, out_subject_id);
    rc = sqlite3_bind_null(ins, 2);
    rc = sqlite3_step(ins);
    sqlite3_reset(ins);

    // Trial B: eeg-only (original_row = 0)
    rc = sqlite3_bind_int(ins, 1, out_subject_id);
    rc = sqlite3_bind_int(ins, 2, 0);
    rc = sqlite3_step(ins);
    sqlite3_reset(ins);

    // Trial C: both (original_row = 1)
    rc = sqlite3_bind_int(ins, 1, out_subject_id);
    rc = sqlite3_bind_int(ins, 2, 1);
    rc = sqlite3_step(ins);
    sqlite3_finalize(ins);

    // fetch trial ids
    rc = sqlite3_prepare_v2(
        db, "SELECT id, original_row FROM trial WHERE subject_id = ? ORDER BY id", -1, &q, nullptr);
    if (rc != SQLITE_OK)
    {
        sqlite3_close(db);
        throw std::runtime_error("prepare failed");
    }
    rc = sqlite3_bind_int(q, 1, out_subject_id);
    std::vector<int> trial_ids;
    while ((rc = sqlite3_step(q)) == SQLITE_ROW)
    {
        trial_ids.push_back(sqlite3_column_int(q, 0));
    }
    sqlite3_finalize(q);

    // insert audio_samples for trial A and trial C
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

    std::vector<double> audio_buf(audio_n);
    for (std::size_t i = 0; i < audio_n; ++i) audio_buf[i] = static_cast<double>(i) * 0.001;

    // Trial A -> audio_row 10
    rc = sqlite3_bind_int(ins, 1, trial_ids[0]);
    rc = sqlite3_bind_int(ins, 2, 10);
    rc = sqlite3_bind_blob(
        ins, 3, audio_buf.data(), static_cast<int>(audio_n * sizeof(double)), SQLITE_TRANSIENT);
    rc = sqlite3_step(ins);
    sqlite3_reset(ins);

    // Trial C -> audio_row 11
    rc = sqlite3_bind_int(ins, 1, trial_ids[2]);
    rc = sqlite3_bind_int(ins, 2, 11);
    rc = sqlite3_bind_blob(
        ins, 3, audio_buf.data(), static_cast<int>(audio_n * sizeof(double)), SQLITE_TRANSIENT);
    rc = sqlite3_step(ins);
    sqlite3_finalize(ins);

    // insert eeg_samples for trial B and C (one row each)
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

    std::vector<double> chbuf(eeg_n);
    for (std::size_t i = 0; i < eeg_n; ++i) chbuf[i] = static_cast<double>(i) * 0.0001;

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
    sqlite3_finalize(ins);

    sqlite3_close(db);
    return db_path;
}

} // namespace nn::testing

#endif // NN_TESTING_SQLITE_TEST_HELPERS_HPP
