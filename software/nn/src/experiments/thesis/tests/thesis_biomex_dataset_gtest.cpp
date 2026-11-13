// The BIOMEX-DB reader of ThesisDataset (dataset.source "biomex2021"): it reads the paired
// SQLite written by scripts/data/biomex_to_sqlite.py and refuses anything else.
//
// The converter's own tests (scripts/testing/test_biomex_convert.py) check the pairing;
// these check the C++ side of the contract: what the file declares is what the view holds
// (rates, lengths, channel count, phrases), and a file that breaks the declaration fails.

#include <gtest/gtest.h>
#include <sqlite3.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "ThesisDataset.hpp"
#include "ThesisTestConfig.hpp"

namespace
{
namespace fs = std::filesystem;

constexpr int kVoice = 1600; // samples per voice window (a short stand-in for 16000)
constexpr int kChannels = 14;
constexpr int kEpoch = 256;

void exec(sqlite3* db, const std::string& sql)
{
    char* err = nullptr;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK)
    {
        const std::string msg = err ? err : "?";
        sqlite3_free(err);
        throw std::runtime_error(msg + " in " + sql);
    }
}

struct BiomexFile
{
    fs::path path;
    explicit BiomexFile(
        const std::string& name, const std::string& source = "biomex2021", int voice_bytes_cut = 0)
        : path(fs::temp_directory_path() / name)
    {
        fs::remove(path);
        sqlite3* db = nullptr;
        sqlite3_open(path.c_str(), &db);
        exec(db,
            "CREATE TABLE recording (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE speaker (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);"
            "CREATE TABLE stimulus (id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);"
            "CREATE TABLE pair (id INTEGER PRIMARY KEY, speaker_id INTEGER, stimulus_id INTEGER,"
            " session TEXT, series INTEGER, position INTEGER, voice BLOB, eeg BLOB,"
            " eeg_interpolated INTEGER, voice_onset_s REAL, eeg_onset_s REAL);");
        exec(db,
            "INSERT INTO recording VALUES ('format','thesis-paired-v1'),('source','" + source +
                "'),('voice_sample_rate','16000'),('eeg_sample_rate','128'),"
                "('voice_window_samples','" +
                std::to_string(kVoice) +
                "'),('eeg_epoch_samples','256'),"
                "('eeg_channels','AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4');"
                "INSERT INTO speaker VALUES (1,'F001'),(2,'M001');"
                "INSERT INTO stimulus VALUES (1,'uno'),(10,'cero');");
        sqlite3_stmt* st = nullptr;
        sqlite3_prepare_v2(db,
            "INSERT INTO pair (speaker_id, stimulus_id, session, series, position, voice, eeg,"
            " eeg_interpolated, voice_onset_s, eeg_onset_s) VALUES (?,?,'01G10',1,?,?,?,0,0,0)",
            -1,
            &st,
            nullptr);
        int position = 0;
        for (int speaker : {1, 2})
            for (int stim : {1, 10, 1})
            {
                // Every value names its pair: voice = speaker + stim/100, EEG = channel.
                std::vector<float> voice(kVoice, static_cast<float>(speaker) + stim / 100.0f);
                std::vector<float> eeg(kChannels * kEpoch);
                for (int c = 0; c < kChannels; ++c)
                    for (int s = 0; s < kEpoch; ++s) eeg[c * kEpoch + s] = static_cast<float>(c);
                sqlite3_bind_int(st, 1, speaker);
                sqlite3_bind_int(st, 2, stim);
                sqlite3_bind_int(st, 3, ++position);
                sqlite3_bind_blob(st,
                    4,
                    voice.data(),
                    static_cast<int>(voice.size() * sizeof(float)) - voice_bytes_cut,
                    SQLITE_TRANSIENT);
                sqlite3_bind_blob(st,
                    5,
                    eeg.data(),
                    static_cast<int>(eeg.size() * sizeof(float)),
                    SQLITE_TRANSIENT);
                sqlite3_step(st);
                sqlite3_reset(st);
            }
        sqlite3_finalize(st);
        sqlite3_close(db);
    }
    ~BiomexFile()
    {
        std::error_code ec;
        fs::remove(path, ec);
    }
    BiomexFile(const BiomexFile&) = delete;
    BiomexFile& operator=(const BiomexFile&) = delete;
};

thesis::ThesisConfig::Dataset biomex_dataset(const fs::path& root)
{
    thesis::ThesisConfig c = thesis::test::phase00("eeg");
    c.dataset.source = "biomex2021";
    c.dataset.root = root.string();
    return c.dataset;
}

std::string refusal(const thesis::ThesisConfig::Dataset& d)
{
    try
    {
        (void) thesis::load_dataset(d);
    }
    catch (const std::exception& e)
    {
        return e.what();
    }
    return "";
}
} // namespace

TEST(ThesisBiomexDataset, TheViewHoldsWhatTheFileDeclares)
{
    const BiomexFile f("thesis_biomex_ok.sqlite");
    const thesis::ThesisDatasetView v = thesis::load_dataset(biomex_dataset(f.path));
    ASSERT_EQ(v.samples.size(), 6u);
    EXPECT_EQ(v.n_subjects, 2);
    EXPECT_EQ(v.n_stimuli, 2);
    EXPECT_DOUBLE_EQ(v.voice_sample_rate, 16000.0);
    EXPECT_DOUBLE_EQ(v.eeg_sample_rate, 128.0);
    EXPECT_TRUE(v.subject_files.empty());
    for (const auto& s : v.samples)
    {
        EXPECT_EQ(s.audio.rows(), kVoice);
        EXPECT_EQ(s.audio.cols(), 1);
        EXPECT_EQ(s.eeg.rows(), kChannels); // channel-major, as the Coretto view
        EXPECT_EQ(s.eeg.cols(), kEpoch);
        EXPECT_FLOAT_EQ(s.audio.at(0, 0), static_cast<float>(s.subject_id) + s.stimulus / 100.0f);
        EXPECT_FLOAT_EQ(s.eeg.at(13, 255), 13.0f);
        EXPECT_EQ(s.text_phrase, s.stimulus == 10 ? "cero" : "uno");
    }
    // In the file's order: speaker 1's three pairs first.
    EXPECT_EQ(v.samples[0].subject_id, 1);
    EXPECT_EQ(v.samples[3].subject_id, 2);
}

TEST(ThesisBiomexDataset, ATruncatedBlobIsRefused)
{
    const BiomexFile f("thesis_biomex_cut.sqlite", "biomex2021", 4);
    const std::string msg = refusal(biomex_dataset(f.path));
    EXPECT_NE(msg.find("voice of 6396 bytes"), std::string::npos) << msg;
}

TEST(ThesisBiomexDataset, AnotherSourceIsRefused)
{
    const BiomexFile f("thesis_biomex_other.sqlite", "someother");
    const std::string msg = refusal(biomex_dataset(f.path));
    EXPECT_NE(msg.find("source \"someother\""), std::string::npos) << msg;
}

TEST(ThesisBiomexDataset, ACorettoFileIsNotReadAsBiomex)
{
    const fs::path p = fs::temp_directory_path() / "thesis_biomex_coretto_like.sqlite";
    fs::remove(p);
    sqlite3* db = nullptr;
    sqlite3_open(p.c_str(), &db);
    exec(db, "CREATE TABLE trial (id INTEGER PRIMARY KEY);");
    sqlite3_close(db);
    const std::string msg = refusal(biomex_dataset(p));
    fs::remove(p);
    EXPECT_NE(msg.find("recording"), std::string::npos) << msg;
}

TEST(ThesisBiomexDataset, AViewWithoutRatesIsRefusedByItsReaders)
{
    EXPECT_THROW((void) thesis::require_rate(0.0, "voice"), std::invalid_argument);
    EXPECT_DOUBLE_EQ(thesis::require_rate(128.0, "EEG"), 128.0);
}
