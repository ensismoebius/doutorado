// The BIOMEX-DB reader of ThesisDataset (dataset.source "biomex2021"): it reads the paired
// SQLite written by scripts/data/biomex_to_sqlite.py and refuses anything else.
//
// The converter's own tests (scripts/testing/test_biomex_convert.py) check the pairing;
// these check the C++ side of the contract: what the file declares is what the view holds
// (rates, lengths, channel count, phrases), and a file that breaks the declaration fails.

#include <gtest/gtest.h>
#include <sqlite3.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "ThesisDataset.hpp"
#include "ThesisFeatureExtraction.hpp"
#include "ThesisTestConfig.hpp"

namespace
{
namespace fs = std::filesystem;

constexpr int kVoice = 1600; // samples per voice window (a short stand-in for 44100)
constexpr int kChannels = 14;
constexpr int kEpoch = 256; // (a short stand-in for 2048)

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
    explicit BiomexFile(const std::string& name,
        const std::string& source = "biomex2021",
        int voice_bytes_cut = 0,
        const std::string& voice_rate = "44100")
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
                "'),('voice_sample_rate','" + voice_rate +
                "'),('eeg_sample_rate','1024'),"
                "('voice_window_samples','" +
                std::to_string(kVoice) + "'),('eeg_epoch_samples','" + std::to_string(kEpoch) +
                "'),"
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
    EXPECT_DOUBLE_EQ(v.voice_sample_rate, 44100.0);
    EXPECT_DOUBLE_EQ(v.eeg_sample_rate, 1024.0);
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

// The base's native 16 kHz: the same DTWPT level would cut other Hz than on Coretto.
TEST(ThesisBiomexDataset, AFileNotAtTheCorettoRatesIsRefused)
{
    const BiomexFile f("thesis_biomex_native.sqlite", "biomex2021", 0, "16000");
    const std::string msg = refusal(biomex_dataset(f.path));
    EXPECT_NE(msg.find("16000 Hz voice / 1024 Hz EEG"), std::string::npos) << msg;
    EXPECT_NE(msg.find("Coretto rates (44100 / 1024 Hz)"), std::string::npos) << msg;
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

// ── Both bases reach the classifier in one format ──────────────────────────────────────
//
// The classifier sees one row per sample. For the replication to compare like with like,
// the row must have the same columns on both bases for every source. The two bases differ in
// what they record (Coretto: 4 s of voice, 6 EEG channels x 4 s; BIOMEX-DB as converted:
// 1 s of voice, 14 channels x 2 s), not in rate (both 44.1 kHz / 1024 Hz, enforced above).
// Here both shapes go through the frozen Coretto winner (Haar, lfcc, DTWPT level 4, six
// descriptors): 16 sub-bands x 6 = 96 columns for voice, EEG and early fusion, 192 for late
// fusion, all finite. (Scale, gain and length differences inside a base are per-column
// constants that the per-fold z-score removes; they do not change the format.)
namespace
{
thesis::ThesisDatasetView synthetic_view(nn::Index voice_len, nn::Index channels, nn::Index epoch)
{
    thesis::ThesisDatasetView v;
    v.voice_sample_rate = 44100.0;
    v.eeg_sample_rate = 1024.0;
    v.n_subjects = 2;
    v.n_stimuli = 2;
    for (int s = 0; s < 4; ++s)
    {
        thesis::ThesisSample x;
        x.subject_id = 1 + s % 2;
        x.stimulus = 1 + s / 2;
        x.text_phrase = s / 2 == 0 ? "uno" : "dos";
        x.audio = nn::Tensor(voice_len, 1);
        for (nn::Index i = 0; i < voice_len; ++i)
            x.audio.at(i, 0) = thesis::test::synthetic_value(
                static_cast<int>(i), 0.003 + 0.001 * s, static_cast<std::uint32_t>(s));
        x.eeg = nn::Tensor(channels, epoch);
        for (nn::Index c = 0; c < channels; ++c)
            for (nn::Index t = 0; t < epoch; ++t)
                x.eeg.at(c, t) =
                    20.0f * thesis::test::synthetic_value(static_cast<int>(c * epoch + t),
                                0.01 + 0.002 * s,
                                static_cast<std::uint32_t>(100 + s));
        v.samples.push_back(std::move(x));
    }
    return v;
}
} // namespace

TEST(ThesisBiomexDataset, BothBasesReachTheClassifierInTheSameFormat)
{
    const thesis::ThesisDatasetView coretto = synthetic_view(176400, 6, 4096);
    const thesis::ThesisDatasetView biomex = synthetic_view(44100, 14, 2048);
    struct Source
    {
        const char* modality;
        const char* fusion;
        std::size_t columns;
    };
    for (const Source src : {Source{"voice", "", 96},
             Source{"eeg", "", 96},
             Source{"fused", "early", 96},
             Source{"fused", "late", 192}})
    {
        SCOPED_TRACE(std::string(src.modality) + " " + src.fusion);
        thesis::ThesisConfig cfg = thesis::test::phase00(src.modality);
        if (std::string(src.fusion).size() > 0) cfg.dataset.fusion_mode = src.fusion;
        thesis::test::set_handcrafted(cfg);
        cfg.feature_extraction.handcrafted.wavelet = "haar";
        cfg.feature_extraction.handcrafted.descriptors =
            std::vector<std::string>{"energy", "zcr", "entropy", "teager", "jitter", "shimmer"};
        for (const auto* view : {&coretto, &biomex})
        {
            const auto sets = thesis::extract_features(*view, cfg, nullptr);
            ASSERT_EQ(sets.size(), 1u);
            ASSERT_EQ(sets[0].vectors.size(), view->samples.size());
            for (const auto& row : sets[0].vectors)
            {
                EXPECT_EQ(row.size(), src.columns);
                for (double x : row) EXPECT_TRUE(std::isfinite(x));
            }
        }
    }
}

TEST(ThesisBiomexDataset, AViewWithoutRatesIsRefusedByItsReaders)
{
    EXPECT_THROW((void) thesis::require_rate(0.0, "voice"), std::invalid_argument);
    EXPECT_DOUBLE_EQ(thesis::require_rate(128.0, "EEG"), 128.0);
}
