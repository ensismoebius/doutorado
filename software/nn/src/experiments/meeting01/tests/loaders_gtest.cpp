// loaders_gtest.cpp — fast checks for the grouped-window dataset sources wired
// into the nested-LOSO pipeline (FSDD, AudioMNIST, MIT-BIH, and the EDF-format
// EEG loader shared by eegmmidb/siena). Real-corpus tests are skipped (not
// failed) when their dataset root is absent, so CI without the corpora still
// passes; on a developer machine with the databases present they exercise the
// real loaders and the grouped fold assignment. The EEG loader additionally
// gets a synthetic-fixture test (EegSyntheticEdfDecodesAndGroupsBySubject)
// that writes a minimal hand-built .edf, so its header parsing and digital
// decoding are actually exercised even when neither real EEG corpus is present.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <numbers>
#include <set>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "Meeting01Dataset.hpp"
#include "Meeting01Eeg.hpp"
#include "Meeting01MitBih.hpp"
#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"
#include "wave/Wav.hpp"

namespace
{
namespace fs = std::filesystem;

const std::string kDbRoot = "/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases";

meeting01::Meeting01Config base_config()
{
    meeting01::Meeting01Config c;
    c.experiment.run_tag = "loaders_test";
    c.experiment.seed = 42;
    c.experiment.repeats = 1;
    c.dataset.window_size = 256;
    c.dataset.cv_num_folds = 6;
    c.dataset.results_dir = ""; // no manifest side effects
    return c;
}

int distinct_speakers(const std::vector<meeting01::WindowMetadata>& m)
{
    std::set<int> s;
    for (const auto& w : m) s.insert(w.speaker_id);
    return static_cast<int>(s.size());
}

std::string edf_field(const std::string& value, std::size_t width)
{
    if (value.size() >= width) return value.substr(0, width);
    return value + std::string(width - value.size(), ' ');
}

// Writes a minimal single-signal EDF file: header fields sized exactly per the
// spec (https://www.edfplus.info/specs/edf.html), one data record per
// samples_per_record-sized chunk of `digital_values`, little-endian int16
// samples. physical_min/max and digital_min/max are chosen so scale == 1 and
// offset == 0 (physical == digital exactly), which keeps the test's expected
// values simple without needing to special-case the conversion formula.
void write_synthetic_edf(const std::filesystem::path& path,
    const std::vector<std::int16_t>& digital_values,
    int samples_per_record,
    const std::string& record_seconds = "1")
{
    const int n_records = static_cast<int>(digital_values.size()) / samples_per_record;
    ASSERT_EQ(static_cast<int>(digital_values.size()), n_records * samples_per_record);

    std::ofstream out(path, std::ios::binary);
    ASSERT_TRUE(out.is_open());

    const int header_bytes = 256 + 256; // main header + 1 signal's per-signal header
    out << edf_field("0", 8);           // version
    out << edf_field("synthetic patient", 80);
    out << edf_field("synthetic recording", 80);
    out << edf_field("01.01.01", 8); // startdate
    out << edf_field("00.00.00", 8); // starttime
    out << edf_field(std::to_string(header_bytes), 8);
    out << edf_field("", 44); // reserved
    out << edf_field(std::to_string(n_records), 8);
    out << edf_field(record_seconds, 8); // duration of a data record, seconds
    out << edf_field("1", 4);            // ns

    out << edf_field("EEG synth", 16); // label
    out << edf_field("", 80);          // transducer
    out << edf_field("uV", 8);         // physical dimension
    out << edf_field("-16384", 8);     // physical_min
    out << edf_field("16383", 8);      // physical_max
    out << edf_field("-16384", 8);     // digital_min
    out << edf_field("16383", 8);      // digital_max (scale == 1, offset == 0)
    out << edf_field("", 80);          // prefiltering
    out << edf_field(std::to_string(samples_per_record), 8);
    out << edf_field("", 32); // reserved

    out.write(reinterpret_cast<const char*>(digital_values.data()),
        static_cast<std::streamsize>(digital_values.size() * sizeof(std::int16_t)));
}

// A per-recording cap of `cap` windows must hold, and must not mean "the first cap windows".
void expect_drawn_across_recordings(const std::vector<meeting01::WindowMetadata>& meta, int cap)
{
    std::map<int, int> per_recording;
    int furthest = 0;
    for (const auto& m : meta)
    {
        EXPECT_LE(++per_recording[m.recording_id], cap);
        furthest = std::max(furthest, m.source_window_index);
    }
    EXPECT_GE(furthest, cap) << "no window beyond each recording's start";
}

} // namespace

// The real file, read by hand: its first sample is -369 at byte 44 (0_jackson_0.wav). Before
// 2026-11-04 Wav::read gave 18770 here, the letters "RI" of the header.
TEST(Meeting01Loaders, FsddFirstSampleIsAudioNotTheHeader)
{
    const fs::path file = kDbRoot + "/fsdDataset/recordings/0_jackson_0.wav";
    if (!fs::exists(file)) GTEST_SKIP() << "no FSDD root";
    Wav reader;
    reader.read(file.string());
    ASSERT_FALSE(reader.get_data().empty());
    EXPECT_EQ(reader.get_data().front(), -369.0);
}

// The production caps on the real corpus: FSDD fold 0 kept digits 0-4 in train and 0-2 in val.
TEST(Meeting01Loaders, FsddCapsKeepEveryDigitOnTheRealCorpus)
{
    const std::string root = kDbRoot + "/fsdDataset";
    if (!fs::exists(root)) GTEST_SKIP() << "no FSDD root";

    auto cfg = base_config();
    cfg.dataset.dataset_root = root;
    meeting01::Meeting01Config::DatasetSource source{"fsdd", root, 256, 6, 8000, 0};
    source.loso_max_train_windows = 1000;
    source.loso_max_val_windows = 150;
    source.loso_max_test_windows = 1500;
    cfg.dataset.sources.push_back(source);
    const auto split = meeting01::build_split(cfg, "fsdd", 0);

    const auto digits = [](const std::vector<meeting01::WindowMetadata>& meta)
    {
        std::map<int, int> counts;
        for (const auto& m : meta) ++counts[m.digit];
        return counts;
    };
    ASSERT_EQ(split.train_samples.size(), 1000U);
    ASSERT_EQ(split.val_samples.size(), 150U);
    EXPECT_EQ(digits(split.train_meta).size(), 10U);
    EXPECT_EQ(digits(split.val_meta).size(), 10U);
    for (const auto& [digit, n] : digits(split.train_meta)) EXPECT_GE(n, 90) << "digit " << digit;
    for (const auto& [digit, n] : digits(split.val_meta)) EXPECT_GE(n, 12) << "digit " << digit;
}

TEST(Meeting01Loaders, FsddGroupedFoldIsSpeakerDisjoint)
{
    const std::string root = kDbRoot + "/fsdDataset";
    if (!fs::exists(root)) GTEST_SKIP() << "no FSDD root";

    auto cfg = base_config();
    cfg.dataset.dataset_root = root;
    cfg.evaluation.datasets = {"fsdd"};

    const auto split = meeting01::build_split(cfg, "fsdd", 0);
    EXPECT_FALSE(split.train_samples.empty());
    EXPECT_FALSE(split.val_samples.empty());
    EXPECT_FALSE(split.test_samples.empty());

    std::set<std::string> tr, va, te;
    for (const auto& m : split.train_meta) tr.insert(m.speaker);
    for (const auto& m : split.val_meta) va.insert(m.speaker);
    for (const auto& m : split.test_meta) te.insert(m.speaker);
    for (const auto& s : te) EXPECT_EQ(tr.count(s), 0u) << s;
    for (const auto& s : te) EXPECT_EQ(va.count(s), 0u) << s;
    for (const auto& s : va) EXPECT_EQ(tr.count(s), 0u) << s;
}

TEST(Meeting01Loaders, AudioMnistResampledCorpusLoadsAndGroupsBySpeaker)
{
    const std::string root = kDbRoot + "/audioMNIST_8k";
    if (!fs::exists(root)) GTEST_SKIP() << "no AudioMNIST 8k root";

    nn::dataLoaders::fsdd::FsddWindowDataset ds(root, 256);
    ASSERT_FALSE(ds.windows().empty());
    // 60 speakers in the published corpus; grouped into 6 folds of 10.
    EXPECT_GE(distinct_speakers(ds.metadata()), 10);
    // Every window's valid_length is a real sample count in (0, window_size] -- the
    // trailing partial window of a variable-length recording is the only case < 256.
    for (const auto& m : ds.metadata())
    {
        EXPECT_GT(m.valid_length, 0);
        EXPECT_LE(m.valid_length, 256);
    }

    auto cfg = base_config();
    cfg.dataset.dataset_root = root;
    cfg.dataset.sources.push_back(
        {"audiomnist", root, 256, 6, 8000, /*max_windows_per_recording=*/2});
    const auto split = meeting01::build_split(cfg, "audiomnist", 0);
    for (const auto& m : split.train_meta) EXPECT_LT(m.source_window_index, 2);
    EXPECT_FALSE(split.test_samples.empty());
}

TEST(Meeting01Loaders, MitBihFormat212DecodesAndWindows)
{
    const std::string root = kDbRoot + "/mitbih";
    if (!fs::exists(root)) GTEST_SKIP() << "no MIT-BIH root";

    meeting01::MitBihWindowDataset ds(root, 256);
    ASSERT_FALSE(ds.windows().empty());
    EXPECT_EQ(ds.windows().size(), ds.metadata().size());
    for (const auto& w : ds.windows()) EXPECT_EQ(w.size(), 256);
    // The windowing loop drops a trailing partial window rather than padding it (unlike
    // FSDD/AudioMNIST), so every window here is full -- see WindowMetadata::valid_length.
    for (const auto& m : ds.metadata()) EXPECT_EQ(m.valid_length, 256);
    // 48 records in mitdb.
    EXPECT_EQ(distinct_speakers(ds.metadata()), 48);

    auto cfg = base_config();
    cfg.dataset.dataset_root = root;
    cfg.dataset.sources.push_back({"mitbih", root, 256, 6, 360, 40});
    const auto split = meeting01::build_split(cfg, "mitbih", 0);
    expect_drawn_across_recordings(split.train_meta, 40);
    EXPECT_FALSE(split.test_samples.empty());
}

TEST(Meeting01Loaders, EegSyntheticEdfDecodesAndGroupsBySubject)
{
    const fs::path root = fs::temp_directory_path() / "meeting01_eeg_synth_test";
    fs::remove_all(root);
    fs::create_directories(root / "subjA");
    fs::create_directories(root / "subjB");

    // Monotonically increasing digital ramp; physical_min/max == digital_min/max in
    // write_synthetic_edf, so scale == 1 and physical == digital exactly. This test exercises
    // decode/windowing/subject-grouping mechanics only -- it deliberately does NOT assert
    // per-sample monotonicity survives z-score any more, because the bandpass+notch filter now
    // runs first: a monotonic ramp IS low-frequency/DC content by construction, so a 0.5 Hz
    // highpass correctly removes almost all of it. That is the filter doing its job, not a
    // regression -- see BandpassNotchFilter.SubLowCutoffDriftIsAttenuated for the dedicated
    // frequency-response test.
    std::vector<std::int16_t> ramp(16);
    for (std::size_t i = 0; i < ramp.size(); ++i) ramp[i] = static_cast<std::int16_t>(i);
    // 16 samples in one record of 1/32 s: 512 Hz, the rate the dataset below asks for.
    write_synthetic_edf(root / "subjA" / "subjA_01.edf", ramp, 16, "0.03125");
    write_synthetic_edf(root / "subjB" / "subjB_01.edf", ramp, 16, "0.03125");

    meeting01::EegWindowDataset ds(root,
        /*window_size=*/8,
        /*sampling_rate=*/512.0,
        /*notch_hz=*/-1.0);
    ASSERT_EQ(ds.size(), 4u); // 2 windows/file x 2 files
    EXPECT_EQ(ds.windows().size(), ds.metadata().size());
    EXPECT_EQ(distinct_speakers(ds.metadata()), 2);

    for (std::size_t i = 0; i < ds.windows().size(); ++i)
    {
        const auto& w = ds.windows()[i];
        EXPECT_EQ(w.size(), 8);

        const auto& m = ds.metadata()[i];
        EXPECT_TRUE(m.speaker == "subjA" || m.speaker == "subjB");
        EXPECT_EQ(m.source_window_index, static_cast<int>(i) % 2);
        EXPECT_EQ(m.valid_length, 8); // EEG windowing never pads a trailing partial window
    }

    fs::remove_all(root);
}

// S088, S092 and S100 of EEGMMIDB are at 128 Hz among files at 160 Hz. The same 10 Hz tone at
// both rates must give the same windows: same count, same duration, same values. Treated as
// 160 Hz, as they were until 2026-11-04, the 128 Hz windows held a 12.5 Hz tone and the file
// fewer of them.
TEST(Meeting01Loaders, EegFilesAtAnotherRateBecomeTheDatasetRate)
{
    const fs::path root = fs::temp_directory_path() / "meeting01_eeg_rates_test";
    fs::remove_all(root);
    fs::create_directories(root / "S001");
    fs::create_directories(root / "S088");

    constexpr int kSeconds = 41; // 128 Hz: the last sample falls 8 ms short of the end
    const auto tone = [](int rate)
    {
        std::vector<std::int16_t> x(static_cast<std::size_t>(rate * kSeconds));
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = static_cast<std::int16_t>(std::lround(
                8000.0 * std::sin(2.0 * std::numbers::pi * 10.0 * static_cast<double>(i) / rate)));
        return x;
    };
    write_synthetic_edf(root / "S001" / "S001R01.edf", tone(160), 160);
    write_synthetic_edf(root / "S088" / "S088R01.edf", tone(128), 128);

    const meeting01::EegWindowDataset ds(root, 256, /*sampling_rate=*/160.0, /*notch_hz=*/60.0);
    EXPECT_EQ(ds.resampled_recordings(), 1);

    std::vector<std::size_t> a;
    std::vector<std::size_t> b;
    for (std::size_t i = 0; i < ds.size(); ++i)
        (ds.metadata()[i].speaker == "S001" ? a : b).push_back(i);
    ASSERT_EQ(a.size(), 25U);
    ASSERT_EQ(b.size(), a.size());

    // Windows 8-16: well inside the recording, away from the filter's edges.
    for (std::size_t w = 8; w <= 16; ++w)
    {
        const auto& x = ds.windows()[a[w]];
        const auto& y = ds.windows()[b[w]];
        double squared = 0.0;
        for (nn::Index t = 0; t < 256; ++t)
        {
            const double d = static_cast<double>(x.at(t, 0)) - static_cast<double>(y.at(t, 0));
            squared += d * d;
        }
        EXPECT_LT(squared / 256.0, 1e-3) << "window " << w;
    }
    fs::remove_all(root);
}

TEST(Meeting01Loaders, EegRefusesARecordOfNoDurationAndARateBelowTheBand)
{
    const fs::path root = fs::temp_directory_path() / "meeting01_eeg_bad_header_test";
    fs::remove_all(root);
    fs::create_directories(root / "S001");
    const std::vector<std::int16_t> samples(320, 1);
    write_synthetic_edf(root / "S001" / "S001R01.edf", samples, 160, "0");
    EXPECT_THROW(meeting01::EegWindowDataset(root, 256, 160.0, 60.0), std::runtime_error);
    EXPECT_THROW(meeting01::EegWindowDataset(root, 256, 64.0, 60.0), std::invalid_argument);
    fs::remove_all(root);
}

TEST(Meeting01Loaders, EegmmidbRealCorpusLoadsAndGroupsBySubject)
{
    const std::string root = kDbRoot + "/eegmmidb";
    if (!fs::exists(root)) GTEST_SKIP() << "no PhysioNet eegmmidb root";

    meeting01::EegWindowDataset ds(root, 256, /*sampling_rate=*/160.0, /*notch_hz=*/60.0);
    ASSERT_FALSE(ds.windows().empty());
    EXPECT_EQ(ds.windows().size(), ds.metadata().size());
    for (const auto& w : ds.windows()) EXPECT_EQ(w.size(), 256);
    for (const auto& m : ds.metadata()) EXPECT_EQ(m.valid_length, 256);

    auto cfg = base_config();
    cfg.dataset.dataset_root = root;
    cfg.dataset.sources.push_back({"eegmmidb", root, 256, 6, 160, 40});
    const auto split = meeting01::build_split(cfg, "eegmmidb", 0);
    expect_drawn_across_recordings(split.train_meta, 40);
    EXPECT_FALSE(split.test_samples.empty());
    EXPECT_EQ(ds.resampled_recordings(), 36) << "S088, S092 and S100, at 128 Hz";
}

TEST(Meeting01Loaders, SienaRealCorpusLoadsAndGroupsBySubject)
{
    const std::string root = kDbRoot + "/siena";
    if (!fs::exists(root)) GTEST_SKIP() << "no Siena root";

    meeting01::EegWindowDataset ds(root, 256, /*sampling_rate=*/512.0, /*notch_hz=*/50.0);
    ASSERT_FALSE(ds.windows().empty());
    EXPECT_EQ(ds.windows().size(), ds.metadata().size());
    for (const auto& w : ds.windows()) EXPECT_EQ(w.size(), 256);
    for (const auto& m : ds.metadata()) EXPECT_EQ(m.valid_length, 256);

    auto cfg = base_config();
    cfg.dataset.dataset_root = root;
    cfg.dataset.sources.push_back({"siena", root, 256, 6, 512, 40});
    const auto split = meeting01::build_split(cfg, "siena", 0);
    expect_drawn_across_recordings(split.train_meta, 40);
    EXPECT_FALSE(split.test_samples.empty());
    EXPECT_EQ(ds.resampled_recordings(), 0);
}
