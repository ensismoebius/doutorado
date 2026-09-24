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

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "Meeting01Dataset.hpp"
#include "Meeting01Eeg.hpp"
#include "Meeting01MitBih.hpp"
#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"

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
    int samples_per_record)
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
    out << edf_field("1", 8); // duration of a data record, seconds
    out << edf_field("1", 4); // ns

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

} // namespace

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
    for (const auto& m : split.train_meta) EXPECT_LT(m.source_window_index, 40);
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
    write_synthetic_edf(root / "subjA" / "subjA_01.edf", ramp, /*samples_per_record=*/4);
    write_synthetic_edf(root / "subjB" / "subjB_01.edf", ramp, /*samples_per_record=*/4);

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
    for (const auto& m : split.train_meta) EXPECT_LT(m.source_window_index, 40);
    EXPECT_FALSE(split.test_samples.empty());
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
    for (const auto& m : split.train_meta) EXPECT_LT(m.source_window_index, 40);
    EXPECT_FALSE(split.test_samples.empty());
}
