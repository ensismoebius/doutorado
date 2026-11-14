// loaders_corpus_gtest.cpp -- the real corpora through the real loaders: FSDD, AudioMNIST (8 kHz),
// MIT-BIH, eegmmidb and Siena. Window counts, speaker/subject grouping and the fold assignment
// are checked on the published data, which the synthetic fixtures of loaders_gtest cannot do.
//
// The corpora are large and are not in the repository, so these tests are opt-in:
//   cmake --preset=max-performance -DNN_TEST_DB_ROOT=<dir holding the corpora>
// registers (with ctest) the suite of every corpus found under <dir>, and says at configure time
// which it did not register. The directory names are fsdDataset, audioMNIST_8k, mitbih,
// eegmmidb and siena. Run directly, the binary reads the same variable from the environment.
//
// A test never skips: when NN_TEST_DB_ROOT is unset, or the corpus its suite needs is not under
// it, the test FAILS and says which directory to provide. A silent skip reads as a pass.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "Meeting01Config.hpp"
#include "Meeting01Dataset.hpp"
#include "Meeting01Eeg.hpp"
#include "Meeting01MitBih.hpp"
#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"
#include "loaders_test_support.hpp"
#include "wave/Wav.hpp"

namespace
{
namespace fs = std::filesystem;
using meeting01_test::base_config;
using meeting01_test::distinct_speakers;

// <NN_TEST_DB_ROOT>/<corpus>, which must exist. Throws, naming the cause and the remedy.
std::string corpus_root(const std::string& corpus)
{
    const char* base = std::getenv("NN_TEST_DB_ROOT");
    if (base == nullptr || *base == '\0')
    {
        throw std::runtime_error(
            "NN_TEST_DB_ROOT is not set: export it (or configure with "
            "-DNN_TEST_DB_ROOT=<dir>) to the directory that holds '" +
            corpus + "'");
    }
    const fs::path root = fs::path(base) / corpus;
    if (!fs::is_directory(root))
    {
        throw std::runtime_error("corpus directory '" + root.string() +
                                 "' does not exist: download the corpus there, or do not "
                                 "register this suite (configure without it)");
    }
    return root.string();
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
TEST(CorpusFsdd, FirstSampleIsAudioNotTheHeader)
{
    const fs::path file = fs::path(corpus_root("fsdDataset")) / "recordings/0_jackson_0.wav";
    Wav reader;
    reader.read(file.string());
    ASSERT_FALSE(reader.get_data().empty());
    EXPECT_EQ(reader.get_data().front(), -369.0);
}

// The production caps on the real corpus: FSDD fold 0 kept digits 0-4 in train and 0-2 in val.
TEST(CorpusFsdd, CapsKeepEveryDigit)
{
    const std::string root = corpus_root("fsdDataset");

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

TEST(CorpusFsdd, GroupedFoldIsSpeakerDisjoint)
{
    const std::string root = corpus_root("fsdDataset");

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

TEST(CorpusAudioMnist, LoadsAndGroupsBySpeaker)
{
    const std::string root = corpus_root("audioMNIST_8k");

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

TEST(CorpusMitBih, Format212DecodesAndWindows)
{
    const std::string root = corpus_root("mitbih");

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

TEST(CorpusEegmmidb, LoadsAndGroupsBySubject)
{
    const std::string root = corpus_root("eegmmidb");

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

TEST(CorpusSiena, LoadsAndGroupsBySubject)
{
    const std::string root = corpus_root("siena");

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
