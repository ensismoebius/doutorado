// loaders_gtest.cpp — fast checks for the three grouped-window dataset sources
// wired into the nested-LOSO pipeline (FSDD, AudioMNIST, MIT-BIH). Each test is
// skipped (not failed) when its dataset root is absent, so CI without the corpora
// still passes; on a developer machine with the databases present they exercise
// the real loaders and the grouped fold assignment.

#include <gtest/gtest.h>

#include <filesystem>
#include <set>
#include <string>

#include "GuayaquilConfig.hpp"
#include "GuayaquilDataset.hpp"
#include "GuayaquilMitBih.hpp"
#include "data_loaders/10.5281/zenodo.1342401/datasets/FsddWindowDataset.hpp"

namespace
{
namespace fs = std::filesystem;

const std::string kDbRoot = "/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases";

guayaquil::GuayaquilConfig base_config()
{
    guayaquil::GuayaquilConfig c;
    c.experiment.run_tag = "loaders_test";
    c.experiment.seed = 42;
    c.experiment.repeats = 1;
    c.dataset.window_size = 256;
    c.dataset.cv_num_folds = 6;
    c.dataset.results_dir = ""; // no manifest side effects
    return c;
}

int distinct_speakers(const std::vector<guayaquil::WindowMetadata>& m)
{
    std::set<int> s;
    for (const auto& w : m) s.insert(w.speaker_id);
    return static_cast<int>(s.size());
}

} // namespace

TEST(GuayaquilLoaders, FsddGroupedFoldIsSpeakerDisjoint)
{
    const std::string root = kDbRoot + "/fsdDataset";
    if (!fs::exists(root)) GTEST_SKIP() << "no FSDD root";

    auto cfg = base_config();
    cfg.dataset.dataset_root = root;
    cfg.evaluation.datasets = {"fsdd"};

    const auto split = guayaquil::build_split(cfg, "fsdd", 0);
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

TEST(GuayaquilLoaders, AudioMnistResampledCorpusLoadsAndGroupsBySpeaker)
{
    const std::string root = kDbRoot + "/audioMNIST_8k";
    if (!fs::exists(root)) GTEST_SKIP() << "no AudioMNIST 8k root";

    nn::dataLoaders::fsdd::FsddWindowDataset ds(root, 256);
    ASSERT_FALSE(ds.windows().empty());
    // 60 speakers in the published corpus; grouped into 6 folds of 10.
    EXPECT_GE(distinct_speakers(ds.metadata()), 10);

    auto cfg = base_config();
    cfg.dataset.dataset_root = root;
    cfg.dataset.sources.push_back(
        {"audiomnist", root, 256, 6, 8000, /*max_windows_per_recording=*/2});
    const auto split = guayaquil::build_split(cfg, "audiomnist", 0);
    for (const auto& m : split.train_meta) EXPECT_LT(m.source_window_index, 2);
    EXPECT_FALSE(split.test_samples.empty());
}

TEST(GuayaquilLoaders, MitBihFormat212DecodesAndWindows)
{
    const std::string root = kDbRoot + "/mitbih";
    if (!fs::exists(root)) GTEST_SKIP() << "no MIT-BIH root";

    guayaquil::MitBihWindowDataset ds(root, 256);
    ASSERT_FALSE(ds.windows().empty());
    EXPECT_EQ(ds.windows().size(), ds.metadata().size());
    for (const auto& w : ds.windows()) EXPECT_EQ(w.size(), 256);
    // 48 records in mitdb.
    EXPECT_EQ(distinct_speakers(ds.metadata()), 48);

    auto cfg = base_config();
    cfg.dataset.dataset_root = root;
    cfg.dataset.sources.push_back({"mitbih", root, 256, 6, 360, 40});
    const auto split = guayaquil::build_split(cfg, "mitbih", 0);
    for (const auto& m : split.train_meta) EXPECT_LT(m.source_window_index, 40);
    EXPECT_FALSE(split.test_samples.empty());
}
