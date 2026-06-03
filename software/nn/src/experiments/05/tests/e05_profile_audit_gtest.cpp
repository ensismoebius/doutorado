// Profile audit: every Experiment05 profile must parse cleanly and populate
// all required config fields. Catches silent key drift after profile edits.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../lib/include/E05Config.hpp"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using e05::E05Config;

namespace
{

const std::vector<std::string>& all_profiles()
{
    static const std::vector<std::string> profiles = {
        "debug.json",
        "handcrafted-eeg.json",
        "handcrafted-voice.json",
        "handcrafted-fused.json",
        "autoencoder-eeg.json",
        "autoencoder-voice.json",
        "autoencoder-fused.json",
        "article-full.json",
    };
    return profiles;
}

fs::path profiles_dir()
{
    fs::path here = fs::path(__FILE__).parent_path();
    return here.parent_path() / "profiles";
}

E05Config load(const std::string& name)
{
    const fs::path path = profiles_dir() / name;
    std::ifstream f(path);
    EXPECT_TRUE(f.is_open()) << "missing profile: " << path;
    nlohmann::json j;
    f >> j;
    return E05Config::from_json(j);
}

} // namespace

class E05ProfileAuditTest : public ::testing::TestWithParam<std::string>
{
};

TEST_P(E05ProfileAuditTest, ParsesCleanly)
{
    auto cfg = load(GetParam());
    cfg.validate();
    EXPECT_FALSE(cfg.experiment.run_tag.empty());
    EXPECT_NE(cfg.experiment.seed, 0u);
    EXPECT_GT(cfg.experiment.repeats, 0);
    EXPECT_FALSE(cfg.dataset.root.empty());
}

TEST_P(E05ProfileAuditTest, DatasetModalityValid)
{
    auto cfg = load(GetParam());
    EXPECT_TRUE(cfg.dataset.modality == "voice" ||
                cfg.dataset.modality == "eeg"   ||
                cfg.dataset.modality == "fused")
        << "unexpected modality: " << cfg.dataset.modality;
}

TEST_P(E05ProfileAuditTest, StrategyValid)
{
    auto cfg = load(GetParam());
    EXPECT_TRUE(cfg.feature_extraction.strategy == "handcrafted" ||
                cfg.feature_extraction.strategy == "autoencoder")
        << "unexpected strategy: " << cfg.feature_extraction.strategy;
}

TEST_P(E05ProfileAuditTest, ClassifierValid)
{
    auto cfg = load(GetParam());
    EXPECT_TRUE(cfg.classifier.type == "rnn" || cfg.classifier.type == "dsnn")
        << "unexpected classifier type: " << cfg.classifier.type;
    EXPECT_FALSE(cfg.classifier.layer_spec.empty());
    EXPECT_TRUE(cfg.classifier.text_mode == "dependent" ||
                cfg.classifier.text_mode == "independent")
        << "unexpected text_mode: " << cfg.classifier.text_mode;
}

TEST_P(E05ProfileAuditTest, TrainingParamsPositive)
{
    auto cfg = load(GetParam());
    EXPECT_GT(cfg.training.epochs, 0);
    EXPECT_GT(cfg.training.learning_rate, 0.0f);
    EXPECT_GT(cfg.training.samples_per_batch, 0);
    EXPECT_GE(cfg.training.early_stop_patience, 0);
    EXPECT_GE(cfg.training.k_folds, 2);
}

TEST_P(E05ProfileAuditTest, SeedDeterministicFalseForArticleProfiles)
{
    const std::string name = GetParam();
    if (name == "debug.json") return; // debug profile uses deterministic mode
    auto cfg = load(name);
    EXPECT_FALSE(cfg.experiment.seed_deterministic)
        << name << " should have seed_deterministic=false for reproducible sweeps";
}

INSTANTIATE_TEST_SUITE_P(AllProfiles,
    E05ProfileAuditTest,
    ::testing::ValuesIn(all_profiles()),
    [](const ::testing::TestParamInfo<std::string>& param_info)
    {
        std::string name = param_info.param;
        // Replace non-alphanumeric chars with underscore for GTest naming.
        for (char& c : name)
            if (!std::isalnum(c)) c = '_';
        return name;
    });
