// Profile audit: every shipping article profile must parse cleanly via
// E04Config::from_nested_json AND must populate the live config
// fields with non-default values that the experiment harness will actually
// consume. Catches silent profile-key drift (e.g. a future rename moving a
// field outside the parser's lookup keys).

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../lib/include/E04Config.hpp"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using e04::E04Config;

namespace
{

const std::vector<std::string>& article_profiles()
{
    static const std::vector<std::string> profiles = {
        "article-lstm-ae.json",
        "article-snn-dense.json",
        "article-snn-conv1d.json",
        "article-snn-recurrent.json",
        "article-backend-bench.json",
    };
    return profiles;
}

fs::path profiles_dir()
{
    // Tests run from the build dir; profiles live at <repo>/software/nn/src/experiments/04/profiles.
    fs::path here = fs::path(__FILE__).parent_path();
    return here.parent_path() / "profiles";
}

E04Config load(const std::string& name)
{
    const fs::path path = profiles_dir() / name;
    std::ifstream f(path);
    EXPECT_TRUE(f.is_open()) << "missing profile: " << path;
    nlohmann::json j;
    f >> j;
    auto cfg = E04Config::from_nested_json(j);
    cfg.validate();
    return cfg;
}

} // namespace

class ProfileAuditTest : public ::testing::TestWithParam<std::string>
{
};

TEST_P(ProfileAuditTest, ParsesAndValidates)
{
    auto cfg = load(GetParam());
    EXPECT_FALSE(cfg.experiment.run_tag.empty());
    EXPECT_NE(cfg.experiment.seed, 0u);
    EXPECT_GT(cfg.experiment.repeats, 0);
    EXPECT_FALSE(cfg.dataset.dataset_root.empty());
    EXPECT_GT(cfg.dataset.window_size, 0);
    EXPECT_GT(cfg.dataset.max_loaded_train_samples, 0);
    EXPECT_GT(cfg.dataset.max_validation_samples, 0);
    EXPECT_GT(cfg.training.samples_per_batch, 0);
    EXPECT_GT(cfg.training.epochs, 0);
    EXPECT_GE(cfg.training.early_stop_patience, 0);
    EXPECT_GT(cfg.training.learning_rate, 0.0f);
    EXPECT_FALSE(cfg.model.encoder_layer_spec.empty());
    EXPECT_FALSE(cfg.model.decoder_layer_spec.empty());
    EXPECT_FALSE(cfg.evaluation.datasets.empty());
    EXPECT_FALSE(cfg.evaluation.encodings.empty());
}

TEST_P(ProfileAuditTest, AdamBetasArePopulated)
{
    auto cfg = load(GetParam());
    EXPECT_GT(cfg.training.beta1, 0.0f);
    EXPECT_LT(cfg.training.beta1, 1.0f);
    EXPECT_GT(cfg.training.beta2, 0.0f);
    EXPECT_LT(cfg.training.beta2, 1.0f);
    EXPECT_GT(cfg.training.epsilon, 0.0f);
}

TEST_P(ProfileAuditTest, LossIsMSE)
{
    // Trainer hardcodes MSELoss. Profiles must declare mse so reviewers are
    // not misled. If we later add real loss-type dispatch, relax this.
    auto cfg = load(GetParam());
    EXPECT_EQ(cfg.model.loss_type, "mse");
}

TEST_P(ProfileAuditTest, SeedDeterministicIsFalse)
{
    // Article profiles must produce variance over repeats. seed_deterministic
    // = true would make every repeat identical (silent statistics death).
    auto cfg = load(GetParam());
    EXPECT_FALSE(cfg.experiment.seed_deterministic)
        << "profile " << GetParam() << " has seed_deterministic=true; repeats will be identical";
}

TEST_P(ProfileAuditTest, EvaluationCountsAreConsistent)
{
    auto cfg = load(GetParam());
    if (cfg.evaluation.snn_architectures.empty())
    {
        // LSTM-only profile: no v_th / alpha sweep
        EXPECT_TRUE(cfg.evaluation.v_th_values.empty())
            << "LSTM-only profile " << GetParam() << " must have empty v_th_values";
        EXPECT_TRUE(cfg.evaluation.alpha_values.empty())
            << "LSTM-only profile " << GetParam() << " must have empty alpha_values";
    }
    else
    {
        EXPECT_FALSE(cfg.evaluation.v_th_values.empty())
            << "SNN profile " << GetParam() << " must have non-empty v_th_values";
        EXPECT_FALSE(cfg.evaluation.alpha_values.empty())
            << "SNN profile " << GetParam() << " must have non-empty alpha_values";
    }
}

INSTANTIATE_TEST_SUITE_P(
    ArticleProfiles,
    ProfileAuditTest,
    ::testing::ValuesIn(article_profiles()),
    [](const ::testing::TestParamInfo<std::string>& info)
    {
        std::string name = info.param;
        for (auto& c : name)
            if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
        return name;
    });
