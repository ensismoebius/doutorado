// Profile audit: every shipping article profile must parse cleanly via
// GuayaquilConfig::from_nested_json AND must populate the live config
// fields with non-default values that the experiment harness will actually
// consume. Catches silent profile-key drift (e.g. a future rename moving a
// field outside the parser's lookup keys).

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../lib/include/GuayaquilConfig.hpp"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using guayaquil::GuayaquilConfig;

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
        "article-loso.json",
    };
    return profiles;
}

fs::path profiles_dir()
{
    // Tests run from the build dir; profiles live at
    // <repo>/software/nn/src/experiments/guayaquil/profiles.
    fs::path here = fs::path(__FILE__).parent_path();
    return here.parent_path() / "profiles";
}

GuayaquilConfig load(const std::string& name)
{
    const fs::path path = profiles_dir() / name;
    std::ifstream f(path);
    EXPECT_TRUE(f.is_open()) << "missing profile: " << path;
    nlohmann::json j;
    f >> j;
    auto cfg = GuayaquilConfig::from_nested_json(j);
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
    if (cfg.dataset.cv_fold < 0)
    {
        // Pooled-split budgets only apply to the legacy path; nested LOSO uses every
        // window of the speaker-disjoint partitions.
        EXPECT_GT(cfg.dataset.max_loaded_train_samples, 0);
        EXPECT_GT(cfg.dataset.max_validation_samples, 0);
    }
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

INSTANTIATE_TEST_SUITE_P(ArticleProfiles,
    ProfileAuditTest,
    ::testing::ValuesIn(article_profiles()),
    [](const ::testing::TestParamInfo<std::string>& info)
    {
        std::string name = info.param;
        for (auto& c : name)
            if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
        return name;
    });

// The tests above only ever validate profiles that are CORRECT, so the
// failure path -- the half of `validate()` that decides a config is bad --
// had no coverage at all. These cover it, and in particular the property
// that makes the accumulate-then-throw design worth having: one run of
// validation reports every problem, so a broken profile is fixable in one
// pass instead of one error at a time.

namespace
{

/// A config that passes validation, as the starting point for "break one
/// field and check it is caught".
GuayaquilConfig valid_config()
{
    return load("article-lstm-ae.json");
}

std::string validation_error(const GuayaquilConfig& cfg)
{
    try
    {
        cfg.validate();
    }
    catch (const std::invalid_argument& e)
    {
        return e.what();
    }
    return {};
}

} // namespace

TEST(GuayaquilConfigValidation, AcceptsAShippingProfile)
{
    EXPECT_NO_THROW(valid_config().validate());
}

TEST(GuayaquilConfigValidation, RejectsEachSectionAndNamesTheField)
{
    {
        auto cfg = valid_config();
        cfg.experiment.repeats = 0;
        EXPECT_NE(validation_error(cfg).find("experiment.repeats"), std::string::npos);
    }
    {
        auto cfg = valid_config();
        cfg.dataset.window_size = 0;
        EXPECT_NE(validation_error(cfg).find("dataset.window_size"), std::string::npos);
    }
    {
        auto cfg = valid_config();
        cfg.training.epochs = 0;
        EXPECT_NE(validation_error(cfg).find("training.epochs"), std::string::npos);
    }
    {
        auto cfg = valid_config();
        cfg.model.encoder_layer_spec.clear();
        EXPECT_NE(validation_error(cfg).find("model.encoder_layer_spec"), std::string::npos);
    }
    {
        auto cfg = valid_config();
        cfg.evaluation.encodings = {"telepathy"};
        EXPECT_NE(validation_error(cfg).find("unknown encoding"), std::string::npos);
    }
}

TEST(GuayaquilConfigValidation, ReportsEveryProblemInOneMessage)
{
    auto cfg = valid_config();
    cfg.experiment.repeats = 0;
    cfg.dataset.window_size = 0;
    cfg.training.epochs = 0;
    cfg.model.decoder_layer_spec.clear();

    const std::string message = validation_error(cfg);
    EXPECT_NE(message.find("experiment.repeats"), std::string::npos);
    EXPECT_NE(message.find("dataset.window_size"), std::string::npos);
    EXPECT_NE(message.find("training.epochs"), std::string::npos);
    EXPECT_NE(message.find("model.decoder_layer_spec"), std::string::npos);
}

TEST(GuayaquilConfigValidation, LeavesTheSnnKnobsAloneForAnLstmOnlyRun)
{
    // Empty `snn_architectures` means this run is LSTM-only, so unset
    // thresholds are legitimate rather than missing.
    auto cfg = valid_config();
    cfg.evaluation.snn_architectures.clear();
    cfg.evaluation.v_th_values.clear();
    cfg.evaluation.alpha_values.clear();
    EXPECT_NO_THROW(cfg.validate());
}

TEST(GuayaquilConfigValidation, RequiresTheSnnKnobsOnceAnArchitectureIsAsked)
{
    auto cfg = valid_config();
    cfg.evaluation.snn_architectures = {"dense"};
    cfg.evaluation.v_th_values.clear();
    EXPECT_NE(validation_error(cfg).find("v_th_values is empty"), std::string::npos);
}

TEST(GuayaquilConfigValidation, RejectsUnknownBaselineFamily)
{
    auto cfg = valid_config();
    cfg.evaluation.baselines = {"lstm-ae", "mlp-ae"};
    EXPECT_NE(validation_error(cfg).find("unknown family"), std::string::npos);
}

TEST(GuayaquilConfigValidation, RejectsEmptyBaselineList)
{
    auto cfg = valid_config();
    cfg.evaluation.baselines.clear();
    EXPECT_NE(validation_error(cfg).find("evaluation.baselines is empty"), std::string::npos);
}

TEST(GuayaquilConfigValidation, AcceptsTheTrainedBaselineTriple)
{
    auto cfg = valid_config();
    cfg.evaluation.baselines = {"lstm-ae", "gru-ae", "transformer-ae"};
    EXPECT_NO_THROW(cfg.validate());
}

TEST(GuayaquilConfigValidation, DatasetSourceResolutionInheritsAndOverrides)
{
    auto cfg = valid_config();
    cfg.dataset.dataset_root = "/data/fsdd";
    cfg.dataset.window_size = 256;
    cfg.dataset.cv_num_folds = 6;
    cfg.dataset.max_windows_per_recording = 0;
    cfg.dataset.sources = {
        {"fsdd", "/data/fsdd", 0, 6, 0, 0},
        {"mitbih", "/data/mitbih", 0, 6, 360, 40},
    };

    const auto fsdd = cfg.dataset.resolve("fsdd");
    EXPECT_EQ(fsdd.root, "/data/fsdd");
    EXPECT_EQ(fsdd.window_size, 256); // inherited
    EXPECT_EQ(fsdd.max_windows_per_recording, 0);

    const auto mit = cfg.dataset.resolve("mitbih");
    EXPECT_EQ(mit.root, "/data/mitbih");
    EXPECT_EQ(mit.window_size, 256); // inherited
    EXPECT_EQ(mit.sample_rate, 360); // overridden
    EXPECT_EQ(mit.max_windows_per_recording, 40);

    // A name with no entry falls back entirely to the singular Dataset fields.
    const auto other = cfg.dataset.resolve("audiomnist");
    EXPECT_EQ(other.root, "/data/fsdd");
}

TEST(GuayaquilConfigValidation, RejectsAFoldOutsideTheFoldCount)
{
    auto cfg = valid_config();
    cfg.dataset.cv_fold = 6;
    cfg.dataset.cv_num_folds = 6;
    EXPECT_NE(validation_error(cfg).find("cv_fold"), std::string::npos);
}

TEST(GuayaquilConfigValidation, LosoFoldRelaxesThePooledSampleCaps)
{
    // Under nested LOSO (cv_fold >= 0) the split is speaker-disjoint and uses
    // every window, so max_loaded_train_samples / max_validation_samples = 0 is fine.
    auto cfg = valid_config();
    cfg.dataset.cv_fold = 0;
    cfg.dataset.cv_num_folds = 6;
    cfg.dataset.max_loaded_train_samples = 0;
    cfg.dataset.max_validation_samples = 0;
    EXPECT_NO_THROW(cfg.validate());
}

// The two rules below span sections. They are the ones a refactor that
// splits validation per section can silently drop: each checker sees only
// its own struct, so the relation between two structs has nowhere to live
// unless someone deliberately keeps it. Both checkers deliberately take the
// whole config for this reason, and these tests are what proves it stuck.

TEST(GuayaquilConfigValidation, CatchesAFrameSizeThatDoesNotDivideTheWindow)
{
    // dataset.window_size / model.lstm_frame_size is the LSTM's timestep
    // count. A remainder means the last timestep is short, so the rule is a
    // relation between the dataset and the model, not a fact about either.
    auto cfg = valid_config();
    cfg.dataset.window_size = 100;
    cfg.model.lstm_frame_size = 8;

    const std::string message = validation_error(cfg);
    EXPECT_NE(message.find("lstm_frame_size"), std::string::npos);
    EXPECT_NE(message.find("must divide"), std::string::npos);
}

TEST(GuayaquilConfigValidation, CatchesABatchLargerThanTheLoadedSampleBudget)
{
    // A batch bigger than everything loaded cannot ever be filled; the two
    // numbers live in different sections.
    auto cfg = valid_config();
    cfg.dataset.max_loaded_train_samples = 10;
    cfg.training.samples_per_batch = 64;

    EXPECT_NE(validation_error(cfg).find("exceeds max_loaded_train_samples"), std::string::npos);
}
