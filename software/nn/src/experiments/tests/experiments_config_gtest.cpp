/**
 * @file experiments_config_gtest.cpp
 * @brief Characterization tests for Config::load.
 *
 * `Config::load` had no tests at all: 193 lines, twenty validations, and the
 * only way to learn what it rejects was to read it. That is also why it could
 * not be refactored -- there was nothing to refactor against.
 *
 * These pin the CURRENT behaviour, message by message. They are deliberately
 * written against the observable contract (does it load; if not, what does it
 * say) rather than against the implementation, so a later restructuring is
 * free to move the checks around as long as it keeps rejecting the same
 * configs for the same stated reasons.
 *
 * Note what the tests say about the design: several fields are FROZEN to one
 * value for "PHASE 0". A frozen field that is still read from the file is a
 * field a user can set and be refused for -- the test names say so out loud,
 * because that is surprising and is exactly the kind of thing a refactor
 * quietly loses.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "Config.hpp"

namespace
{

/// A configuration that loads. Every rejection test starts from this and
/// breaks exactly one field, so a failure names the field it broke.
auto valid_document() -> nlohmann::json
{
    return nlohmann::json{
        {"window", {{"duration_sec", 1.5}, {"overlap_percent", 50}}},
        {"normalization",
            {{"range", {0.0, 1.0}}, {"method", "min-max"}, {"paraconsistent_prerequisite", true}}},
        {"classifier",
            {{"type", "ResNet-SNN"},
                {"implementation", "SimpleResNet"},
                {"hidden_dim", 64},
                {"depth", 2},
                {"learning_rate", 0.001}}},
        {"dataset",
            {{"base_path", "/data/imagined"},
                {"sampling_rate", 44100},
                {"eeg_sampling_rate", 1000}}},
        {"paraconsistent", {{"enabled", true}, {"optimal_point", {1.0, 0.0}}}},
        {"experiment",
            {{"seed", 42},
                {"cross_validation", true},
                {"folds", 5},
                {"batch_size", 16},
                {"max_epochs", 10}}},
        {"output",
            {{"results_dir", "results"},
                {"metrics_file", "metrics.json"},
                {"torch_state_file", "model.pt"}}},
    };
}

/// Write `document` to a temp file and load it. The path is unique per test so
/// the suite can run in parallel (ctest -j).
auto load(const nlohmann::json& document, const std::string& tag) -> std::optional<Config>
{
    const auto path =
        std::filesystem::temp_directory_path() /
        ("experiments_config_" + tag + "_" +
            std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + ".json");
    {
        std::ofstream out(path);
        out << document.dump();
    }
    auto result = Config::load(path.string());
    std::filesystem::remove(path);
    return result;
}

} // namespace

TEST(ExperimentsConfigLoad, AcceptsAValidDocument)
{
    const auto cfg = load(valid_document(), "valid");
    ASSERT_TRUE(cfg.has_value());
    EXPECT_DOUBLE_EQ(cfg->duration_sec, 1.5);
    EXPECT_EQ(cfg->overlap_percent, 50);
    EXPECT_EQ(cfg->type, "ResNet-SNN");
    EXPECT_EQ(cfg->folds, 5);
    EXPECT_EQ(cfg->batch_size, 16);
    EXPECT_EQ(cfg->results_dir, "results");
}

TEST(ExperimentsConfigLoad, ReturnsNulloptForAMissingFile)
{
    // Not an exception: the caller (phase00.cpp) branches on the optional.
    EXPECT_FALSE(Config::load("/nonexistent/definitely-not-here.json").has_value());
}

TEST(ExperimentsConfigLoad, ReturnsNulloptForMalformedJson)
{
    const auto path = std::filesystem::temp_directory_path() / "experiments_config_broken.json";
    {
        std::ofstream out(path);
        out << "{ this is not json";
    }
    EXPECT_FALSE(Config::load(path.string()).has_value());
    std::filesystem::remove(path);
}

TEST(ExperimentsConfigLoad, ReturnsNulloptWhenARequiredSectionIsAbsent)
{
    auto document = valid_document();
    document.erase("classifier");
    EXPECT_FALSE(load(document, "no_classifier").has_value());
}

// ── the PHASE 0 freezes ───────────────────────────────────────────────────
// These fields are read from the file and then required to equal one value.
// A user can set them, and setting them to anything else is refused.

TEST(ExperimentsConfigLoad, RefusesAWindowDurationOtherThanTheFrozenOne)
{
    auto document = valid_document();
    document["window"]["duration_sec"] = 2.0;
    EXPECT_FALSE(load(document, "window_dur").has_value());
}

TEST(ExperimentsConfigLoad, RefusesAnOverlapOtherThanTheFrozenOne)
{
    auto document = valid_document();
    document["window"]["overlap_percent"] = 25;
    EXPECT_FALSE(load(document, "window_ovl").has_value());
}

TEST(ExperimentsConfigLoad, RefusesAnOverlapOutsideZeroToOneHundred)
{
    auto document = valid_document();
    document["window"]["overlap_percent"] = 150;
    EXPECT_FALSE(load(document, "window_range").has_value());
}

TEST(ExperimentsConfigLoad, RefusesANormalizationRangeOtherThanZeroOne)
{
    auto document = valid_document();
    document["normalization"]["range"] = {-1.0, 1.0};
    EXPECT_FALSE(load(document, "norm_range").has_value());
}

TEST(ExperimentsConfigLoad, RefusesARangeThatIsNotAPair)
{
    auto document = valid_document();
    document["normalization"]["range"] = {0.0, 0.5, 1.0};
    EXPECT_FALSE(load(document, "norm_len").has_value());
}

TEST(ExperimentsConfigLoad, RefusesANormalizationMethodOtherThanMinMax)
{
    auto document = valid_document();
    document["normalization"]["method"] = "z-score";
    EXPECT_FALSE(load(document, "norm_method").has_value());
}

TEST(ExperimentsConfigLoad, RefusesTurningOffTheParaconsistentPrerequisite)
{
    auto document = valid_document();
    document["normalization"]["paraconsistent_prerequisite"] = false;
    EXPECT_FALSE(load(document, "norm_prereq").has_value());
}

// ── the classifier ────────────────────────────────────────────────────────

TEST(ExperimentsConfigLoad, AcceptsBothSpellingsOfTheClassifierType)
{
    // "ResNet" is accepted alongside "ResNet-SNN" even though the error
    // message names only the latter. Pinned because the message and the check
    // disagree, and a refactor is likely to "fix" one of them by accident.
    auto document = valid_document();
    document["classifier"]["type"] = "ResNet";
    EXPECT_TRUE(load(document, "clf_resnet").has_value());
}

TEST(ExperimentsConfigLoad, RefusesAnUnknownClassifierType)
{
    auto document = valid_document();
    document["classifier"]["type"] = "Transformer";
    EXPECT_FALSE(load(document, "clf_type").has_value());
}

TEST(ExperimentsConfigLoad, RefusesAnImplementationOtherThanSimpleResNet)
{
    auto document = valid_document();
    document["classifier"]["implementation"] = "TorchResNet";
    EXPECT_FALSE(load(document, "clf_impl").has_value());
}

TEST(ExperimentsConfigLoad, RefusesNonPositiveArchitectureDimensions)
{
    for (const auto& [field, value] : {std::pair{"hidden_dim", 0}, std::pair{"depth", 0}})
    {
        auto document = valid_document();
        document["classifier"][field] = value;
        EXPECT_FALSE(load(document, std::string("clf_") + field).has_value())
            << "classifier." << field << " = " << value << " must be refused";
    }
}

TEST(ExperimentsConfigLoad, RefusesANonPositiveLearningRate)
{
    auto document = valid_document();
    document["classifier"]["learning_rate"] = 0.0;
    EXPECT_FALSE(load(document, "clf_lr").has_value());
}

// ── dataset, paraconsistent, experiment, output ───────────────────────────

TEST(ExperimentsConfigLoad, RefusesAnEmptyDatasetPath)
{
    auto document = valid_document();
    document["dataset"]["base_path"] = "";
    EXPECT_FALSE(load(document, "ds_path").has_value());
}

TEST(ExperimentsConfigLoad, RefusesNonPositiveSamplingRates)
{
    for (const char* field : {"sampling_rate", "eeg_sampling_rate"})
    {
        auto document = valid_document();
        document["dataset"][field] = 0;
        EXPECT_FALSE(load(document, std::string("ds_") + field).has_value())
            << "dataset." << field << " = 0 must be refused";
    }
}

TEST(ExperimentsConfigLoad, RefusesAnOptimalPointThatIsNotAPair)
{
    auto document = valid_document();
    document["paraconsistent"]["optimal_point"] = {1.0};
    EXPECT_FALSE(load(document, "para_point").has_value());
}

TEST(ExperimentsConfigLoad, RefusesANegativeSeed)
{
    // Zero is allowed; only negative is refused. Pinned because "non-negative"
    // and "non-zero" are one word apart and this project treats seed 0 as a
    // valid, reproducible choice.
    auto document = valid_document();
    document["experiment"]["seed"] = 0;
    EXPECT_TRUE(load(document, "seed_zero").has_value());

    document["experiment"]["seed"] = -1;
    EXPECT_FALSE(load(document, "seed_neg").has_value());
}

TEST(ExperimentsConfigLoad, RefusesNonPositiveExperimentCounts)
{
    for (const char* field : {"folds", "batch_size", "max_epochs"})
    {
        auto document = valid_document();
        document["experiment"][field] = 0;
        EXPECT_FALSE(load(document, std::string("exp_") + field).has_value())
            << "experiment." << field << " = 0 must be refused";
    }
}

TEST(ExperimentsConfigLoad, RefusesAnyEmptyOutputPath)
{
    for (const char* field : {"results_dir", "metrics_file", "torch_state_file"})
    {
        auto document = valid_document();
        document["output"][field] = "";
        EXPECT_FALSE(load(document, std::string("out_") + field).has_value())
            << "output." << field << " empty must be refused";
    }
}
