// Profile audit: every Experiment05 profile must parse cleanly and populate
// all required config fields. Catches silent key drift after profile edits.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "../lib/include/ThesisConfig.hpp"
#include "ThesisTestConfig.hpp"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using thesis::ThesisConfig;

namespace
{

fs::path profiles_dir()
{
    fs::path here = fs::path(__FILE__).parent_path();
    return here.parent_path() / "profiles";
}

// The profiles are generated (scripts/pipeline/thesis/00_thesis_generate_profiles.py), and
// the generator writes profiles.manifest: every profile it wrote, plus the hand-written
// debug.json. The audit runs on that list; EveryProfileOnDiskIsAudited closes the other
// direction (a file on disk the manifest does not name, e.g. a hand-added profile).
const std::vector<std::string>& all_profiles()
{
    static const std::vector<std::string> profiles = []
    {
        const fs::path path = profiles_dir() / "profiles.manifest";
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error(
                "missing " + path.string() +
                ": run scripts/pipeline/thesis/00_thesis_generate_profiles.py");
        std::vector<std::string> out;
        for (std::string line; std::getline(f, line);)
            if (!line.empty() && line[0] != '#') out.push_back(line);
        if (out.empty()) throw std::runtime_error(path.string() + " lists no profile");
        return out;
    }();
    return profiles;
}

// Anti-drift guard: every profile shipped on disk (outside the generated smoke/
// mirror) must be in the manifest. A profile added by hand would otherwise escape the
// audit and be overwritten by the next generator run.
std::vector<std::string> profiles_on_disk()
{
    std::vector<std::string> out;
    for (const auto& entry : fs::recursive_directory_iterator(profiles_dir()))
    {
        if (entry.is_directory()) continue;
        const fs::path rel = fs::relative(entry.path(), profiles_dir());
        if (rel.string().find("smoke/") == 0) continue; // generated mirror, not source of truth
        if (rel.extension() != ".json") continue;
        out.push_back(rel.generic_string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

ThesisConfig load(const std::string& name)
{
    const fs::path path = profiles_dir() / name;
    std::ifstream f(path);
    EXPECT_TRUE(f.is_open()) << "missing profile: " << path;
    nlohmann::json j;
    f >> j;
    return ThesisConfig::from_json(j);
}

} // namespace

// Every profile listed in all_profiles() must exist on disk (the parametrized
// tests above already fail on a missing file via load()). This test closes the
// other drift direction: profiles present on disk but absent from the list.
TEST(ThesisProfileAudit, EveryProfileOnDiskIsAudited)
{
    std::vector<std::string> listed = all_profiles();
    std::sort(listed.begin(), listed.end());

    const std::vector<std::string> on_disk = profiles_on_disk();
    std::vector<std::string> missing_from_list;
    std::set_difference(on_disk.begin(),
        on_disk.end(),
        listed.begin(),
        listed.end(),
        std::back_inserter(missing_from_list));

    EXPECT_TRUE(missing_from_list.empty())
        << "profiles on disk but not in profiles.manifest: " << [&]()
    {
        std::string s;
        for (const auto& p : missing_from_list) s += p + " ";
        return s;
    }();
}

class ThesisProfileAuditTest : public ::testing::TestWithParam<std::string>
{
};

TEST_P(ThesisProfileAuditTest, ParsesCleanly)
{
    // from_json already refused any missing, unread, unknown or mistyped key.
    auto cfg = load(GetParam());
    cfg.validate();
    EXPECT_FALSE(cfg.experiment.run_tag->empty());
    EXPECT_NE(cfg.experiment.seed, 0u);
    EXPECT_GT(cfg.experiment.repeats, 0);
    EXPECT_FALSE(cfg.dataset.root->empty());
}

TEST_P(ThesisProfileAuditTest, DatasetModalityValid)
{
    auto cfg = load(GetParam());
    EXPECT_TRUE(cfg.dataset.modality == "voice" || cfg.dataset.modality == "eeg" ||
                cfg.dataset.modality == "fused")
        << "unexpected modality: " << cfg.dataset.modality;
}

TEST_P(ThesisProfileAuditTest, StrategyValid)
{
    auto cfg = load(GetParam());
    EXPECT_TRUE(cfg.feature_extraction.strategy == "handcrafted" ||
                cfg.feature_extraction.strategy == "autoencoder" ||
                cfg.feature_extraction.strategy == "pca")
        << "unexpected strategy: " << cfg.feature_extraction.strategy;
}

TEST_P(ThesisProfileAuditTest, ClassifierValid)
{
    auto cfg = load(GetParam());
    if (!cfg.classifier.enabled) return; // Phase 00: no other classifier key exists
    EXPECT_TRUE(cfg.classifier.type == "ann" || cfg.classifier.type == "dsnn" ||
                cfg.classifier.type == "cosine")
        << "unexpected classifier type: " << cfg.classifier.type;
    if (cfg.classifier.type != "cosine")
    {
        EXPECT_FALSE(cfg.classifier.layer_spec->empty());
    }
    EXPECT_TRUE(
        cfg.classifier.text_mode == "dependent" || cfg.classifier.text_mode == "independent")
        << "unexpected text_mode: " << cfg.classifier.text_mode;
}

TEST_P(ThesisProfileAuditTest, TrainingParamsPositive)
{
    auto cfg = load(GetParam());
    if (!cfg.training.epochs.declared()) return; // nothing trains (handcrafted/PCA Phase 00)
    EXPECT_GT(cfg.training.epochs, 0);
    EXPECT_GT(cfg.training.learning_rate, 0.0f); // declared: what ran is what is written
    EXPECT_GT(cfg.training.samples_per_batch, 0);
    if (cfg.training.early_stop_patience.declared())
    {
        EXPECT_GE(cfg.training.early_stop_patience, 0);
    }
    if (cfg.training.k_folds.declared())
    {
        EXPECT_GE(cfg.training.k_folds, 2);
    }
}

// No field has a default. A config nobody filled in declares nothing, cannot be read, and
// is refused by validate() naming the first key it lacks. (The old defaults included the
// model "lstm-ae" — the extractor the thesis dropped — and an lr resolved from the
// optimizer; both were values no profile had chosen.)
TEST(ThesisNoDefaults, AnEmptyConfigDeclaresNothing)
{
    const ThesisConfig c;
    EXPECT_EQ(c.declared_json(), nlohmann::json::object());
    EXPECT_THROW((void) c.feature_extraction.autoencoder.model.get(), std::logic_error);
    EXPECT_THROW((void) c.training.learning_rate.get(), std::logic_error);
    try
    {
        c.validate();
        FAIL() << "an empty config validated";
    }
    catch (const std::invalid_argument& e)
    {
        EXPECT_NE(std::string(e.what()).find("experiment.run_tag"), std::string::npos) << e.what();
    }
}

// Gradient clipping is a declared knob: 0 = off (what PyTorch does unless the caller clips),
// any positive norm is honoured, a negative one is refused rather than silently used.
TEST(ThesisFidelity, GradientClipIsDeclaredAndChecked)
{
    ThesisConfig c = thesis::test::phase00("eeg");
    thesis::test::set_training(c, 5, 1e-3f, 8);
    thesis::test::set_autoencoder(c,
        "ann-ae",
        {"linear:64:leaky", "linear:16:identity"},
        {"linear:64:leaky", "linear:output:identity"},
        "mse");
    EXPECT_NO_THROW(c.validate());
    c.training.gradient_clip_norm = 0.5f;
    EXPECT_NO_THROW(c.validate());
    c.training.gradient_clip_norm = -1.0f;
    EXPECT_THROW(c.validate(), std::invalid_argument);
}

// .wiki/Guides/Engineering-Fixes-Log.md D6: bark/mel are cochlear (hearing) scales with no
// physiological basis for EEG, and group_by_scale()'s Nyquist normalization made them provably
// degenerate to lfcc there (16 sub-bands -> 16 distinct bins -> one group each == lfcc). They are
// rejected for modality=eeg so the redundant axis cannot be reintroduced.
TEST(ThesisEegScaleAxis, BarkAndMelAreRejectedForEeg)
{
    auto make = [](const std::string& modality, const std::string& scale)
    {
        ThesisConfig c = thesis::test::phase00(modality);
        thesis::test::set_handcrafted(c);
        c.feature_extraction.handcrafted.scale = scale;
        return c;
    };

    // EEG: only lfcc (linear) is admissible.
    EXPECT_NO_THROW(make("eeg", "lfcc").validate());
    EXPECT_THROW(make("eeg", "bark").validate(), std::invalid_argument);
    EXPECT_THROW(make("eeg", "mel").validate(), std::invalid_argument);

    // Voice: all three remain valid — there the scales genuinely differ (bark merges the
    // 16 sub-bands into 9 groups, mel into 11), so the axis carries real information.
    for (const auto* s : {"lfcc", "bark", "mel"}) EXPECT_NO_THROW(make("voice", s).validate());

    // Fused is deliberately NOT restricted: its voice half legitimately uses bark/mel.
    for (const auto* s : {"lfcc", "bark", "mel"}) EXPECT_NO_THROW(make("fused", s).validate());
}

// The learning rate is required for every optimizer. It used to be optional and resolve to
// the optimizer's reference value (adam 1e-3, sgd 1e-2, lion 1e-4, schedule-free-adamw
// 2.5e-3); every shipped profile declared it anyway, so making it required changed no run.
TEST(ThesisOptimizerLearningRate, RequiredForEveryOptimizerAndReadVerbatim)
{
    for (const char* opt : {"adam", "sgd", "lion", "schedule-free-adamw"})
    {
        ThesisConfig c = thesis::test::phase00("eeg");
        thesis::test::set_training(c, 5, 5e-5f, 8);
        c.training.optimizer_type = opt;
        if (std::string(opt) == "sgd") c.training.optimizer_momentum = 0.9f;
        thesis::test::set_autoencoder(c,
            "ann-ae",
            {"linear:64:leaky", "linear:16:identity"},
            {"linear:64:leaky", "linear:output:identity"},
            "mse");
        EXPECT_NO_THROW(c.validate()) << opt;
        EXPECT_FLOAT_EQ(c.training.learning_rate, 5e-5f) << opt;

        nlohmann::json j = c.declared_json();
        j["training"].erase("learning_rate");
        EXPECT_THROW(ThesisConfig::from_json(j), std::invalid_argument) << opt;
    }
}

// Every profile must name an optimizer the OptimizerFactory can actually build; a typo
// here would otherwise only surface as a runtime throw part-way into a long experiment.
TEST_P(ThesisProfileAuditTest, OptimizerTypeIsSupported)
{
    auto cfg = load(GetParam());
    if (!cfg.training.optimizer_type.declared()) return; // nothing trains
    const std::string& t = cfg.training.optimizer_type;
    const bool known = (t == "adam" || t == "sgd" || t == "lion" || t == "schedule-free-adamw");
    EXPECT_TRUE(known) << "unsupported training.optimizer_type: " << t;
    if (cfg.training.optimizer_momentum.declared()) // sgd only
    {
        EXPECT_GE(cfg.training.optimizer_momentum, 0.0f);
        EXPECT_LT(cfg.training.optimizer_momentum, 1.0f);
    }
}

TEST_P(ThesisProfileAuditTest, RegularizationParamsValid)
{
    auto cfg = load(GetParam());
    const auto& t = cfg.training;
    if (t.weight_decay.declared())
    {
        EXPECT_GE(t.weight_decay, 0.0f);
    }
    if (t.firing_rate_reg_lambda.declared())
    {
        EXPECT_GE(t.firing_rate_reg_lambda, 0.0f);
    }
    if (t.firing_rate_min.declared())
    {
        EXPECT_GE(t.firing_rate_min, 0.0f);
        EXPECT_LE(t.firing_rate_max, 1.0f);
        EXPECT_LE(t.firing_rate_min, t.firing_rate_max);
    }
}

TEST_P(ThesisProfileAuditTest, SeedDeterministicFalseForArticleProfiles)
{
    const std::string name = GetParam();
    if (name == "debug.json") return; // debug profile uses deterministic mode
    auto cfg = load(name);
    if (!cfg.experiment.seed_deterministic.declared()) return; // one repeat: nothing to decide
    EXPECT_FALSE(cfg.experiment.seed_deterministic)
        << name << " should have seed_deterministic=false for reproducible sweeps";
}

// ─── classifier.enabled phase gate + handcrafted axis validation ─────────────
// These are non-parametric: they exercise ThesisConfig::from_json/validate on
// synthetic configs rather than the shipped profiles.

namespace
{
// A complete Phase-01 handcrafted dsnn profile: exactly the keys that run reads.
nlohmann::json base_config()
{
    return nlohmann::json{
        {"experiment", {{"run_tag", "t"}, {"seed", 42}, {"repeats", 1}}},
        {"dataset",
            {{"source", "coretto2017"},
                {"root", "/x"},
                {"results_dir", "results/thesis"},
                {"modality", "voice"},
                {"max_samples", 0}}},
        {"feature_extraction",
            {{"strategy", "handcrafted"},
                {"handcrafted",
                    {{"transform", "dtwpt"},
                        {"scale", "lfcc"},
                        {"wavelet", "daub4"},
                        {"descriptors", {"energy"}},
                        {"dtwpt_level", 4},
                        {"cepstral", false}}}}},
        {"paraconsistent", {{"enabled", true}, {"speaker_subsets", false}}},
        {"degradation", {{"enabled", false}}},
        {"classifier",
            {{"type", "dsnn"},
                {"layer_spec", {"linear:64:relu", "residual:1", "linear:N_speakers:identity"}},
                {"text_mode", "independent"},
                {"enabled", true}}},
        {"training",
            {{"epochs", 5},
                {"learning_rate", 0.001},
                {"optimizer_type", "adam"},
                {"gradient_clip_norm", 0.0},
                {"samples_per_batch", 8},
                {"k_folds", 2},
                {"nested_cv", false},
                {"standardize_features", true},
                {"weight_decay", 0.0},
                {"firing_rate_reg_lambda", 0.0},
                {"batch_normalization", "none"}}},
    };
}
} // namespace

TEST(ThesisConfigGate, CompletePhase01ProfileValidates)
{
    EXPECT_NO_THROW(ThesisConfig::from_json(base_config()).validate());
}

TEST(ThesisConfigGate, Phase00HasNoClassifierOrTrainingKeys)
{
    // classifier.enabled=false (Phase 00) with handcrafted features: nothing trains, so the
    // classifier keys and the whole training block are refused, not merely ignored.
    auto j = base_config();
    j["classifier"] = nlohmann::json{{"enabled", false}};
    j.erase("degradation"); // read only when a classifier runs
    EXPECT_THROW(ThesisConfig::from_json(j), std::invalid_argument);
    j.erase("training");
    auto cfg = ThesisConfig::from_json(j);
    EXPECT_FALSE(cfg.classifier.enabled);
    EXPECT_NO_THROW(cfg.validate());
}

TEST(ThesisConfigGate, Phase01RequiresEnabledAndLayerSpec)
{
    // classifier.enabled has no default (it used to default to true)...
    auto j = base_config();
    j["classifier"].erase("enabled");
    EXPECT_THROW(ThesisConfig::from_json(j), std::invalid_argument);
    // ...and a running classifier needs its layer_spec.
    j = base_config();
    j["classifier"].erase("layer_spec");
    EXPECT_THROW(ThesisConfig::from_json(j), std::invalid_argument);
}

TEST(ThesisConfigGate, FusionModeValidated)
{
    auto j = base_config();
    j["dataset"]["modality"] = "fused";
    j["dataset"]["eeg_condition"] = "pronounced";
    EXPECT_THROW(ThesisConfig::from_json(j), std::invalid_argument) << "fused needs fusion_mode";
    j["dataset"]["fusion_mode"] = "early";
    EXPECT_NO_THROW(ThesisConfig::from_json(j).validate());
    j["dataset"]["fusion_mode"] = "bogus";
    EXPECT_THROW(ThesisConfig::from_json(j).validate(), std::invalid_argument);
}

TEST(ThesisConfigGate, WaveletValidated)
{
    auto j = base_config();
    j["feature_extraction"]["handcrafted"]["wavelet"] = "daub46";
    EXPECT_NO_THROW(ThesisConfig::from_json(j).validate());
    j["feature_extraction"]["handcrafted"]["wavelet"] = "daub48"; // no traits
    EXPECT_THROW(ThesisConfig::from_json(j).validate(), std::invalid_argument);
    j["feature_extraction"]["handcrafted"]["wavelet"] = "not-a-wavelet";
    EXPECT_THROW(ThesisConfig::from_json(j).validate(), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(AllProfiles,
    ThesisProfileAuditTest,
    ::testing::ValuesIn(all_profiles()),
    [](const ::testing::TestParamInfo<std::string>& param_info)
    {
        std::string name = param_info.param;
        // Replace non-alphanumeric chars with underscore for GTest naming.
        for (char& c : name)
            if (!std::isalnum(c)) c = '_';
        return name;
    });
