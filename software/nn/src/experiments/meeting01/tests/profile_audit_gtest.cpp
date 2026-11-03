// Profile audit: every shipping production profile must parse cleanly via
// Meeting01Config::from_nested_json AND must populate the live config
// fields with non-default values that the experiment harness will actually
// consume. Catches silent profile-key drift (e.g. a future rename moving a
// field outside the parser's lookup keys).

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../lib/include/Meeting01Cli.hpp"
#include "../lib/include/Meeting01Config.hpp"
#include "../lib/include/Meeting01Training.hpp"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;
using meeting01::Meeting01Config;

namespace
{

// The four article-*.json profiles (article-lstm-ae, article-snn-{dense,conv1d,
// recurrent}) were deleted 2026-09-23: they never set dataset.cv_fold, so they ran
// the pooled/shuffled legacy split -- the same train/validation speaker-leakage
// defect a reviewer flagged as strong-reject on submission 71 -- and CLAUDE.md
// still pointed at them as "the paper pipeline". meeting01-loso.json (the fixed,
// nested-LOSO grid) is now the only profile that gets this suite's full,
// hand-maintained validation; every other shipped profile still gets the looser
// directory-wide walk below (ProfileDirectoryAudit).
const std::vector<std::string>& production_profiles()
{
    static const std::vector<std::string> profiles = {
        "meeting01-loso.json",
    };
    return profiles;
}

fs::path profiles_dir()
{
    // Tests run from the build dir; profiles live at
    // <repo>/software/nn/src/experiments/meeting01/profiles.
    fs::path here = fs::path(__FILE__).parent_path();
    return here.parent_path() / "profiles";
}

/// The nested-vs-flat rule from Meeting01Cli.cpp::load_config, kept identical so the
/// directory audit below loads each profile the way a real run loads it.
bool is_nested_schema(const nlohmann::json& j)
{
    return j.contains("experiment") && j.contains("dataset") && j.contains("training") &&
           j.contains("model") && j.contains("evaluation");
}

Meeting01Config load(const std::string& name)
{
    const fs::path path = profiles_dir() / name;
    std::ifstream f(path);
    EXPECT_TRUE(f.is_open()) << "missing profile: " << path;
    nlohmann::json j;
    f >> j;
    auto cfg = Meeting01Config::from_nested_json(j);
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
    // Every profile is nested LOSO (cv_fold >= 0 required, validated); the legacy
    // pooled split -- and its max_loaded_train_samples/max_validation_samples
    // budget checks -- was removed 2026-09-23.
    EXPECT_GE(cfg.dataset.cv_fold, 0);
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
    // Production profiles must produce variance over repeats. seed_deterministic
    // = true would make every repeat identical (silent statistics death).
    auto cfg = load(GetParam());
    EXPECT_FALSE(cfg.experiment.seed_deterministic)
        << "profile " << GetParam() << " has seed_deterministic=true; repeats will be identical";
}

TEST_P(ProfileAuditTest, GaBoundsAreSaneWhenSnnArchitecturesPresent)
{
    // SNN architecture search is GA-only (no grid path exists): every profile with a
    // non-empty snn_architectures pool must carry legal GA bounds, or the run has no
    // valid genome to draw from at all.
    auto cfg = load(GetParam());
    if (cfg.evaluation.snn_architectures.empty())
    {
        return; // LSTM-only profile: no SNN arm, GA bounds irrelevant
    }

    const auto& ga = cfg.evaluation.ga.snn;
    EXPECT_GT(ga.population_size, 0) << "profile " << GetParam();
    EXPECT_GE(ga.generations, 0) << "profile " << GetParam();
    EXPECT_LE(ga.min_layers, ga.max_layers) << "profile " << GetParam();
    EXPECT_LE(ga.min_width, ga.max_width) << "profile " << GetParam();
    EXPECT_GT(ga.voltage_threshold_min, 0.0f) << "profile " << GetParam();
    EXPECT_LE(ga.voltage_threshold_min, ga.voltage_threshold_max) << "profile " << GetParam();
    EXPECT_GT(ga.alpha_min, 0.0f) << "profile " << GetParam();
    EXPECT_LT(ga.alpha_max, 1.0f) << "profile " << GetParam();
    EXPECT_LE(ga.alpha_min, ga.alpha_max) << "profile " << GetParam();
}

INSTANTIATE_TEST_SUITE_P(ProductionProfiles,
    ProfileAuditTest,
    ::testing::ValuesIn(production_profiles()),
    [](const ::testing::TestParamInfo<std::string>& param_info)
    {
        std::string name = param_info.param;
        for (auto& c : name)
            if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
        return name;
    });

// End-to-end check for the 2026-09-23 per-dataset latent_dim change: loads the
// REAL shipping meeting01-loso.json (not a synthetic cfg like
// DatasetSourceResolutionInheritsAndOverridesLatentDim above) and confirms
// resolve() gives every dataset the domain-appropriate bottleneck width -- audio
// (fsdd/audiomnist) narrower than EEG (eegmmidb/siena), matching the literature
// table in .wiki/Experiments/Meeting01.md's "latent_dim is fixed, not evolved"
// section. Deliberately does not run the grid itself (see the expensive-experiment
// guard) -- this only proves the config resolution a real run would depend on.
TEST(ProductionProfiles, MeetingOneLosoResolvesLatentDimPerDataset)
{
    auto cfg = load("meeting01-loso.json");
    EXPECT_EQ(cfg.dataset.resolve("fsdd").latent_dim, 16);
    EXPECT_EQ(cfg.dataset.resolve("audiomnist").latent_dim, 16);
    EXPECT_EQ(cfg.dataset.resolve("eegmmidb").latent_dim, 64);
    EXPECT_EQ(cfg.dataset.resolve("siena").latent_dim, 64);
}

// The list above is hand-maintained and names only meeting01-loso.json, so a
// dev/smoke profile could -- and did -- ship on disk in a state that fails
// `validate()` outright, with nothing noticing until someone ran it. On 2026-09-22
// four of them were in exactly that state (`early_stop_patience >= epochs`:
// debug_nested 5>=2, lstm-lightweight 30>=1, minimal-dat-test 2>=1,
// test-dat-writers 5>=1). The failure was loud when finally run, but arbitrarily
// late -- typically the moment someone reached for a quick smoke test.
//
// This walks the directory instead of a list, so a new profile is covered the day it
// lands rather than the day someone remembers to add it here.
TEST(ProfileDirectoryAudit, EveryProfileOnDiskParsesAndValidates)
{
    const fs::path dir = profiles_dir();
    ASSERT_TRUE(fs::is_directory(dir)) << "missing profiles dir: " << dir;

    int seen = 0;
    for (const auto& entry : fs::directory_iterator(dir))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;
        ++seen;

        const std::string name = entry.path().filename().string();
        std::ifstream f(entry.path());
        ASSERT_TRUE(f.is_open()) << name;

        nlohmann::json j;
        ASSERT_NO_THROW(f >> j) << name << ": not parseable JSON";

        Meeting01Config cfg;
        // Same dispatch the CLI uses (Meeting01Cli.cpp::load_config): debug.json is the
        // one remaining flat-schema profile. Duplicating the rule here rather than
        // assuming nested is deliberate -- the test must load a profile exactly the way
        // a real run loads it, or it audits something the binary never sees.
        if (is_nested_schema(j))
        {
            ASSERT_NO_THROW(cfg = Meeting01Config::from_nested_json(j))
                << name << ": from_nested_json threw";
        }
        else
        {
            ASSERT_NO_THROW(cfg = Meeting01Config::from_flat_json(j))
                << name << ": from_flat_json threw";
        }
        EXPECT_NO_THROW(cfg.validate()) << name << ": validate() rejected a shipped profile";
    }

    // 4 profiles remain after the 2026-09-23 unused-profile cleanup (meeting01-loso,
    // lstm-compare, debug, lstm-bench -- each has a real consumer: production run,
    // CLI default, sole flat-schema fixture, and LSTM-Performance.md's cited benchmark,
    // respectively). This floor exists to catch "wrong path" (an empty/near-empty
    // directory), not to pin the exact count.
    EXPECT_GE(seen, 4) << "profiles dir looks empty -- wrong path?";
}

// EVERY profile must state the SNN's simulation depth T explicitly -- including the
// LSTM-only ones. lstm-bench.json once inherited the struct default silently, which changed
// what that throughput benchmark measured without a single line of the profile changing.
// T is stated in one of two ways:
//   fixed     model.time_steps = T
//   searched  evaluation.ga.snn.time_steps_min / time_steps_max (results_format 6: T is a
//             gene, and model.time_steps is then read by no SNN, so a profile may omit it)
// Since results_format 4 the LSTM/GRU/Transformer-AE read model.recurrent_time_steps, not T.
//
// The point is that a run's temporal resolution must be readable from the profile rather
// than from a header nobody opens.
TEST(ProfileDirectoryAudit, EveryProfileDeclaresTimeStepsExplicitly)
{
    for (const auto& entry : fs::directory_iterator(profiles_dir()))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;

        const std::string name = entry.path().filename().string();
        std::ifstream f(entry.path());
        nlohmann::json j;
        f >> j;

        if (!is_nested_schema(j)) continue; // flat schema has no model section

        const nlohmann::json snn_ga =
            j.value("/evaluation/ga/snn"_json_pointer, nlohmann::json::object());
        const bool searched =
            snn_ga.contains("time_steps_min") || snn_ga.contains("time_steps_max");
        const bool fixed = j.contains("model") && j["model"].contains("time_steps");
        ASSERT_TRUE(fixed || searched)
            << name
            << ": neither model.time_steps nor evaluation.ga.snn.time_steps_min/max -- the "
               "SNN would silently run the struct default's T";
        const int steps =
            searched ? snn_ga.at("time_steps_min").get<int>() : j["model"]["time_steps"].get<int>();
        EXPECT_GE(steps, 2) << name
                            << ": a single step disables membrane dynamics and "
                               "spike coding entirely";
    }
}

// The same rule for the sequence models: since results_format 4 the LSTM/GRU/Transformer-AE
// read S = model.recurrent_time_steps * window_size / lstm_frame_size frames, so that field
// is their temporal resolution. lstm-bench.json and lstm-compare.json inherited it (1) until
// 2026-11-02 while lstm-bench's comment still claimed model.time_steps set its unroll (16,
// i.e. S = 512): the benchmark measured S = 32 and its own profile said otherwise.
// Required only of a profile that runs one of them: the matched control ann-ae overrides
// both fields (run_ann_twin: one frame, the whole window), so meeting01-loso.json, which
// since 2026-11-02 runs no sequence baseline, would only carry a value nothing reads.
TEST(ProfileDirectoryAudit, EveryProfileDeclaresTheBaselinesSequenceExplicitly)
{
    for (const auto& entry : fs::directory_iterator(profiles_dir()))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;

        const std::string name = entry.path().filename().string();
        std::ifstream f(entry.path());
        nlohmann::json j;
        f >> j;

        if (!is_nested_schema(j)) continue; // flat schema has no model section
        // Absent, the list is Meeting01Config's default, which is lstm-ae.
        const auto baselines =
            j["evaluation"].value("baselines", std::vector<std::string>{"lstm-ae"});
        const bool reads_frames = std::any_of(
            baselines.begin(), baselines.end(), [](const std::string& b) { return b != "ann-ae"; });
        if (!reads_frames) continue;

        ASSERT_TRUE(j.contains("model") && j["model"].contains("recurrent_time_steps"))
            << name
            << ": no model.recurrent_time_steps -- the LSTM/GRU/Transformer-AE would silently "
               "read the struct default's sequence length";
        EXPECT_GE(j["model"]["recurrent_time_steps"].get<int>(), 1) << name;
    }
}

// A profile still carrying the pre-2026-09-22 key must fail loudly. Silently ignoring it
// would fall back to the 16-step default -- a run that completes and reports a plausible
// number under a temporal resolution nobody chose, which is the exact failure mode this
// audit existed to eliminate. Copies of these profiles live on the cluster, so the old
// key WILL show up again.
TEST(RenamedKey, OldSnnTimeStepsKeyIsRejectedNotIgnored)
{
    const auto base = nlohmann::json::parse(R"({
        "experiment": {"run_tag": "t", "seed": 42, "repeats": 1},
        "dataset": {"dataset_root": "/tmp", "window_size": 64,
                    "max_loaded_train_samples": 10, "max_validation_samples": 5},
        "training": {"samples_per_batch": 1, "epochs": 2, "early_stop_patience": 1, "early_stop_min_skill": 0.1,
                     "learning_rate": 0.001},
        "model": {"encoder_layer_spec": ["linear:16:leaky", "linear:8:identity"],
                  "decoder_layer_spec": ["linear:8:leaky", "linear:output:identity"],
                  "time_steps": 8},
        "evaluation": {"datasets": ["fsdd"], "encodings": ["direct"],
                       "snn_architectures": []}
    })");

    // Sanity: the renamed key parses and lands where it should.
    Meeting01Config ok;
    ASSERT_NO_THROW(ok = Meeting01Config::from_nested_json(base));
    EXPECT_EQ(ok.model.time_steps, 8);

    nlohmann::json stale = base;
    stale["model"].erase("time_steps");
    stale["model"]["snn_time_steps"] = 8;

    try
    {
        (void) Meeting01Config::from_nested_json(stale);
        ADD_FAILURE() << "the old key was accepted; it must throw";
    }
    catch (const std::invalid_argument& e)
    {
        // The exception must name the cause AND the remedy (CLAUDE.md's no-fallbacks rule).
        const std::string msg = e.what();
        EXPECT_NE(msg.find("snn_time_steps"), std::string::npos) << msg;
        EXPECT_NE(msg.find("time_steps"), std::string::npos) << msg;
        EXPECT_NE(msg.find("Remedy"), std::string::npos) << msg;
    }

    // Carrying BOTH keys is also a rejection, not a "the new one wins" merge.
    nlohmann::json both = base;
    both["model"]["snn_time_steps"] = 4;
    EXPECT_THROW((void) Meeting01Config::from_nested_json(both), std::invalid_argument);
}

// The GA budget must be readable from the profile as well. Four dev profiles inherited
// population 10 x (1 + 8 generations) = 90 evaluations from the struct default, which at
// lstm-compare's settings (500 windows, 100 epochs, 3 repeats) is roughly a day and a
// half of CPU -- on the profile the CLI runs when given no --comparative-config at all
// (Meeting01Cli.cpp: kDefaultComparativeProfileStem = "lstm-compare").
TEST(ProfileDirectoryAudit, SnnProfilesDeclareTheirGaBudgetExplicitly)
{
    for (const auto& entry : fs::directory_iterator(profiles_dir()))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".json") continue;

        const std::string name = entry.path().filename().string();
        std::ifstream f(entry.path());
        nlohmann::json j;
        f >> j;

        if (!is_nested_schema(j)) continue;

        const auto& eval = j["evaluation"];
        const bool has_snn = eval.contains("snn_architectures") &&
                             !eval["snn_architectures"].get<std::vector<std::string>>().empty();
        if (!has_snn) continue; // no SNN arm: the GA never runs

        ASSERT_TRUE(eval.contains("ga"))
            << name
            << " declares an SNN arm but no evaluation.ga block; the search budget "
               "would be inherited invisibly from Meeting01Config::Ga";
        ASSERT_TRUE(eval["ga"].contains("snn"))
            << name
            << " declares an SNN arm but no evaluation.ga.snn block (2026-09-22 nested "
               "schema); the search budget would be inherited invisibly from "
               "Meeting01Config::Ga's defaults";
        const auto& ga = eval["ga"]["snn"];
        EXPECT_TRUE(ga.contains("population_size")) << name;
        EXPECT_TRUE(ga.contains("generations")) << name;
    }
}

// The tests above only ever validate profiles that are CORRECT, so the
// failure path -- the half of `validate()` that decides a config is bad --
// had no coverage at all. These cover it, and in particular the property
// that makes the accumulate-then-throw design worth having: one run of
// validation reports every problem, so a broken profile is fixable in one
// pass instead of one error at a time.

namespace
{

/// A config that passes validation, as the starting point for "break one
/// field and check it is caught". article-lstm-ae.json (used here before
/// 2026-09-23) was deleted along with the other three article-*.json profiles --
/// they never set cv_fold and so ran the leakage-prone legacy pooled split.
/// meeting01-loso.json is now the only production profile.
Meeting01Config valid_config()
{
    return load("meeting01-loso.json");
}

std::string validation_error(const Meeting01Config& cfg)
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

TEST(Meeting01ConfigValidation, AcceptsAShippingProfile)
{
    EXPECT_NO_THROW(valid_config().validate());
}

TEST(Meeting01ConfigValidation, RejectsEachSectionAndNamesTheField)
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
    {
        auto cfg = valid_config();
        cfg.model.denoising_noise_std = -0.1F;
        EXPECT_NE(validation_error(cfg).find("model.denoising_noise_std"), std::string::npos);
    }
}

TEST(Meeting01ConfigValidation, DenoisingNoiseStdDefaultsToDisabledAndValidates)
{
    // 0.0f = disabled (exact pass-through in GaussianNoise), matching the
    // codebase-wide "0 = derive/disabled" convention (latent_dim, lstm_hidden_size).
    auto cfg = valid_config();
    EXPECT_FLOAT_EQ(cfg.model.denoising_noise_std, 0.0F);
    EXPECT_NO_THROW(cfg.validate());
}

TEST(Meeting01ConfigValidation, AcceptsAPositiveDenoisingNoiseStd)
{
    auto cfg = valid_config();
    cfg.model.denoising_noise_std = 0.05F;
    EXPECT_NO_THROW(cfg.validate());
}

TEST(Meeting01ConfigValidation, DenoisingNoiseStdParsesFromNestedJson)
{
    const auto j = nlohmann::json::parse(R"({
        "experiment": {"run_tag": "t", "seed": 42, "repeats": 1},
        "dataset": {"dataset_root": "/tmp", "window_size": 64,
                    "max_loaded_train_samples": 10, "max_validation_samples": 5},
        "training": {"samples_per_batch": 1, "epochs": 2, "early_stop_patience": 1, "early_stop_min_skill": 0.1,
                     "learning_rate": 0.001},
        "model": {"encoder_layer_spec": ["linear:16:leaky", "linear:8:identity"],
                  "decoder_layer_spec": ["linear:8:leaky", "linear:output:identity"],
                  "time_steps": 8, "denoising_noise_std": 0.03},
        "evaluation": {"datasets": ["fsdd"], "encodings": ["direct"],
                       "snn_architectures": []}
    })");

    Meeting01Config cfg;
    ASSERT_NO_THROW(cfg = Meeting01Config::from_nested_json(j));
    EXPECT_FLOAT_EQ(cfg.model.denoising_noise_std, 0.03F);
}

// The SNN-AE design the 2026-11-01 gap diagnosis led to (results_format 5): latent averaged
// over the T steps, analog decoder run once; and the run the author fixed on 2026-11-02
// (format 6): T a free gene in [2, 32], the matched control ann-ae, 1000 training windows.
// Pinned so a stray edit of the paper profile shows up here instead of in a finished grid.
TEST(ProductionProfiles, MeetingOneLosoUsesTheDiagnosedSnnDesign)
{
    const auto cfg = valid_config();
    EXPECT_EQ(cfg.model.snn_time_pool, "latent");
    EXPECT_EQ(cfg.model.snn_decoder_neuron, "analog");
    EXPECT_EQ(cfg.model.initializer, "xavier");
    EXPECT_EQ(cfg.evaluation.ga.snn.time_steps_min, 2);
    EXPECT_EQ(cfg.evaluation.ga.snn.time_steps_max, 32);
    EXPECT_EQ(meeting01::snn_time_steps_range(cfg), (std::pair<int, int>{2, 32}));
    // The matched control only: the searched LSTM-AE, GRU-AE and Transformer-AE were removed
    // 2026-11-02 on the evidence of the pre-validation (meeting01-prevalidation.json).
    EXPECT_EQ(cfg.evaluation.baselines, (std::vector<std::string>{"ann-ae"}));
    EXPECT_EQ(cfg.dataset.loso_max_train_windows, 1000);
}

namespace
{

nlohmann::json read_profile_json(const std::string& name)
{
    std::ifstream f(profiles_dir() / name);
    EXPECT_TRUE(f.is_open()) << "missing profile: " << name;
    nlohmann::json j;
    f >> j;
    return j;
}

// The profile without its "_..." comment keys, at every depth.
nlohmann::json without_comments(const nlohmann::json& j)
{
    if (!j.is_object()) return j;
    nlohmann::json out = nlohmann::json::object();
    for (const auto& [k, v] : j.items())
        if (k.empty() || k[0] != '_') out[k] = without_comments(v);
    return out;
}

} // namespace

// The pre-validation is evidence for removing the three sequence baselines from the grid only
// if it ran them the way the grid would have: the same windows, caps, training, SNN search
// space and model, on the development folds, with only the search budget reduced. Each
// difference allowed below is one the pre-validation declares; any other one (an edit of
// meeting01-loso.json not carried over, or the reverse) fails here, before hours of runs
// produce a table that compares something else. The three families' own GA blocks and the
// sequence fields exist in the pre-validation only: the grid no longer runs those families.
TEST(ProductionProfiles, PrevalidationDiffersFromTheGridOnlyWhereDeclared)
{
    nlohmann::json grid = without_comments(read_profile_json("meeting01-loso.json"));
    nlohmann::json pre = without_comments(read_profile_json("meeting01-prevalidation.json"));
    const std::vector<std::string> declared = {"/experiment/run_tag",
        "/experiment/repeats",
        "/dataset/results_dir",
        "/dataset/save_models",
        "/evaluation/datasets",
        "/evaluation/baselines",
        "/evaluation/ga/snn/population_size",
        "/evaluation/ga/snn/generations",
        "/evaluation/ga/snn/winner_seeds",
        "/evaluation/ga/lstm",
        "/evaluation/ga/gru",
        "/evaluation/ga/transformer",
        "/model/recurrent_time_steps",
        "/model/lstm_frame_size",
        "/model/lstm_hidden_size"};
    for (const auto& ptr : declared)
    {
        const nlohmann::json::json_pointer p(ptr);
        for (auto* j : {&grid, &pre})
            if (j->contains(p)) (*j)[p.parent_pointer()].erase(p.back());
    }
    EXPECT_EQ(grid, pre) << "meeting01-prevalidation.json differs from meeting01-loso.json "
                            "outside its declared fields: "
                         << nlohmann::json::diff(grid, pre).dump();
}

// What the pre-validation runs: every family, the SNN arm, the development folds' datasets
// (DEVELOPMENT_FOLDS in scripts/pipeline/meeting01/meeting01_results.py, which the run script
// checks against too), and ONE reduced budget for all of them -- a family searched longer than
// another would win the comparison by budget, not by architecture.
TEST(ProductionProfiles, PrevalidationRunsEveryFamilyOnOneReducedBudget)
{
    const auto grid = load("meeting01-loso.json");
    const auto pre = load("meeting01-prevalidation.json");
    EXPECT_EQ(pre.evaluation.baselines,
        (std::vector<std::string>{"ann-ae", "lstm-ae", "gru-ae", "transformer-ae"}));
    EXPECT_EQ(pre.evaluation.snn_architectures, grid.evaluation.snn_architectures);
    EXPECT_EQ(pre.evaluation.datasets, (std::vector<std::string>{"fsdd", "eegmmidb"}));

    const auto& g = pre.evaluation.ga;
    const auto budget = [](int pop, int gens, int seeds) { return std::tuple{pop, gens, seeds}; };
    const auto snn = budget(g.snn.population_size, g.snn.generations, g.snn.winner_seeds);
    EXPECT_EQ(budget(g.lstm.population_size, g.lstm.generations, g.lstm.winner_seeds), snn);
    EXPECT_EQ(budget(g.gru.population_size, g.gru.generations, g.gru.winner_seeds), snn);
    EXPECT_EQ(
        budget(
            g.transformer.population_size, g.transformer.generations, g.transformer.winner_seeds),
        snn);
    EXPECT_LT(g.snn.population_size * (1 + g.snn.generations),
        grid.evaluation.ga.snn.population_size * (1 + grid.evaluation.ga.snn.generations));
    // The paper gives one range for LSTM-AE and GRU-AE.
    EXPECT_EQ(g.lstm.min_hidden, g.gru.min_hidden);
    EXPECT_EQ(g.lstm.max_hidden, g.gru.max_hidden);
    EXPECT_EQ(g.lstm.min_layers, g.gru.min_layers);
    EXPECT_EQ(g.lstm.max_layers, g.gru.max_layers);
}

// The per-fold LaTeX exports (dataset.latex_data_dir) are files nothing reads: the paper's
// data come only from 02_/04_/00_. Pointed at the paper's data/, they sat next to the
// generated files, and the pre-validation's four family processes -- one run tag -- wrote the
// same names there and overwrote each other (smoke test, 2026-11-02).
TEST(ProductionProfiles, LeaveNoExportsNothingReads)
{
    for (const char* name : {"meeting01-loso.json", "meeting01-prevalidation.json"})
        EXPECT_EQ(load(name).dataset.latex_data_dir, "") << name;
}

// The stopping rule's arming level (Meeting01Stopping.hpp): the grid and the pre-validation
// must train under the same rule, at the value its rationale was measured for.
TEST(ProductionProfiles, ArmPatienceAtTenPercentSkill)
{
    for (const char* name : {"meeting01-loso.json", "meeting01-prevalidation.json"})
    {
        const auto cfg = load(name);
        EXPECT_FLOAT_EQ(cfg.training.early_stop_min_skill, 0.10F) << name;
        EXPECT_EQ(cfg.training.early_stop_patience, 5) << name;
    }
}

TEST(Meeting01ConfigValidation, EarlyStopMinSkillIsRequiredAndBelowOne)
{
    for (const float bad : {-1.0F, 1.0F, 1.5F})
    {
        auto cfg = valid_config();
        cfg.training.early_stop_min_skill = bad;
        EXPECT_NE(validation_error(cfg).find("early_stop_min_skill"), std::string::npos) << bad;
    }
    auto zero = valid_config();
    zero.training.early_stop_min_skill = 0.0F; // arms at the mean predictor's loss itself
    EXPECT_EQ(validation_error(zero).find("early_stop_min_skill"), std::string::npos);
}

TEST(Meeting01ConfigValidation, SnnTimePoolAndDecoderNeuronDefaultToTheFormat4Network)
{
    const Meeting01Config::Model defaults;
    EXPECT_EQ(defaults.snn_time_pool, "none");
    EXPECT_EQ(defaults.snn_decoder_neuron, "lif");
    auto cfg = valid_config();
    cfg.model.snn_time_pool = "none";
    cfg.model.snn_decoder_neuron = "lif";
    EXPECT_NO_THROW(cfg.validate());
    cfg.model.snn_time_pool = "output"; // an average after a spiking decoder is legal
    EXPECT_NO_THROW(cfg.validate());
}

TEST(Meeting01ConfigValidation, RejectsAnUnknownSnnTimePoolOrDecoderNeuron)
{
    auto pool = valid_config();
    pool.model.snn_time_pool = "mean"; // not a token: refused, not read as "none"
    EXPECT_NE(validation_error(pool).find("model.snn_time_pool"), std::string::npos);

    auto neuron = valid_config();
    neuron.model.snn_decoder_neuron = "relu";
    EXPECT_NE(validation_error(neuron).find("model.snn_decoder_neuron"), std::string::npos);
}

TEST(Meeting01ConfigValidation, RejectsALatentPoolWithASpikingDecoder)
{
    // After the latent average the decoder sees one row per window: a LifBPTT there has no
    // time axis to unroll.
    auto cfg = valid_config();
    cfg.model.snn_time_pool = "latent";
    cfg.model.snn_decoder_neuron = "lif";
    const std::string msg = validation_error(cfg);
    EXPECT_NE(msg.find("needs model.snn_decoder_neuron"), std::string::npos) << msg;
}

TEST(Meeting01ConfigValidation, SnnTimePoolAndDecoderNeuronParseFromNestedJson)
{
    const auto j = nlohmann::json::parse(R"({
        "experiment": {"run_tag": "t", "seed": 42, "repeats": 1},
        "dataset": {"dataset_root": "/tmp", "window_size": 64,
                    "max_loaded_train_samples": 10, "max_validation_samples": 5},
        "training": {"samples_per_batch": 1, "epochs": 2, "early_stop_patience": 1, "early_stop_min_skill": 0.1,
                     "learning_rate": 0.001},
        "model": {"encoder_layer_spec": ["linear:16:leaky", "linear:8:identity"],
                  "decoder_layer_spec": ["linear:8:leaky", "linear:output:identity"],
                  "time_steps": 4, "snn_time_pool": "output", "snn_decoder_neuron": "analog"},
        "evaluation": {"datasets": ["fsdd"], "encodings": ["direct"],
                       "snn_architectures": []}
    })");

    Meeting01Config cfg;
    ASSERT_NO_THROW(cfg = Meeting01Config::from_nested_json(j));
    EXPECT_EQ(cfg.model.snn_time_pool, "output");
    EXPECT_EQ(cfg.model.snn_decoder_neuron, "analog");
}

TEST(Meeting01ConfigValidation, ReportsEveryProblemInOneMessage)
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

TEST(Meeting01ConfigValidation, LeavesGaBoundsAloneForAnLstmOnlyRun)
{
    // Empty `snn_architectures` means this run is LSTM-only — GA bounds are
    // irrelevant and unchecked (there is no SNN arm to search).
    auto cfg = valid_config();
    cfg.evaluation.snn_architectures.clear();
    cfg.evaluation.baselines = {"lstm-ae"}; // ann-ae copies an SNN winner: none here
    EXPECT_NO_THROW(cfg.validate());
}

TEST(Meeting01ConfigValidation, RequiresLegalGaBoundsOnceAnArchitectureIsAsked)
{
    // SNN architecture search is GA-only (no grid path exists): a non-empty
    // snn_architectures pool with an illegal GA bound must be rejected.
    auto cfg = valid_config();
    cfg.evaluation.snn_architectures = {"dense"};
    cfg.evaluation.ga.snn.population_size = 0;
    EXPECT_NE(validation_error(cfg).find("population_size"), std::string::npos);
}

TEST(Meeting01ConfigValidation, RejectsUnknownBaselineFamily)
{
    auto cfg = valid_config();
    cfg.evaluation.baselines = {"lstm-ae", "mlp-ae"};
    EXPECT_NE(validation_error(cfg).find("unknown family"), std::string::npos);
}

TEST(Meeting01ConfigValidation, RejectsEmptyBaselineList)
{
    auto cfg = valid_config();
    cfg.evaluation.baselines.clear();
    EXPECT_NE(validation_error(cfg).find("evaluation.baselines is empty"), std::string::npos);
}

TEST(Meeting01ConfigValidation, AcceptsTheTrainedBaselineTriple)
{
    auto cfg = valid_config();
    cfg.evaluation.baselines = {"lstm-ae", "gru-ae", "transformer-ae"};
    EXPECT_NO_THROW(cfg.validate());
}

TEST(Meeting01ConfigValidation, AnnAeIsTheMatchedControlOfTheSnnArm)
{
    // "ann-ae" copies the SNN winner's layer widths, so it is valid only next to an SNN arm.
    auto cfg = valid_config();
    cfg.evaluation.snn_architectures = {"dense"};
    cfg.evaluation.baselines = {"gru-ae", "ann-ae"};
    cfg.model.initializer = "xavier";
    EXPECT_NO_THROW(cfg.validate());

    cfg.evaluation.snn_architectures.clear();
    EXPECT_NE(validation_error(cfg).find("matched ANN control"), std::string::npos);
}

TEST(Meeting01ConfigValidation, AnnAeNeedsTheSameInitializerAsTheSnn)
{
    // Left empty, each builder keeps its own init (kaiming SNN, xavier ANN), and the control
    // would differ from the SNN in its starting weights as well as its neuron.
    auto cfg = valid_config();
    cfg.evaluation.snn_architectures = {"dense"};
    cfg.evaluation.baselines = {"ann-ae"};
    cfg.model.initializer.clear();
    EXPECT_NE(validation_error(cfg).find("model.initializer is empty"), std::string::npos);

    cfg.model.initializer = "he"; // not a token
    EXPECT_NE(validation_error(cfg).find("model.initializer must be"), std::string::npos);

    cfg.model.initializer = "kaiming";
    EXPECT_NO_THROW(cfg.validate());
}

TEST(Meeting01ConfigValidation, DatasetSourceResolutionInheritsAndOverrides)
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

TEST(Meeting01ConfigValidation, DatasetSourceResolutionInheritsAndOverridesLatentDim)
{
    // 2026-09-23: latent_dim is per-dataset (audio vs. EEG bottleneck width), same
    // 0-means-inherit / >0-means-override pattern as sample_rate/max_windows_per_recording
    // above -- mirrors that test.
    auto cfg = valid_config();
    cfg.dataset.dataset_root = "/data/fsdd";
    cfg.dataset.window_size = 256;
    cfg.dataset.cv_num_folds = 6;
    cfg.dataset.latent_dim = 32; // Dataset-level fallback
    cfg.dataset.sources = {
        {"fsdd", "/data/fsdd", 0, 6, 0, 0, 0, 0, 0, 16},
        {"audiomnist", "/data/audiomnist", 0, 6, 0, 0, 0, 0, 0, 0},
    };

    const auto fsdd = cfg.dataset.resolve("fsdd");
    EXPECT_EQ(fsdd.latent_dim, 16); // source-level override wins

    const auto audiomnist = cfg.dataset.resolve("audiomnist");
    EXPECT_EQ(audiomnist.latent_dim, 32); // no source override -> inherits Dataset-level

    // Neither a source override nor a Dataset-level value -> resolves to 0. The 0
    // itself is not a bottleneck width; run_comparative_experiment only overrides
    // model.latent_dim when resolve(...).latent_dim > 0, so 0 here means "this
    // dataset kept the profile's single global model.latent_dim untouched".
    cfg.dataset.latent_dim = 0;
    const auto eegmmidb = cfg.dataset.resolve("eegmmidb"); // no matching source entry at all
    EXPECT_EQ(eegmmidb.latent_dim, 0);
}

TEST(Meeting01ConfigValidation, RejectsAFoldOutsideTheFoldCount)
{
    auto cfg = valid_config();
    cfg.dataset.cv_fold = 6;
    cfg.dataset.cv_num_folds = 6;
    EXPECT_NE(validation_error(cfg).find("cv_fold"), std::string::npos);
}

TEST(Meeting01ConfigValidation, LosoFoldAcceptsZeroPooledSampleCaps)
{
    // Nested LOSO (the only split) uses every window of the speaker-disjoint
    // partitions, not a pooled sample budget, so max_loaded_train_samples /
    // max_validation_samples = 0 is fine.
    auto cfg = valid_config();
    cfg.dataset.cv_fold = 0;
    cfg.dataset.cv_num_folds = 6;
    cfg.dataset.max_loaded_train_samples = 0;
    cfg.dataset.max_validation_samples = 0;
    EXPECT_NO_THROW(cfg.validate());
}

TEST(Meeting01ConfigValidation, RejectsAMissingCvFold)
{
    // The pooled/shuffled legacy split (cv_fold < 0) was removed 2026-09-23 -- it
    // let the same speaker/recording land in both train and validation (the
    // leakage defect a reviewer flagged as strong-reject on submission 71).
    // cv_fold is now REQUIRED, not merely optional-with-a-fallback.
    auto cfg = valid_config();
    cfg.dataset.cv_fold = -1;
    const std::string message = validation_error(cfg);
    EXPECT_NE(message.find("cv_fold"), std::string::npos);
    EXPECT_NE(message.find("must be set"), std::string::npos);
}

// The rule below spans sections. It is the kind a refactor that splits
// validation per section can silently drop: each checker sees only its own
// struct, so a relation between two structs has nowhere to live unless
// someone deliberately keeps it. check_dataset deliberately takes the whole
// config for this reason, and this test is what proves it stuck.

TEST(Meeting01ConfigValidation, CatchesAFrameSizeThatDoesNotDivideTheWindow)
{
    // A frame holds lstm_frame_size neighbouring samples of ONE simulation step, so it
    // must not straddle two steps: window_size has to be a multiple of lstm_frame_size.
    // The rule is a relation between the dataset and the model, not a fact about either.
    auto cfg = valid_config();
    cfg.dataset.window_size = 100;
    cfg.model.lstm_frame_size = 8;

    const std::string message = validation_error(cfg);
    EXPECT_NE(message.find("lstm_frame_size"), std::string::npos);
    EXPECT_NE(message.find("must divide"), std::string::npos);
}

// ---- command line (Meeting01Cli.cpp::parse_cli) ----

namespace
{

auto parse_args(std::vector<std::string> args) -> meeting01::CliOptions
{
    args.insert(args.begin(), "meeting01");
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    return meeting01::parse_cli(static_cast<int>(argv.size()), argv.data());
}

} // namespace

TEST(Meeting01Cli, ParsesTheFoldAndDatasetTheRunScriptPasses)
{
    const auto opts = parse_args(
        {"--comparative-config", "meeting01-loso.json", "--dataset", "fsdd", "--cv-fold", "2"});
    EXPECT_EQ(opts.comparative_config, "meeting01-loso.json");
    EXPECT_EQ(opts.dataset, "fsdd");
    EXPECT_EQ(opts.cv_fold, 2);
    EXPECT_TRUE(opts.cv_fold_set);
}

TEST(Meeting01Cli, RefusesAnUnknownArgument)
{
    // A misspelt option must not fall through to a full training run of the profile's own
    // fold, which would truncate that fold's events log and rewrite its result files.
    EXPECT_THROW(parse_args({"--comparative-config", "meeting01-loso.json", "--cv_fold", "3"}),
        std::runtime_error);
    // Nor may the removed --dump-reference-inputs-only, briefly suggested (never committed)
    // for giving an older run its reference inputs: that pairs references with family
    // errors computed on differently normalized windows (see kResultsFormat).
    EXPECT_THROW(
        parse_args({"--comparative-config", "meeting01-loso.json", "--dump-reference-inputs-only"}),
        std::runtime_error);
}

// The checkpoint gate (config_hash) must move with every field that can change a result.
// Until 2026-11-02 it hashed a subset that missed these, so a profile edited in one of them
// between two RESUME=1 starts reused checkpoints trained under the old value -- silently.
TEST(Meeting01ConfigHash, MovesWithEveryResultChangingField)
{
    const auto base = valid_config();
    const auto h0 = meeting01::config_hash(base);
    const std::vector<std::pair<const char*, void (*)(Meeting01Config&)>> edits = {
        {"model.snn_time_pool", [](Meeting01Config& c) { c.model.snn_time_pool = "output"; }},
        {"model.snn_decoder_neuron",
            [](Meeting01Config& c) { c.model.snn_decoder_neuron = "lif"; }},
        {"model.initializer", [](Meeting01Config& c) { c.model.initializer = "kaiming"; }},
        {"model.latent_dim", [](Meeting01Config& c) { c.model.latent_dim += 1; }},
        {"model.recurrent_time_steps",
            [](Meeting01Config& c) { c.model.recurrent_time_steps = 2; }},
        {"model.lstm_frame_size", [](Meeting01Config& c) { c.model.lstm_frame_size = 16; }},
        {"model.time_steps", [](Meeting01Config& c) { c.model.time_steps += 1; }},
        {"training.learning_rate_biophysical",
            [](Meeting01Config& c) { c.training.learning_rate_biophysical *= 2.0F; }},
        {"training.early_stop_min_skill",
            [](Meeting01Config& c) { c.training.early_stop_min_skill = 0.2F; }},
        {"ga.snn.winner_seeds", [](Meeting01Config& c) { c.evaluation.ga.snn.winner_seeds += 1; }},
        {"ga.snn.time_steps_max",
            [](Meeting01Config& c) { c.evaluation.ga.snn.time_steps_max -= 1; }},
        {"ga.lstm.mutation_prob",
            [](Meeting01Config& c) { c.evaluation.ga.lstm.mutation_prob = 0.5; }},
        {"ga.transformer.winner_seeds",
            [](Meeting01Config& c) { c.evaluation.ga.transformer.winner_seeds += 1; }},
        {"dataset.sources[].latent_dim",
            [](Meeting01Config& c) { c.dataset.sources.front().latent_dim += 1; }},
    };
    for (const auto& [name, edit] : edits)
    {
        auto cfg = base;
        edit(cfg);
        EXPECT_NE(meeting01::config_hash(cfg), h0) << name << " does not reach the hash";
    }

    // Output-only fields: changing them must NOT throw away a finished checkpoint.
    auto out_only = base;
    out_only.dataset.latex_data_dir = "/elsewhere";
    out_only.dataset.save_models = !base.dataset.save_models;
    out_only.evaluation.ga.snn.checkpoint_every_generations += 1;
    EXPECT_EQ(meeting01::config_hash(out_only), h0);
}

// --print-resolved-config and the split manifest carry the resolved configuration; the
// fields the paper's configuration macros are generated from must be in it.
TEST(Meeting01ConfigHash, ResolvedConfigJsonCarriesThePaperFields)
{
    const auto j = meeting01::resolved_config_json(valid_config());
    for (const char* path : {"/training/epochs",
             "/training/early_stop_patience",
             "/training/early_stop_min_skill",
             "/training/learning_rate",
             "/training/learning_rate_biophysical",
             "/model/snn_time_pool",
             "/model/initializer",
             "/model/recurrent_time_steps",
             "/model/lstm_frame_size",
             "/dataset/loso_max_train_windows",
             "/evaluation/ga/snn/time_steps_min",
             "/evaluation/ga/snn/time_steps_max",
             "/evaluation/ga/lstm/max_hidden",
             "/evaluation/ga/transformer/head_choices",
             "/experiment/repeats"})
        EXPECT_TRUE(j.contains(nlohmann::json::json_pointer(path))) << path;
    EXPECT_FALSE(j["dataset"].contains("latex_data_dir"));
}
