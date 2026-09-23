// Profile audit: every shipping production profile must parse cleanly via
// Meeting01Config::from_nested_json AND must populate the live config
// fields with non-default values that the experiment harness will actually
// consume. Catches silent profile-key drift (e.g. a future rename moving a
// field outside the parser's lookup keys).

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../lib/include/Meeting01Config.hpp"
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
// (fsdd/audiomnist) narrower than EEG (eegmmidb/chbmit), matching the literature
// table in .wiki/Experiments/Meeting01.md's "latent_dim is fixed, not evolved"
// section. Deliberately does not run the grid itself (see the expensive-experiment
// guard) -- this only proves the config resolution a real run would depend on.
TEST(ProductionProfiles, MeetingOneLosoResolvesLatentDimPerDataset)
{
    auto cfg = load("meeting01-loso.json");
    EXPECT_EQ(cfg.dataset.resolve("fsdd").latent_dim, 16);
    EXPECT_EQ(cfg.dataset.resolve("audiomnist").latent_dim, 16);
    EXPECT_EQ(cfg.dataset.resolve("eegmmidb").latent_dim, 64);
    EXPECT_EQ(cfg.dataset.resolve("chbmit").latent_dim, 64);
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

// EVERY profile must state its simulation depth explicitly -- including the LSTM-only
// ones. `time_steps` is misleadingly named: `run_baseline` reads it too
// (Meeting01Experiment.cpp), so it sets the sequence length the LSTM/GRU/Transformer
// baselines are trained on (T * window_size / lstm_frame_size), not just the SNN's
// membrane depth. lstm-bench.json inherited the struct default silently, which changed
// what that throughput benchmark measured without a single line of the profile changing.
//
// The struct default (16) is the right value; the point is that a run's temporal
// resolution must be readable from the profile rather than from a header nobody opens.
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

        ASSERT_TRUE(j.contains("model") && j["model"].contains("time_steps"))
            << name
            << ": no model.time_steps -- it governs the baselines too, so "
               "inheriting it silently changes what the run measures";
        const int steps = j["model"]["time_steps"].get<int>();
        EXPECT_GE(steps, 2) << name
                            << ": a single step disables membrane dynamics and "
                               "spike coding entirely";
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
        "training": {"samples_per_batch": 1, "epochs": 2, "early_stop_patience": 1,
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
