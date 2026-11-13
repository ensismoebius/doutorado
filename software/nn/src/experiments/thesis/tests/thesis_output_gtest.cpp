// Output-artifact contract: summary.json must be self-describing. It carries the declared
// profile under "config" — every key the run read and nothing else (ThesisConfig refuses
// undeclared keys the run reads and declared keys it ignores) — plus the dataset
// composition, so a result file can be interpreted without the source profile.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "../lib/include/ThesisConfig.hpp"
#include "../lib/include/ThesisOutput.hpp"
#include "ThesisTestConfig.hpp"
#include "nlohmann/json.hpp"

using thesis::ThesisConfig;

namespace
{
nlohmann::json read_summary(const std::string& dir, const std::string& tag)
{
    const std::string path = dir + "/e05_" + tag + "_summary.json";
    std::ifstream f(path);
    if (!f)
    {
        throw std::runtime_error("summary not written: " + path);
    }
    nlohmann::json j;
    f >> j;
    return j;
}

ThesisConfig base_cfg(const std::string& tag)
{
    ThesisConfig cfg = thesis::test::phase00("eeg");
    cfg.experiment.run_tag = tag;
    cfg.experiment.seed = 7u;
    return cfg;
}

// Written under the test framework's temporary directory and removed after reading, so a run
// leaves nothing behind (a relative "./e05_output_test" used to appear in whatever directory
// the binary was started from).
nlohmann::json write_and_read(const ThesisConfig& cfg, const std::string& tag)
{
    cfg.validate(); // the config is one a real profile could declare
    const std::filesystem::path dir =
        std::filesystem::path(::testing::TempDir()) / ("thesis_output_gtest_" + tag);
    std::filesystem::create_directories(dir);
    thesis::write_summary_json(dir.string(), tag, cfg, {}, {}, 15, 11, 1974, 0u);
    auto j = read_summary(dir.string(), tag);
    std::filesystem::remove_all(dir);
    return j;
}
} // namespace

TEST(ThesisOutput, HandcraftedSummaryRecordsExtractionConfig)
{
    auto cfg = base_cfg("t_hc");
    cfg.dataset.modality = "voice"; // bark is refused for EEG
    cfg.dataset.eeg_condition = thesis::Required<std::string>{"dataset.eeg_condition"};
    thesis::test::set_handcrafted(cfg);
    cfg.feature_extraction.handcrafted.wavelet = "daub12";
    cfg.feature_extraction.handcrafted.scale = "bark";
    cfg.feature_extraction.handcrafted.cepstral = true;
    // Cepstral coefficients replace the band energies; "energy" would be refused.
    cfg.feature_extraction.handcrafted.descriptors =
        std::vector<std::string>{"zcr", "entropy", "teager"};

    const auto j = write_and_read(cfg, "t_hc");
    const auto& fe = j["config"]["feature_extraction"];
    EXPECT_EQ(fe["handcrafted"]["wavelet"], "daub12");
    EXPECT_EQ(fe["handcrafted"]["scale"], "bark");
    EXPECT_TRUE(fe["handcrafted"]["cepstral"]);
    EXPECT_FALSE(fe.contains("autoencoder"));
    EXPECT_FALSE(j["config"].contains("training")) << "a handcrafted Phase 00 run trains nothing";
}

TEST(ThesisOutput, SnnAeSummaryRecordsEncodingAndThreshold)
{
    auto cfg = base_cfg("t_snn");
    thesis::test::set_training(cfg, 3, 1e-3f, 8);
    thesis::test::set_autoencoder(cfg,
        "snn-ae",
        {"linear:64:leaky", "linear:16:identity"},
        {"linear:64:leaky", "linear:output:identity"},
        "spiketime");
    thesis::test::set_snn(cfg, "latency", 16, 0.2f, 0.0f, 0.0f, 0.0f);

    const auto j = write_and_read(cfg, "t_snn");
    const auto& ae = j["config"]["feature_extraction"]["autoencoder"];
    EXPECT_EQ(ae["model"], "snn-ae");
    EXPECT_EQ(ae["encoding"], "latency");
    EXPECT_EQ(ae["time_steps"], 16);
    EXPECT_FLOAT_EQ(ae["voltage_threshold"].get<float>(), 0.2f);
    EXPECT_FALSE(ae.contains("firing_rate_min")) << "no band without the regularizer";
}

TEST(ThesisOutput, SnnAeDirectEncodingHasNoTimeSteps)
{
    // "direct" is a single analog frame: there is no T, so the profile cannot declare one
    // and the summary records none (it used to print 1 over whatever the profile said).
    auto cfg = base_cfg("t_direct");
    thesis::test::set_training(cfg, 3, 1e-3f, 8);
    thesis::test::set_autoencoder(cfg,
        "snn-ae",
        {"linear:64:leaky", "linear:16:identity"},
        {"linear:64:leaky", "linear:output:identity"},
        "mse");
    thesis::test::set_snn(cfg, "direct", 0, 1.0f, 0.0f, 0.0f, 0.0f);

    const auto j = write_and_read(cfg, "t_direct");
    const auto& ae = j["config"]["feature_extraction"]["autoencoder"];
    EXPECT_EQ(ae["encoding"], "direct");
    EXPECT_FALSE(ae.contains("time_steps"));
}

TEST(ThesisOutput, AnnAeSummaryOmitsSnnOnlyFields)
{
    auto cfg = base_cfg("t_ann");
    thesis::test::set_training(cfg, 3, 1e-3f, 8);
    thesis::test::set_autoencoder(cfg,
        "ann-ae",
        {"linear:64:leaky", "linear:16:identity"},
        {"linear:64:leaky", "linear:output:identity"},
        "mse");

    const auto j = write_and_read(cfg, "t_ann");
    const auto& ae = j["config"]["feature_extraction"]["autoencoder"];
    EXPECT_EQ(ae["model"], "ann-ae");
    EXPECT_FALSE(ae.contains("encoding"));
    EXPECT_FALSE(ae.contains("voltage_threshold"));
}

TEST(ThesisOutput, PcaSummaryRecordsComponents)
{
    auto cfg = base_cfg("t_pca");
    cfg.feature_extraction.strategy = "pca";
    cfg.feature_extraction.pca.n_components = 16;

    const auto j = write_and_read(cfg, "t_pca");
    EXPECT_EQ(j["strategy"], "pca");
    EXPECT_EQ(j["config"]["feature_extraction"]["pca"]["n_components"], 16);
    EXPECT_FALSE(j["config"].contains("training")) << "PCA has no training";
}

TEST(ThesisOutput, SummaryRecordsTheDeclaredLearningRate)
{
    // The lr used to be optional and resolved from the optimizer; the summary then had to
    // say which (learning_rate_source). Now it is required, so the declared value is the
    // value that ran, and there is no source to report.
    auto cfg = base_cfg("t_lr");
    thesis::test::set_training(cfg, 3, 2.5e-3f, 8);
    thesis::test::set_autoencoder(cfg,
        "ann-ae",
        {"linear:64:leaky", "linear:16:identity"},
        {"linear:64:leaky", "linear:output:identity"},
        "mse");

    const auto j = write_and_read(cfg, "t_lr");
    EXPECT_FLOAT_EQ(j["config"]["training"]["learning_rate"].get<float>(), 2.5e-3f);
    EXPECT_FALSE(j["config"]["training"].contains("optimizer_momentum")) << "adam has none";
    EXPECT_FALSE(j.contains("learning_rate_source"));
}

TEST(ThesisOutput, SummaryRecordsDatasetComposition)
{
    auto cfg = base_cfg("t_ds");
    thesis::test::set_handcrafted(cfg);

    const auto j = write_and_read(cfg, "t_ds");
    EXPECT_EQ(j["dataset"]["n_subjects"], 15);
    EXPECT_EQ(j["dataset"]["n_stimuli"], 11);
    EXPECT_EQ(j["dataset"]["n_samples"], 1974);
}
