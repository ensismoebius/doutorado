// Output-artifact contract: summary.json must be self-describing — carrying
// enough feature-extraction config (handcrafted OR autoencoder, including the
// SNN-AE encoding/time_steps/voltage_threshold) and dataset composition that a
// result file can be interpreted without cross-referencing the source profile.

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <sstream>

#include "../lib/include/E05Config.hpp"
#include "../lib/include/E05Output.hpp"
#include "nlohmann/json.hpp"

using e05::E05Config;

namespace
{
nlohmann::json read_summary(const std::string& dir, const std::string& tag)
{
    std::ifstream f(dir + "/e05_" + tag + "_summary.json");
    nlohmann::json j;
    f >> j;
    return j;
}

E05Config base_cfg(const std::string& tag)
{
    E05Config cfg;
    cfg.experiment.run_tag = tag;
    cfg.experiment.seed = 7;
    cfg.dataset.modality = "eeg";
    return cfg;
}
} // namespace

TEST(E05Output, HandcraftedSummaryRecordsExtractionConfig)
{
    auto cfg = base_cfg("t_hc");
    cfg.feature_extraction.strategy = "handcrafted";
    cfg.feature_extraction.handcrafted.wavelet = "daub12";
    cfg.feature_extraction.handcrafted.scale = "bark";
    cfg.feature_extraction.handcrafted.cepstral = true;

    e05::write_summary_json("./e05_output_test", "t_hc", cfg, {}, {}, 15, 11, 1974);
    auto j = read_summary("./e05_output_test", "t_hc");

    EXPECT_EQ(j["handcrafted"]["wavelet"], "daub12");
    EXPECT_EQ(j["handcrafted"]["scale"], "bark");
    EXPECT_TRUE(j["handcrafted"]["cepstral"]);
    EXPECT_FALSE(j.contains("autoencoder"));
}

TEST(E05Output, SnnAeSummaryRecordsEncodingAndThreshold)
{
    auto cfg = base_cfg("t_snn");
    cfg.feature_extraction.strategy = "autoencoder";
    cfg.feature_extraction.autoencoder.model = "snn-ae";
    cfg.feature_extraction.autoencoder.encoding = "latency";
    cfg.feature_extraction.autoencoder.time_steps = 16;
    cfg.feature_extraction.autoencoder.voltage_threshold = 0.2f;

    e05::write_summary_json("./e05_output_test", "t_snn", cfg, {}, {}, 15, 11, 1974);
    auto j = read_summary("./e05_output_test", "t_snn");

    EXPECT_EQ(j["autoencoder"]["model"], "snn-ae");
    EXPECT_EQ(j["autoencoder"]["encoding"], "latency");
    EXPECT_EQ(j["autoencoder"]["time_steps"], 16);
    EXPECT_FLOAT_EQ(j["autoencoder"]["voltage_threshold"].get<float>(), 0.2f);
}

TEST(E05Output, SnnAeDirectEncodingForcesTimeStepsToOne)
{
    // "direct" is a single analog frame regardless of the profile's time_steps
    // value — the summary must record what actually ran (1), not the config
    // field verbatim, so results stay interpretable in isolation.
    auto cfg = base_cfg("t_direct");
    cfg.feature_extraction.strategy = "autoencoder";
    cfg.feature_extraction.autoencoder.model = "snn-ae";
    cfg.feature_extraction.autoencoder.encoding = "direct";
    cfg.feature_extraction.autoencoder.time_steps = 16; // ignored for direct

    e05::write_summary_json("./e05_output_test", "t_direct", cfg, {}, {}, 15, 11, 1974);
    auto j = read_summary("./e05_output_test", "t_direct");

    EXPECT_EQ(j["autoencoder"]["encoding"], "direct");
    EXPECT_EQ(j["autoencoder"]["time_steps"], 1);
}

TEST(E05Output, AnnAeSummaryOmitsSnnOnlyFields)
{
    auto cfg = base_cfg("t_ann");
    cfg.feature_extraction.strategy = "autoencoder";
    cfg.feature_extraction.autoencoder.model = "ann-ae";

    e05::write_summary_json("./e05_output_test", "t_ann", cfg, {}, {}, 15, 11, 1974);
    auto j = read_summary("./e05_output_test", "t_ann");

    EXPECT_EQ(j["autoencoder"]["model"], "ann-ae");
    EXPECT_FALSE(j["autoencoder"].contains("encoding"));
    EXPECT_FALSE(j["autoencoder"].contains("voltage_threshold"));
}

TEST(E05Output, SummaryRecordsDatasetComposition)
{
    auto cfg = base_cfg("t_ds");
    cfg.feature_extraction.strategy = "handcrafted";

    e05::write_summary_json("./e05_output_test", "t_ds", cfg, {}, {}, 15, 11, 1974);
    auto j = read_summary("./e05_output_test", "t_ds");

    EXPECT_EQ(j["dataset"]["n_subjects"], 15);
    EXPECT_EQ(j["dataset"]["n_stimuli"], 11);
    EXPECT_EQ(j["dataset"]["n_samples"], 1974);
}
