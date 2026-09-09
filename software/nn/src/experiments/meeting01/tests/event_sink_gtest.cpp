// event_sink_gtest.cpp — the JSONL event sink that feeds
// scripts/pipeline/meeting01/monitor.py. Schema helpers + file round-trip. No
// training is exercised here; the full lifecycle (session/fold/config/epoch) is
// covered by a tiny end-to-end run in the plan's verification section.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "Meeting01Events.hpp"
#include "nlohmann/json.hpp"

using meeting01::ExperimentEvents;
using meeting01::jnum;
using meeting01::make_config_id;

TEST(EventSink, JnumMapsNonFiniteToNull)
{
    EXPECT_TRUE(jnum(std::numeric_limits<double>::quiet_NaN()).is_null());
    EXPECT_TRUE(jnum(std::numeric_limits<double>::infinity()).is_null());
    EXPECT_TRUE(jnum(-std::numeric_limits<double>::infinity()).is_null());
    EXPECT_DOUBLE_EQ(jnum(0.125).get<double>(), 0.125);
    EXPECT_DOUBLE_EQ(jnum(-3.0).get<double>(), -3.0);
}

TEST(EventSink, ConfigIdIsStableAndRoleAware)
{
    const std::string snn = make_config_id("snn-ae", "poisson", "snn_sweep", 1.5F, 0.9F, 42U, 3);
    EXPECT_EQ(snn, make_config_id("snn-ae", "poisson", "snn_sweep", 1.5F, 0.9F, 42U, 3));
    EXPECT_NE(snn, make_config_id("snn-ae", "poisson", "snn_final", 1.5F, 0.9F, 42U, 3));
    EXPECT_NE(snn, make_config_id("snn-ae", "poisson", "snn_sweep", 1.0F, 0.9F, 42U, 3));

    // Baselines ignore v_th / alpha (they have none).
    const std::string base = make_config_id("lstm-ae", "direct", "baseline", 0.0F, 0.0F, 42U, 1);
    EXPECT_EQ(base, "lstm-ae_direct_seed42_run1");
}

TEST(EventSink, EmitWritesOneMergedJsonLinePerEvent)
{
    const auto path = std::filesystem::temp_directory_path() /
                      ("meeting01_events_test_" + std::to_string(::getpid()) + ".jsonl");
    std::filesystem::remove(path);

    auto& ev = ExperimentEvents::instance();
    ev.open(path.string());
    ASSERT_TRUE(ev.is_open());
    ev.set_common({{"v", 1}, {"run_tag", "t"}, {"dataset", "fsdd"}, {"fold", 0}});

    ev.emit("session_begin", {{"seed", 42}, {"backend", "xtensor"}});
    ev.emit("epoch",
        {{"config_id", "lstm-ae_direct_seed42_run1"},
            {"epoch", 1},
            {"train_loss", jnum(0.5)},
            {"val_loss", jnum(std::numeric_limits<double>::quiet_NaN())}});
    ev.close();

    std::ifstream in(path);
    std::vector<nlohmann::json> lines;
    for (std::string raw; std::getline(in, raw);)
        if (!raw.empty()) lines.push_back(nlohmann::json::parse(raw));
    ASSERT_EQ(lines.size(), 2U);

    for (const auto& l : lines)
    {
        EXPECT_EQ(l.at("v"), 1);
        EXPECT_EQ(l.at("run_tag"), "t");
        EXPECT_EQ(l.at("dataset"), "fsdd");
        EXPECT_EQ(l.at("fold"), 0);
        EXPECT_TRUE(l.contains("ts_unix"));
        EXPECT_TRUE(l.contains("type"));
    }
    EXPECT_EQ(lines[0].at("type"), "session_begin");
    EXPECT_EQ(lines[0].at("seed"), 42);
    EXPECT_EQ(lines[1].at("type"), "epoch");
    EXPECT_DOUBLE_EQ(lines[1].at("train_loss").get<double>(), 0.5);
    EXPECT_TRUE(lines[1].at("val_loss").is_null());
    EXPECT_GE(lines[1].at("ts_unix").get<double>(), lines[0].at("ts_unix").get<double>());

    std::filesystem::remove(path);
}

TEST(EventSink, DisabledSinkIsANoOpAndPendingContextRoundTrips)
{
    auto& ev = ExperimentEvents::instance();
    ev.open(""); // empty path → sink stays closed
    EXPECT_FALSE(ev.is_open());
    ev.emit("epoch", {{"epoch", 7}}); // must not throw / must not create a file

    meeting01::EventContext ctx;
    ctx.config_id = "cid";
    ctx.model = "gru-ae";
    ctx.seed = 44U;
    ctx.max_epochs = 30;
    ev.set_pending_context(ctx);
    const auto got = ev.pending_context();
    EXPECT_EQ(got.config_id, "cid");
    EXPECT_EQ(got.model, "gru-ae");
    EXPECT_EQ(got.seed, 44U);
    EXPECT_EQ(got.max_epochs, 30);
}
