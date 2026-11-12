/**
 * @file src/experiments/thesis/tests/thesis_cli_gtest.cpp
 * @brief The thesis command line accepts --config and --repeat, and refuses everything else.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "lib/include/ThesisCli.hpp"

namespace
{
auto parse(std::vector<const char*> args) -> thesis::ThesisCli
{
    args.insert(args.begin(), "thesis");
    return thesis::parse_cli(static_cast<int>(args.size()), args.data());
}
} // namespace

TEST(ThesisCli, ConfigAloneRunsEveryRepetition)
{
    const auto cli = parse({"--config", "p.json"});
    EXPECT_EQ(cli.config_path, "p.json");
    EXPECT_FALSE(cli.repeat.has_value());
    const auto r = thesis::repeat_range(cli, 3);
    EXPECT_EQ(r.first, 0);
    EXPECT_EQ(r.last, 3);
}

TEST(ThesisCli, RepeatSelectsOneRepetitionInEitherOrder)
{
    for (const auto& args : {std::vector<const char*>{"--config", "p.json", "--repeat", "2"},
             std::vector<const char*>{"--repeat", "2", "--config", "p.json"}})
    {
        const auto cli = parse(args);
        ASSERT_TRUE(cli.repeat.has_value());
        const auto r = thesis::repeat_range(cli, 3);
        EXPECT_EQ(r.first, 2);
        EXPECT_EQ(r.last, 3);
    }
}

TEST(ThesisCli, RepeatZeroOfASingleRepetitionProfileIsThatRepetition)
{
    const auto r = thesis::repeat_range(parse({"--config", "p.json", "--repeat", "0"}), 1);
    EXPECT_EQ(r.first, 0);
    EXPECT_EQ(r.last, 1);
}

TEST(ThesisCli, RepeatOutsideTheProfileThrows)
{
    EXPECT_THROW(thesis::repeat_range(parse({"--config", "p.json", "--repeat", "3"}), 3),
        std::invalid_argument);
}

TEST(ThesisCli, AnythingElseThrows)
{
    // The old parser scanned for --config and ignored the rest: "--repat 1" ran all repetitions.
    const std::vector<std::vector<const char*>> bad = {
        {},                                                       // no --config
        {"--config"},                                             // no value
        {"--config", "p.json", "--repat", "1"},                   // misspelt option
        {"--config", "p.json", "extra"},                          // stray token
        {"--config", "a.json", "--config", "b.json"},             // twice
        {"--config", ""},                                         // empty path
        {"--config", "p.json", "--repeat"},                       // no value
        {"--config", "p.json", "--repeat", "-1"},                 // negative
        {"--config", "p.json", "--repeat", "1.5"},                // not an integer
        {"--config", "p.json", "--repeat", "x"},                  // not a number
        {"--config", "p.json", "--repeat", "1", "--repeat", "2"}, // twice
    };
    for (const auto& args : bad)
        EXPECT_THROW(parse(args), std::invalid_argument) << "accepted " << args.size() << " args";
}
