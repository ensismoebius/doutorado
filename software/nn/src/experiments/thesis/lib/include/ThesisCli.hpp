/**
 * @file src/experiments/thesis/lib/include/ThesisCli.hpp
 * @brief The thesis binary's command line: exactly what it accepts, and nothing else.
 */
#pragma once

#include <optional>
#include <stdexcept>
#include <string>

namespace thesis
{

inline constexpr const char* kThesisUsage =
    "usage: thesis --config <profile.json> [--repeat <k>]\n"
    "  --repeat k  run only repetition k (0 <= k < experiment.repeats), with the seed and\n"
    "              the _rep<k> output names the full run gives it";

struct ThesisCli
{
    std::string config_path;
    /// Set: run only this repetition. Unset: run every repetition, in order.
    std::optional<int> repeat;
};

/**
 * Why `--repeat` exists. Repetitions of one profile are independent trainings (each has its
 * own seed), and the program runs fastest on ONE thread: on GridUnesp, one Phase 01 epoch took
 * 2 s at 1 thread and 53 s at 4 (job 5322998). The way to use more CPUs is therefore more
 * processes. One process per (profile, repetition) instead of per profile runs Phase 01's
 * 32 x 3 trainings as 96 processes, finishing in the time of one repetition instead of three.
 * The output of `--repeat k` is the repetition k of the full run, file names and seed included.
 *
 * Strict: an unknown argument, a repeated option, a missing value or a repetition that is not
 * a plain non-negative integer throws std::invalid_argument (the old parser ignored anything
 * it did not recognise, so `--repat 1` ran all repetitions without a word).
 */
inline auto parse_cli(int argc, const char* const argv[]) -> ThesisCli
{
    ThesisCli cli;
    bool have_config = false;
    auto refuse = [](const std::string& why)
    { throw std::invalid_argument("thesis: " + why + "\n" + kThesisUsage); };

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg != "--config" && arg != "--repeat") refuse("unknown argument \"" + arg + "\"");
        if (i + 1 >= argc) refuse(arg + " needs a value");
        const std::string value = argv[++i];
        if (arg == "--config")
        {
            if (have_config) refuse("--config given twice");
            if (value.empty()) refuse("--config is empty");
            cli.config_path = value;
            have_config = true;
        }
        else
        {
            if (cli.repeat.has_value()) refuse("--repeat given twice");
            const bool digits = !value.empty() && value.size() <= 9 &&
                                value.find_first_not_of("0123456789") == std::string::npos;
            if (!digits) refuse("--repeat needs a non-negative integer, got \"" + value + "\"");
            cli.repeat = std::stoi(value);
        }
    }
    if (!have_config) refuse("--config is required");
    return cli;
}

/// The repetitions to run, [first, last): all of them, or the one `--repeat` names.
struct RepeatRange
{
    int first;
    int last;
};

inline auto repeat_range(const ThesisCli& cli, int repeats) -> RepeatRange
{
    if (repeats < 1)
        throw std::invalid_argument("thesis: experiment.repeats is " + std::to_string(repeats));
    if (!cli.repeat.has_value()) return {0, repeats};
    if (*cli.repeat >= repeats)
        throw std::invalid_argument(
            "thesis: --repeat " + std::to_string(*cli.repeat) + " is outside [0, " +
            std::to_string(repeats) +
            ") for this profile (experiment.repeats = " + std::to_string(repeats) + ")");
    return {*cli.repeat, *cli.repeat + 1};
}

} // namespace thesis
