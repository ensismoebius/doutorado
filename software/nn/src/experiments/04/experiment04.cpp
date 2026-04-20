/**
 * @file src/experiments/04/experiment04.cpp
 * @brief Lightweight launcher for Experiment04 CLI.
 *
 * This translation unit implements the small main() wrapper used to parse CLI
 * parameters and invoke the `LstmAutoencoderExperiment` driver. It is intentionally
 * thin; the experiment implementation lives under `lib/src/` and the runtime
 * configuration is declared in `lib/include/experiment04.hpp`.
 */
#include "lib/include/experiment04.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>

#include "nn/logging/Logger.hpp"
#include "nn/logging/StreamRedirector.hpp"

using nn::logging::Level;
using nn::logging::Logger;
using nn::logging::StreamRedirector;
using std::cerr;
using std::cout;
using std::exception;

namespace
{
auto parse_log_level_from_env() -> Level
{
    const char* value = std::getenv("NN_EXPERIMENT04_LOG_LEVEL");
    if (value == nullptr) return Level::Info;

    const std::string_view level{value};
    if (level == "error") return Level::Error;
    if (level == "warn" || level == "warning") return Level::Warn;
    if (level == "debug") return Level::Debug;
    return Level::Info;
}

auto has_help_flag(int argc, char* argv[]) -> bool
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i] ? argv[i] : "";
        if (arg == "-h" || arg == "--help" || arg == "--help-all")
        {
            return true;
        }
    }

    return false;
}
} // namespace

auto main(int argc, char* argv[]) -> int
{
    const bool wants_help = has_help_flag(argc, argv);

    Logger::instance().set_level(parse_log_level_from_env());
    std::unique_ptr<StreamRedirector> redirect;
    if (!wants_help)
    {
        redirect = std::make_unique<StreamRedirector>(true, true);
    }

    LstmAutoencoderExperiment experiment;
    return experiment.run(argc, argv);
}
