/**
 * @file src/experiments/04/experiment04.cpp
 * @brief Standalone Experiment04 entrypoint (thin main wrapper).
 */

#include <cstdlib>
#include <memory>
#include <string_view>

#include "lib/include/LstmAutoencoderExperiment.hpp"
#include "nn/logging/Logger.hpp"
#include "nn/logging/StreamRedirector.hpp"

using nn::logging::Level;
using nn::logging::Logger;
using nn::logging::StreamRedirector;

#ifndef NN_EXPERIMENT04_NO_MAIN
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
    std::unique_ptr<StreamRedirector> redirect;
    Logger::instance().set_level(parse_log_level_from_env());

    const bool wants_help = has_help_flag(argc, argv);
    if (!wants_help)
    {
        redirect = std::make_unique<StreamRedirector>(true, true);
    }

    LstmAutoencoderExperiment experiment;
    return experiment.run(argc, argv);
}
#endif
