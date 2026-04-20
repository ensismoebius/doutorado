/**
 * @file src/experiments/03/experiment03.cpp
 * @brief Lightweight launcher for Experiment03 CLI.
 *
 * This translation unit implements the small main() wrapper used to parse CLI
 * parameters and invoke the `Experiment03` driver. It is intentionally thin;
 * the experiment implementation lives under `lib/src/` and the runtime
 * configuration is declared in `lib/include/experiment03.hpp`.
 */
#include "lib/include/experiment03.hpp"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>

#include "../04/lib/include/experiment04.hpp"
#include "lib/include/ProfileLoader.hpp"
#include "lib/include/cli.hpp"
#include "nn/logging/Logger.hpp"
#include "nn/logging/StreamRedirector.hpp"

using nn::logging::Level;
using nn::logging::Logger;
using nn::logging::StreamRedirector;
using std::cerr;
using std::cout;
using std::exception;
using std::make_shared;
using std::size_t;

namespace
{
auto parse_log_level_from_env() -> Level
{
    const char* value = std::getenv("NN_EXPERIMENT03_LOG_LEVEL");
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
    const bool run_experiment04 = lstm_autoencoder_experiment::should_run_from_cli(argc, argv);
    const bool wants_help = has_help_flag(argc, argv);

    if (run_experiment04)
    {
        Logger::instance().set_level(parse_log_level_from_env());
        std::unique_ptr<StreamRedirector> redirect;
        if (!wants_help)
        {
            redirect = std::make_unique<StreamRedirector>(true, true);
        }

        LstmAutoencoderExperiment experiment;
        return experiment.run(argc, argv);
    }

    Config profile_defaults{};
    std::string profile_error;
    if (!experiment03::load_profile_to_config("default", profile_defaults, profile_error))
    {
        cerr << "Failed to load default profile: " << profile_error << '\n';
        return 1;
    }

    Config config = parseCliParams(argc, argv, profile_defaults);
    Logger::instance().set_level(parse_log_level_from_env());
    std::unique_ptr<StreamRedirector> redirect;
    if (!wants_help)
    {
        redirect = std::make_unique<StreamRedirector>(true, true);
    }

    Experiment03 experiment(config);
    return experiment.run();
}
