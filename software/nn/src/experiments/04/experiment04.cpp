/**
 * @file src/experiments/04/experiment04.cpp
 * @brief Standalone Experiment04 entrypoint (thin main wrapper).
 */

#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "../include/ComparativeCli.hpp"
#include "../include/Experiment04Cli.hpp"
#include "../include/LstmAutoencoderExperiment.hpp"
#include "experiments/04/lib/include/LstmAutoencoderExperiment.hpp"
#include "nn/logging/Logger.hpp"
#include "nn/logging/StreamRedirector.hpp"

using nn::logging::Level;
using nn::logging::Logger;
using nn::logging::StreamRedirector;

namespace
{
/**
 * This allows users to set the log level without needing to pass command-line arguments, which is
 * especially useful when the experiment is invoked programmatically or via scripts. By default,
 * logs are redirected to the Logger, but if the user requests help via command-line flags, we
 * skip redirection to allow the help message to be printed directly to the console.
 */
auto parse_log_level_from_env() -> Level
{
    // Check the environment variable "NN_EXPERIMENT04_LOG_LEVEL" for log level configuration
    const char* value = std::getenv("NN_EXPERIMENT04_LOG_LEVEL");

    // Default to Info if the environment variable is not set or is empty
    if (value == nullptr) return Level::Info;

    // Convert the value to lowercase for case-insensitive comparison
    const std::string_view level{value};
    if (level == "error") return Level::Error;
    if (level == "warn" || level == "warning") return Level::Warn;
    if (level == "debug") return Level::Debug;

    return Level::Info;
}

/**
 * Check if any of the help flags are present in the command-line arguments.
 *
 * @param argc
 * @param argv
 * @return true
 * @return false
 */
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
    std::vector<char*> normalized_argv;
    std::vector<std::string> normalized_args;

    std::unique_ptr<StreamRedirector> redirect;
    Logger::instance().set_level(parse_log_level_from_env());

    const bool wants_help = has_help_flag(argc, argv);
    if (!wants_help)
    {
        redirect = std::make_unique<StreamRedirector>(true, true);
    }

    lstm_autoencoder_experiment::normalize_experiment04_aliases( //
        argc,                                                    //
        argv,                                                    //
        normalized_args                                          //
    );

    lstm_autoencoder_experiment::to_argv( //
        normalized_args,                  //
        normalized_argv                   //
    );

    return lstm_autoencoder_experiment::run_comparative_experiment( //
        static_cast<int>(normalized_argv.size()),                   //
        normalized_argv.data()                                      //
    );
}
