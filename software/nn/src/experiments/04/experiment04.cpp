// TODO Profiles must contain the specification of two nn: LSTM and SNN

/**
 * @file src/experiments/04/experiment04.cpp
 * @brief Standalone Experiment04 entrypoint (thin main wrapper).
 */

#include <cstdlib>
#include <string>
#include <string_view>

#include "../include/Experiment04Cli.hpp"
#include "experiments/04/lib/include/LstmAutoencoderExperiment.hpp"
#include "logging/Logger.hpp"

using nn::logging::Level;
using nn::logging::Logger;

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

    if (level == "error")
    {
        return Level::Error;
    }
    if (level == "warn" || level == "warning")
    {
        return Level::Warn;
    }
    if (level == "debug")
    {
        return Level::Debug;
    }

    return Level::Info;
}

} // namespace

auto main(int argc, char* argv[]) -> int
{
    try
    {
        Logger::instance().set_level(parse_log_level_from_env());

        return lstm_autoencoder_experiment::run_comparative_experiment(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error\n";
        return 1;
    }
}
