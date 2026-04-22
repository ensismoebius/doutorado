#include "../include/Experiment04Cli.hpp"

#include <string>

#include "../include/ComparativeCli.hpp"

namespace lstm_autoencoder_experiment
{

auto should_run_comparative_from_cli(int argc, char* argv[]) -> bool
{
    return comparative_autoencoder_experiment::should_run_comparative_cli(argc, argv);
}

auto should_run_from_cli(int argc, char* argv[]) -> bool
{
    if (should_run_comparative_from_cli(argc, argv))
    {
        return true;
    }

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";
        if (has_experiment04_marker(arg) || arg == "--lstm-profile" || arg.rfind("--lstm-profile=", 0) == 0)
        {
            return true;
        }
    }

    return false;
}

} // namespace lstm_autoencoder_experiment
