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
    return should_run_comparative_from_cli(argc, argv);
}

} // namespace lstm_autoencoder_experiment
