/**
 * @file src/experiments/04/lib/src/experiment04.cpp
 * @brief Experiment04 CLI ownership and comparative delegation.
 */

#include "experiment04.hpp"

#include <string>
#include <vector>

namespace lstm_autoencoder_experiment
{
namespace
{
auto has_experiment04_marker(const std::string& arg) -> bool
{
    return arg == "--experiment04" || arg == "--lstm-autoencoder" || arg == "--experiment=04" ||
           arg == "--experiment=experiment04" || arg == "--experiment=lstm" ||
           arg == "--experiment=lstm-autoencoder";
}

void normalize_experiment04_aliases(int argc, char* argv[], std::vector<std::string>& args)
{
    args.clear();
    args.reserve(static_cast<std::size_t>(argc));

    if (argc > 0)
    {
        args.emplace_back(argv[0] ? argv[0] : "experiment04");
    }
    else
    {
        args.emplace_back("experiment04");
    }

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";

        if (has_experiment04_marker(arg))
        {
            args.emplace_back("--comparative");
            continue;
        }

        if (arg == "--lstm-profile" || arg == "--config")
        {
            args.emplace_back("--comparative-config");
            if (i + 1 < argc)
            {
                args.emplace_back(argv[++i] ? argv[i] : "");
            }
            continue;
        }

        if (arg.rfind("--lstm-profile=", 0) == 0)
        {
            const std::string value = arg.substr(std::string("--lstm-profile=").size());
            args.emplace_back("--comparative-config=" + value);
            continue;
        }

        args.push_back(arg);
    }
}

void to_argv(std::vector<std::string>& args, std::vector<char*>& argv_out)
{
    argv_out.clear();
    argv_out.reserve(args.size());
    for (std::string& arg : args)
    {
        argv_out.push_back(arg.data());
    }
}
} // namespace

auto should_run_from_cli(int argc, char* argv[]) -> bool
{
    if (should_run_comparative_from_cli(argc, argv))
    {
        return true;
    }

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";
        if (has_experiment04_marker(arg) || arg == "--lstm-profile" ||
            arg.rfind("--lstm-profile=", 0) == 0)
        {
            return true;
        }
    }

    return false;
}

} // namespace lstm_autoencoder_experiment

auto LstmAutoencoderExperiment::run(int argc, char* argv[]) -> int
{
    std::vector<std::string> normalized_args;
    lstm_autoencoder_experiment::normalize_experiment04_aliases(argc, argv, normalized_args);

    std::vector<char*> normalized_argv;
    lstm_autoencoder_experiment::to_argv(normalized_args, normalized_argv);

    return lstm_autoencoder_experiment::run_comparative_experiment(
        static_cast<int>(normalized_argv.size()), normalized_argv.data());
}
