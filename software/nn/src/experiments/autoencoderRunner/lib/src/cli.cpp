/**
 * @file src/experiments/autoencoderRunner/lib/src/cli.cpp
 * @brief CLI parsing and option handling for AutoencoderRunner.
 *
 * Uses CLI11 to declare and parse command-line options for the experiment
 * binary. This file adapts values into the `Config` structure used by
 * `AutoencoderRunner`.
 */

#include "../include/cli.hpp"

#include <cstdlib>
#include <stdexcept>

#include "../include/ProfileLoader.hpp"
#include "CLI/CLI.hpp"
#include "data_loaders/options/SamplerOptionResolution.hpp"

using CLI::App;

namespace
{
constexpr std::string_view kDefaultProfileName = "default";

auto has_help_flag(int argc, char* argv[]) -> bool
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";
        if (arg == "-h" || arg == "--help" || arg == "--help-all")
        {
            return true;
        }
    }

    return false;
}

auto parse_profile_name_from_argv(int argc, char* argv[], const std::string& fallback)
    -> std::string
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i] ? argv[i] : "";
        if (arg == "--profile" && i + 1 < argc && argv[i + 1])
        {
            return argv[i + 1];
        }

        const std::string prefix = "--profile=";
        if (arg.rfind(prefix, 0) == 0)
        {
            return arg.substr(prefix.size());
        }
    }

    return fallback;
}

auto resolve_profile_name(const std::string& profile_name) -> std::string
{
    return profile_name.empty() ? std::string(kDefaultProfileName) : profile_name;
}

auto normalize_device_token(const std::string& token) -> std::string
{
    const auto normalized = CLI::detail::to_lower(token);
    if (normalized == "cpu" || normalized == "opencl")
    {
        return normalized;
    }

    throw std::invalid_argument("Unsupported device token: " + token);
}
} // namespace

auto parseCliParams(int argc, char* argv[], const Config& default_config) -> Config
{
    Config config = default_config;
    const bool has_help = has_help_flag(argc, argv);

    std::string profile =
        resolve_profile_name(parse_profile_name_from_argv(argc, argv, default_config.profile_name));

    config.profile_name = profile;

    App app(
        "AutoencoderRunner profile-only launcher. All configuration must come from profile files.");

    app.add_option("--profile", profile, "Configuration profile name (JSON file stem)")
        ->default_val(config.profile_name);

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
        std::exit(app.exit(e));
    }

    if (!has_help)
    {
        std::string profile_error;
        if (!autoencoderRunner::load_profile_to_config(profile, config, profile_error))
        {
            throw std::runtime_error("Failed to load profile '" + profile + "': " + profile_error);
        }
    }

    config.profile_name = profile;
    config.device = normalize_device_token(config.device);
    config.sampler_default_type = normalizeSamplerTypeToken(config.sampler_default_type);

    if (!config.sampler_shuffle_seed.has_value())
    {
        throw std::runtime_error(
            "RNG policy violation: sampler_shuffle_seed must be defined in profile.");
    }
    if (config.kfold_enabled && config.kfold_shuffle && !config.kfold_seed.has_value())
    {
        throw std::runtime_error(
            "RNG policy violation: kfold_seed must be defined when k-fold shuffle is enabled.");
    }

    config.sampler_resolved_options = resolveDefaultSamplerOptions(SamplerOptionSelection{
        .sampler_type = config.sampler_default_type,
        .shuffle = config.sampler_shuffle_samples,
        .seed = config.sampler_shuffle_seed,
        .weights = config.sampler_weights,
        .weighted_num_samples = config.sampler_weighted_num_samples,
        .distributed_num_replicas = config.sampler_distributed_num_replicas,
        .distributed_rank = config.sampler_distributed_rank,
        .distributed_shuffle = config.sampler_distributed_shuffle,
        .distributed_drop_last = config.sampler_distributed_drop_last,
    });
    return config;
}