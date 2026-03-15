#include "../include/cli.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace
{

void toLowerAsciiInPlace(std::string& value)
{
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
}

auto inputModeToCliToken(Protocol101117InputMode mode) -> std::string
{
    switch (mode)
    {
        case Protocol101117InputMode::Concatenated:
            return "concatenated";
        case Protocol101117InputMode::EegOnly:
            return "eeg-only";
        case Protocol101117InputMode::AudioOnly:
            return "audio-only";
    }

    throw std::runtime_error("Unsupported input mode enum value");
}

auto parseInputModeToken(const std::string& token) -> Protocol101117InputMode
{
    if (token == "concatenated")
    {
        return Protocol101117InputMode::Concatenated;
    }

    if (token == "eeg-only")
    {
        return Protocol101117InputMode::EegOnly;
    }

    if (token == "audio-only")
    {
        return Protocol101117InputMode::AudioOnly;
    }

    throw std::runtime_error("Unknown input mode: " + token);
}

auto resolveSamplerOptions(const Config& config) -> DataLoader::DefaultSamplerOptions
{
    DataLoader::DefaultSamplerOptions options{};
    options.seed = config.seed;

    if (config.sampler_type.empty())
    {
        options.type = config.shuffle ? DataLoader::DefaultSamplerType::Random
                                      : DataLoader::DefaultSamplerType::Sequential;
        return options;
    }

    if (config.sampler_type == "sequential")
    {
        options.type = DataLoader::DefaultSamplerType::Sequential;
        return options;
    }

    if (config.sampler_type == "random")
    {
        options.type = DataLoader::DefaultSamplerType::Random;
        return options;
    }

    if (config.sampler_type == "weighted")
    {
        options.type = DataLoader::DefaultSamplerType::WeightedRandom;
        options.weights = config.sampler_weights;
        options.weighted_num_samples = config.weighted_num_samples;
        return options;
    }

    if (config.sampler_type == "distributed")
    {
        options.type = DataLoader::DefaultSamplerType::Distributed;
        options.num_replicas = config.distributed_num_replicas;
        options.rank = config.distributed_rank;
        options.distributed_shuffle = config.distributed_shuffle;
        options.distributed_drop_last = config.distributed_drop_last;
        return options;
    }

    throw std::runtime_error("Unknown sampler type: " + config.sampler_type);
}

} // namespace

auto parseCliParams(int argc, char* argv[], const Config& default_config) -> Config
{
    Config config = default_config;
    std::string input_mode_token = inputModeToCliToken(default_config.input_mode);

    App app("PyTorch-style loader pipeline for 10.1117 EEG+Audio dataset.");

    app.add_option("--dataset-root", config.dataset_root, "Path containing subjects dir")
        ->expected(1)
        ->check(CLI::ExistingDirectory)
        ->default_val(default_config.dataset_root);

    app.add_option(                                                                     //
           "--subject",                                                                 //
           config.subject_regex_pattern,                                                //
           "Subject regex filter pattern (regex should contain a group for subject id)" //
           )
        ->expected(1)
        ->default_val(default_config.subject_regex_pattern);

    app.add_option("--batch-size", config.batch_size, "Mini-batch size")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.batch_size);

    app.add_option("--max-batches", config.max_batches, "Max batches to iterate in this demo")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.max_batches);

    app.add_option(          //
           "--input-mode",   //
           input_mode_token, //
           "Dataset input mode: concatenated|eeg-only|audio-only")
        ->check(CLI::IsMember(                          //
            {"concatenated", "eeg-only", "audio-only"}, //
            CLI::ignore_case))
        ->default_val(inputModeToCliToken(default_config.input_mode));

    app.add_option(
           "--seed", config.seed, "Deterministic seed for shuffling (ignored if --no-shuffle)")
        ->expected(1)
        ->check(CLI::NonNegativeNumber)
        ->default_val(default_config.seed.value());

    app.add_flag(                            //
           "--shuffle,!--no-shuffle",        //
           config.shuffle,                   //
           "Shuffle samples before batching" //
           )
        ->default_val(default_config.shuffle);

    auto* sampler_type_option = app.add_option(                        //
        "--sampler-type",                                              //
        config.sampler_type,                                           //
        "Default sampler type: sequential|random|weighted|distributed" //
    );
    sampler_type_option->check(CLI::IsMember(                //
        {"sequential", "random", "weighted", "distributed"}, //
        CLI::ignore_case));

    // Keep sampler type unset by default so --shuffle/--no-shuffle remains
    // the default behavior unless the user explicitly selects a sampler.
    if (!default_config.sampler_type.empty())
    {
        sampler_type_option->default_val(default_config.sampler_type);
    }

    app.add_option( //
           "--sampler-weights",
           config.sampler_weights,
           "Comma-separated weights for weighted sampler (e.g. 0.1,0.2,0.7)")
        ->delimiter(',');

    app.add_option( //
           "--weighted-num-samples",
           config.weighted_num_samples,
           "Number of sampled indices per epoch for weighted sampler")
        ->check(CLI::PositiveNumber);

    app.add_option( //
           "--distributed-num-replicas",
           config.distributed_num_replicas,
           "Total number of distributed replicas")
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.distributed_num_replicas);

    app.add_option( //
           "--distributed-rank",
           config.distributed_rank,
           "Current distributed rank")
        ->check(CLI::NonNegativeNumber)
        ->default_val(default_config.distributed_rank);

    app.add_flag( //
           "--distributed-shuffle,!--distributed-no-shuffle",
           config.distributed_shuffle,
           "Shuffle globally before distributed partition")
        ->default_val(default_config.distributed_shuffle);

    app.add_flag( //
           "--distributed-drop-last,!--distributed-no-drop-last",
           config.distributed_drop_last,
           "Drop tail to make dataset divisible by replicas")
        ->default_val(default_config.distributed_drop_last);

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
        app.exit(e);
    }

    toLowerAsciiInPlace(config.sampler_type);
    toLowerAsciiInPlace(input_mode_token);

    config.input_mode = parseInputModeToken(input_mode_token);

    config.sampler_options = resolveSamplerOptions(config);
    return config;
}