#include "../include/cli.hpp"

#include "nn/dataLoaders/10.1117/InputModeCodec.hpp"
#include "nn/dataLoaders/SamplerOptionResolution.hpp"

auto parseCliParams(int argc, char* argv[], const Config& default_config) -> Config
{
    Config config = default_config;
    std::string input_mode_token = protocol101117InputModeToToken(default_config.input_mode);

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

    const auto input_mode_tokens = supportedProtocol101117InputModeTokens();
    app.add_option(          //
           "--input-mode",   //
           input_mode_token, //
           "Dataset input mode: concatenated|eeg-only|audio-only")
        ->check(CLI::IsMember(input_mode_tokens, CLI::ignore_case))
        ->default_val(protocol101117InputModeToToken(default_config.input_mode));

    app.add_option("--lookahead", config.lookahead, "Number of batches to prefetch in background")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.lookahead);

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

    config.sampler_type = normalizeSamplerTypeToken(config.sampler_type);

    config.input_mode = parseProtocol101117InputModeToken(input_mode_token);

    config.sampler_options = resolveDefaultSamplerOptions(SamplerOptionSelection{
        .sampler_type = config.sampler_type,
        .shuffle = config.shuffle,
        .seed = config.seed,
        .weights = config.sampler_weights,
        .weighted_num_samples = config.weighted_num_samples,
        .distributed_num_replicas = config.distributed_num_replicas,
        .distributed_rank = config.distributed_rank,
        .distributed_shuffle = config.distributed_shuffle,
        .distributed_drop_last = config.distributed_drop_last,
    });
    return config;
}