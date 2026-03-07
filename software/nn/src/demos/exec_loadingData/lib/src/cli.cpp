#include "../include/cli.hpp"

void parseCliParams(int argc, char* argv[], Config& config, const Config& default_config)
{
    App app("PyTorch-style loader pipeline for 10.1117 EEG+Audio dataset.");

    app.add_option("--dataset-root", config.dataset_root, "Path containing subjects dir")
        ->expected(0, 1)
        ->check(CLI::ExistingDirectory)
        ->default_val(default_config.dataset_root);

    app.add_option("--subject", config.subject_regex_pattern, "Optional subject filter (e.g., S01)")
        ->expected(0, 1)
        ->default_val(default_config.subject_regex_pattern);

    app.add_option("--batch-size", config.batch_size, "Mini-batch size")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.batch_size);

    app.add_option("--max-batches", config.max_batches, "Max batches to iterate in this demo")
        ->expected(1)
        ->check(CLI::PositiveNumber)
        ->default_val(default_config.max_batches);

    app.add_option(
           "--seed", config.seed, "Deterministic seed for shuffling (ignored if --no-shuffle)")
        ->expected(0, 1)
        ->check(CLI::NonNegativeNumber)
        ->default_val(default_config.seed);

    app.add_flag("--shuffle,!--no-shuffle", config.shuffle, "Shuffle samples before batching")
        ->default_val(default_config.shuffle);

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
        app.exit(e);
    }
}
