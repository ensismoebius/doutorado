#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "CLI/CLI.hpp"
#include "nn/dataLoaders/10.1117/Protocol101117Dataset.hpp"
#include "nn/dataLoaders/DataLoader.hpp"

using CLI::App;
using std::optional;
using std::string;

struct Config
{
    string subject_regex_pattern; // Subject Id
    string dataset_root;          // Path containing subject dirs

    size_t batch_size;  // Batch size for DataLoader
    size_t max_batches; // Max batches to iterate

    // Legacy controls (used when sampler_type is not provided)
    bool shuffle = true;         // Shuffle samples before batching?
    optional<unsigned int> seed; // Random seed for shuffling (ignored if shuffle=false)

    // Explicit default sampler selector
    string sampler_type; // Empty means use legacy shuffle behavior

    // Weighted sampler options
    std::vector<double> sampler_weights;
    std::optional<size_t> weighted_num_samples;

    // Distributed sampler options
    size_t distributed_num_replicas = 1;
    size_t distributed_rank = 0;
    bool distributed_shuffle = true;
    bool distributed_drop_last = false;

    // Input modality used by Protocol101117Dataset.
    Protocol101117InputMode input_mode = Protocol101117InputMode::Concatenated;

    // Prefetch lookahead: number of batches to prefetch in background.
    // Default is 1 to preserve prior behavior; experiments may increase this.
    std::size_t lookahead = 1;

    // Parsed and resolved sampler options for DataLoader construction.
    DataLoader::DefaultSamplerOptions sampler_options{};
};

/**
 * Parses command-line arguments and returns a fully populated configuration.
 *
 * - `argc` Argument count from the command line.
 * - `argv` Argument vector from the command line.
 * - `default_config` Default configuration structure.
 * - `return` Parsed configuration including resolved sampler options.
 */
auto parseCliParams(int argc, char* argv[], const Config& default_config) -> Config;