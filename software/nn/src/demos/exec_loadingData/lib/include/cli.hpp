#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "CLI/CLI.hpp"

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
};

/**
 * @brief Parses command-line arguments and updates the configuration accordingly.
 *
 * @param argc Argument count from the command line.
 * @param argv Argument vector from the command line.
 * @param config Configuration structure to update.
 * @param default_config Default configuration structure.
 */
void parseCliParams(int argc, char* argv[], Config& config, const Config& default_config);