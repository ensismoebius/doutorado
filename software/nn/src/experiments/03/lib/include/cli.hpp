/**
 * @file src/experiments/03/lib/include/cli.hpp
 * @brief Command-line interface definitions and configuration for Experiment03.
 *
 * Contains `Config` structure and parsing helper `parseCliParams` used to
 * configure datasets, autoencoders and training hyperparameters for the
 * experiment binary.
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "AutoencoderConfig.hpp"
#include "CLI/CLI.hpp"
#include "nn/dataLoaders/10.1117/protocol/Protocol101117Dataset.hpp"
#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/windowing/WindowSpec.hpp"

using CLI::App;
using std::optional;
using std::string;

enum class Experiment03DatasetType
{
    Protocol,
    EegWindow,
    AudioWindow,
    FusedWindow
};

enum class Experiment03AutoencoderType
{
    ProtocolAnn,
    EegWindowAnn,
    AudioWindowAnn,
    FusedWindowAnn,
    ProtocolSnn,
    EegWindowSnn,
    AudioWindowSnn,
    FusedWindowSnn
};

struct Config
{
    string subject_regex_pattern = ""; // Subject Id
    string dataset_root = "";          // Path containing subject dirs

    size_t batch_size = 32;   // Batch size for DataLoader
    size_t max_batches = 100; // Max batches to iterate

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

    // Dataset variant used by experiment03.
    Experiment03DatasetType dataset_type = Experiment03DatasetType::FusedWindow;

    // Autoencoder variant used by experiment03.
    Experiment03AutoencoderType autoencoder_type = Experiment03AutoencoderType::FusedWindowAnn;

    // Autoencoder architecture hyperparameters.
    int ae_hidden_size = 64;
    int ae_latent_size = 32;
    int ae_depth = 2;
    AutoencoderArchitecture ae_architecture = AutoencoderArchitecture::Auto;
    int ae_branch_hidden_size = 0;
    int ae_fusion_hidden_size = 0;
    int ae_residual_blocks = 1;
    float ae_time_step = 1.0F;
    float ae_resistance = 1.0F;
    float ae_capacitance = 1.0F;

    // Training hyperparameters.
    float learning_rate = 0.001F;
    int epochs = 1;

    // Window specs used by windowing datasets.
    nn::windowing::WindowSpec eeg_window_spec{
        .window_size = 256, .overlap = 0.5f, .sample_rate = 1024};
    nn::windowing::WindowSpec audio_window_spec{
        .window_size = 11025, .overlap = 0.5f, .sample_rate = 44100};

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