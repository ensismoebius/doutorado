/**
 * @file src/experiments/03/lib/include/cli.hpp
 * @brief Command-line interface definitions and configuration for Experiment03.
 *
 * Contains `Config` structure and parsing helper `parseCliParams` used to
 * configure datasets, autoencoders and training hyperparameters for the
 * experiment binary.
 */

#pragma once

#include "Experiment03Config.hpp"

/**
 * Parses command-line arguments and returns a fully populated configuration.
 *
 * - `argc` Argument count from the command line.
 * - `argv` Argument vector from the command line.
 * - `default_config` Default configuration structure.
 * - `return` Parsed configuration including resolved sampler options.
 */
auto parseCliParams(int argc, char* argv[], const Config& default_config) -> Config;