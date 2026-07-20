/**
 * @file src/experiments/autoencoderRunner/lib/include/cli.hpp
 * @brief Command-line interface definitions and configuration for AutoencoderRunner.
 *
 * Contains `Config` structure and parsing helper `parseCliParams` used to
 * configure datasets, autoencoders and training hyperparameters for the
 * experiment binary.
 */

#pragma once

#include "AutoencoderRunnerConfig.hpp"

/**
 * Parses command-line arguments and returns a fully populated configuration.
 *
 * - `argc` Argument count from the command line.
 * - `argv` Argument vector from the command line.
 * - `default_config` Base configuration structure (typically seeded from `default` profile).
 * - `return` Parsed configuration including resolved sampler options.
 */
auto parseCliParams(int argc, char* argv[], const Config& default_config) -> Config;