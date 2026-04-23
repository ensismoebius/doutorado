#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "CliOptions.hpp"
#include "ComparativeConfig.hpp"

namespace comparative_autoencoder_experiment
{

auto has_compare_marker(const std::string& arg) -> bool;

auto source_profile_dir() -> std::filesystem::path;
auto source_results_dir() -> std::filesystem::path;

void print_usage(const char* prog);

auto parse_cli(int argc, char* argv[]) -> CliOptions;
auto resolve_profile_path(const CliOptions& opts) -> std::filesystem::path;
auto load_config(const std::filesystem::path& path) -> ComparativeConfig;
auto config_hash(const ComparativeConfig& cfg) -> std::size_t;

auto should_run_comparative_cli(int argc, char* argv[]) -> bool;

} // namespace comparative_autoencoder_experiment

namespace lstm_autoencoder_experiment
{

auto has_experiment04_marker(const std::string& arg) -> bool;

// Normalizes legacy/alias flags into the comparative runner flags.
void normalize_aliases(int argc, char* argv[], std::vector<std::string>& args);

// Converts an argv-like vector of std::string into a vector of char*.
void to_argv(std::vector<std::string>& args, std::vector<char*>& argv_out);

} // namespace lstm_autoencoder_experiment
