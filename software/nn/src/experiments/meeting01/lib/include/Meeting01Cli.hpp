#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "Meeting01CliOptions.hpp"
#include "Meeting01Config.hpp"

namespace meeting01
{

auto has_compare_marker(const std::string& arg) -> bool;

auto source_profile_dir() -> std::filesystem::path;
auto source_results_dir() -> std::filesystem::path;

void print_usage(const char* prog);

auto parse_cli(int argc, char* argv[]) -> CliOptions;
auto resolve_profile_path(const CliOptions& opts) -> std::filesystem::path;
auto load_config(const std::filesystem::path& path, const CliOptions& cli_opts) -> Meeting01Config;
auto config_hash(const Meeting01Config& cfg) -> std::size_t;

auto should_run_comparative_cli(int argc, char* argv[]) -> bool;

} // namespace meeting01
