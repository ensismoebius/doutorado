#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "GuayaquilCliOptions.hpp"
#include "GuayaquilConfig.hpp"

namespace guayaquil
{

auto has_compare_marker(const std::string& arg) -> bool;

auto source_profile_dir() -> std::filesystem::path;
auto source_results_dir() -> std::filesystem::path;

void print_usage(const char* prog);

auto parse_cli(int argc, char* argv[]) -> CliOptions;
auto resolve_profile_path(const CliOptions& opts) -> std::filesystem::path;
auto load_config(const std::filesystem::path& path, const CliOptions& cli_opts) -> GuayaquilConfig;
auto config_hash(const GuayaquilConfig& cfg) -> std::size_t;

auto should_run_comparative_cli(int argc, char* argv[]) -> bool;

} // namespace guayaquil
