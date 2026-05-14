#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "E04CliOptions.hpp"
#include "E04Config.hpp"

namespace e04
{

auto has_compare_marker(const std::string& arg) -> bool;

auto source_profile_dir() -> std::filesystem::path;
auto source_results_dir() -> std::filesystem::path;

void print_usage(const char* prog);

auto parse_cli(int argc, char* argv[]) -> CliOptions;
auto resolve_profile_path(const CliOptions& opts) -> std::filesystem::path;
auto load_config(const std::filesystem::path& path, const CliOptions& cli_opts)
    -> E04Config;
auto config_hash(const E04Config& cfg) -> std::size_t;

auto should_run_comparative_cli(int argc, char* argv[]) -> bool;

} // namespace e04
