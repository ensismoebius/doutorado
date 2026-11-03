#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "Meeting01CliOptions.hpp"
#include "Meeting01Config.hpp"
#include "nlohmann/json.hpp"

namespace meeting01
{

auto has_compare_marker(const std::string& arg) -> bool;

auto source_profile_dir() -> std::filesystem::path;
auto source_results_dir() -> std::filesystem::path;

void print_usage(const char* prog);

auto parse_cli(int argc, char* argv[]) -> CliOptions;
auto resolve_profile_path(const CliOptions& opts) -> std::filesystem::path;
auto load_config(const std::filesystem::path& path, const CliOptions& cli_opts) -> Meeting01Config;
// Every field of `cfg` that can change a result, as JSON (keys sorted, so dump() is stable).
// Output-only fields are left out: dataset.latex_data_dir, dataset.save_models and the GA
// blocks' checkpoint_every_generations. It is what config_hash hashes and what each fold's
// split manifest records as "config", so the paper can be generated from what ran.
auto resolved_config_json(const Meeting01Config& cfg) -> nlohmann::json;
// One dataset's resolved source (Dataset::resolve) as JSON: window, folds, rate, caps, latent.
auto dataset_source_json(const Meeting01Config::DatasetSource& s) -> nlohmann::json;
auto config_hash(const Meeting01Config& cfg) -> std::size_t;

auto should_run_comparative_cli(int argc, char* argv[]) -> bool;

} // namespace meeting01
