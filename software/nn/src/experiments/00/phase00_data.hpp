/**
 * @file phase00_data.hpp
 * @brief Data aggregation for PHASE 0 (dataset traversal + trial loading).
 */

#ifndef EXPERIMENTS_00_PHASE00_DATA_HPP
#define EXPERIMENTS_00_PHASE00_DATA_HPP

#include <filesystem>
#include <vector>

#include "Config.hpp"

namespace phase00
{
struct TrialData
{
    std::vector<double> features;
    int label;
};

auto default_config_path() -> std::filesystem::path;
auto aggregate_trials(const Config& cfg) -> std::vector<TrialData>;

} // namespace phase00

#endif // EXPERIMENTS_00_PHASE00_DATA_HPP
