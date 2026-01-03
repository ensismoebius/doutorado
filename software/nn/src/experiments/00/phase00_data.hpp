#pragma once

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
