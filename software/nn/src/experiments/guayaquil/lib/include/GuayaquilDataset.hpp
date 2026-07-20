#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "GuayaquilConfig.hpp"
#include "GuayaquilDatasetSplit.hpp"

namespace guayaquil
{

auto to_window_tensor(const nn::Tensor& signal, int window_size) -> std::vector<nn::Tensor>;

auto collect_signal_files(const GuayaquilConfig& cfg, const std::string& dataset)
    -> std::vector<std::filesystem::path>;

auto build_split(const GuayaquilConfig& cfg, const std::string& dataset) -> DatasetSplit;

} // namespace guayaquil
