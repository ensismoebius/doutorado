#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "E04Config.hpp"
#include "E04DatasetSplit.hpp"

namespace e04
{

auto to_window_tensor(const nn::Tensor& signal, int window_size) -> std::vector<nn::Tensor>;

auto collect_signal_files(const E04Config& cfg, const std::string& dataset)
    -> std::vector<std::filesystem::path>;

auto build_split(const E04Config& cfg, const std::string& dataset) -> DatasetSplit;

} // namespace e04
