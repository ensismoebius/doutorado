#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "ComparativeConfig.hpp"
#include "DatasetSplit.hpp"

namespace comparative_autoencoder_experiment
{

auto to_window_tensor(const nn::Tensor& signal, int window_size) -> std::vector<nn::Tensor>;

auto collect_signal_files(const ComparativeConfig& cfg, const std::string& dataset)
    -> std::vector<std::filesystem::path>;

auto build_split(const ComparativeConfig& cfg, const std::string& dataset) -> DatasetSplit;

} // namespace comparative_autoencoder_experiment
