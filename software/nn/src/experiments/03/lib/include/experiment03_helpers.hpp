/**
 * @file src/experiments/03/lib/include/experiment03_helpers.hpp
 * @brief Helper utilities for Experiment03 (conversion, device runtime, model factory).
 *
 * These helpers were previously defined in the experiment driver translation unit.
 * Moving them into the experiment03 library allows reuse and keeps the driver
 * implementation focused on orchestration.
 */

#pragma once

#include <memory>

#include "AutoencoderConfig.hpp"
#include "Experiment03Config.hpp"
#include "Experiment03DatasetType.hpp"
#include "nn/dataLoaders/SqliteBatchSource.hpp"
#include "nn/layers/Module.hpp"
#include "nn/tensor/opencl/OpenCLTensorBackend.hpp"

namespace experiment03
{
// Convert the experiment-level dataset enum to the SqliteBatchSource enum.
auto to_sqlite_dataset_type(Experiment03DatasetType dataset_type)
    -> nn::dataLoaders::SqliteDatasetType;

// Initialize OpenCL runtime when requested; returns RAII runtime scope.
auto initialize_device_runtime_or_throw(const Config& config)
    -> nn::OpenCLTensorBackend::RuntimeScope;

// Build an autoencoder `Module` instance according to experiment `Config`.
auto build_autoencoder_model(const Config& config, nn::Index input_features)
    -> std::unique_ptr<Module>;

} // namespace experiment03
