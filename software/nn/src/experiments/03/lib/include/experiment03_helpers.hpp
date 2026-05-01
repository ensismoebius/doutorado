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
#include <utility>
#include <vector>

#include "Experiment03Config.hpp"
#include "Experiment03DatasetType.hpp"
#include "nn/dataLoaders/sources/SqliteBatchSource.hpp"
#include "nn/layers/Layers.hpp"
#include "nn/layers/base/Module.hpp"
#include "nn/utility/Transforms.hpp"

namespace experiment03
{
using nn::MAELoss;
using nn::MSELoss;

// Convert the experiment-level dataset enum to the SqliteBatchSource enum.
auto to_sqlite_dataset_type(Experiment03DatasetType dataset_type)
    -> nn::dataLoaders::SqliteDatasetType;

// Build an autoencoder `Module` instance according to experiment `Config`.
auto build_autoencoder_model(const Config& config, nn::Index input_features)
    -> std::unique_ptr<Module<nn::Backend>>;

// Fit normalization transform from the training split (or full corpus when trial_ids is null).
auto fit_input_transform(
    const Config& config, size_t max_batches, const std::vector<int>* trial_ids)
    -> std::shared_ptr<nn::transforms::ITransform>;

// Compute modality-specific MSE diagnostics from reconstructed validation inputs.
auto modality_val_losses_from_batch(const nn::Tensor& val_inputs,
    const nn::Tensor& val_reconstruction,
    size_t eeg_features,
    size_t audio_features) -> std::pair<float, float>;

class ReconstructionLoss
{
   public:
    explicit ReconstructionLoss(const std::string& loss_type);

    auto set_target(const nn::Tensor& target) -> void;
    auto forward(const nn::Tensor& input, bool requires_grad) -> nn::Tensor;
    auto backward(const nn::Tensor& grad_output) -> nn::Tensor;

   private:
    std::unique_ptr<MSELoss> mse_;
    std::unique_ptr<MAELoss> mae_;
};

} // namespace experiment03
