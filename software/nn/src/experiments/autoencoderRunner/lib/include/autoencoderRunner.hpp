/**
 * @file src/experiments/autoencoderRunner/lib/include/autoencoderRunner.hpp
 * @brief Public experiment driver API for AutoencoderRunner.
 *
 * Declares the `AutoencoderRunner` class which orchestrates dataset discovery,
 * data loading, model construction and training loop. Consumers should
 * configure the experiment via the `Config` structure in `cli.hpp`.
 */

#pragma once

#include "AutoencoderRunnerConfig.hpp"
#include "Backend.hpp"
#include "data_loaders/datasets/Dataset.hpp"
#include "data_loaders/runtime/BatchPrefetcher.hpp"
#include "data_loaders/runtime/DataLoader.hpp"
#include "layers/base/Module.hpp"

class AutoencoderRunner
{
   public:
    explicit AutoencoderRunner(const Config& config);
    // Run the experiment; returns 0 on success, non-zero on failure.
    int run();

   private:
    Config config_;
    // Dataset and pipeline components are stored as members so they
    // persist across `run()` and can be inspected or reused.
    // dataset_ uses base class pointer to support multiple dataset types.
    std::unique_ptr<BatchPrefetcher> prefetcher_;
    std::unique_ptr<DataLoader> data_loader_;
    std::shared_ptr<Dataset> dataset_;
    std::unique_ptr<Module<nn::Backend>> model_;

    std::size_t seen_batches_ = 0;
    std::size_t processed_samples_ = 0;
    std::size_t dataset_total_samples_ = 0;
};
