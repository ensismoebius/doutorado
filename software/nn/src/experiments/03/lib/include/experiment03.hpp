/**
 * @file src/experiments/03/lib/include/experiment03.hpp
 * @brief Public experiment driver API for Experiment03.
 *
 * Declares the `Experiment03` class which orchestrates dataset discovery,
 * data loading, model construction and training loop. Consumers should
 * configure the experiment via the `Config` structure in `cli.hpp`.
 */

#pragma once

#include "Experiment03Config.hpp"
#include "nn/dataLoaders/datasets/Dataset.hpp"
#include "nn/dataLoaders/runtime/BatchPrefetcher.hpp"
#include "nn/dataLoaders/runtime/DataLoader.hpp"
#include "nn/layers/base/Module.hpp"

class Experiment03
{
   public:
    explicit Experiment03(const Config& config);
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
    std::unique_ptr<Module<nn::EigenTensorBackend>> model_;

    std::size_t seen_batches_ = 0;
    std::size_t processed_samples_ = 0;
    std::size_t dataset_total_samples_ = 0;
};
