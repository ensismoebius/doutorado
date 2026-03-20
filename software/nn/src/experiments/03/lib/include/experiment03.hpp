#pragma once

#include "cli.hpp"
#include "nn/dataLoaders/BatchPrefetcher.hpp"
#include "nn/dataLoaders/DataLoader.hpp"
#include "nn/dataLoaders/Dataset.hpp"
#include "nn/layers/Module.hpp"

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
    std::unique_ptr<DataLoader> loader_;
    std::unique_ptr<Module> model_;
    std::unique_ptr<BatchPrefetcher> prefetcher_;
    std::shared_ptr<Dataset> dataset_;

    std::size_t seen_batches_ = 0;
    std::size_t processed_samples_ = 0;
    std::size_t dataset_total_samples_ = 0;
};
