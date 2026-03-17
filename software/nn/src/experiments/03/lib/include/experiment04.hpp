#pragma once

#include "DemoProbeModel.hpp"
#include "cli.hpp"
#include "nn/dataLoaders/10.1117/Protocol101117Dataset.hpp"
#include "nn/dataLoaders/BatchPrefetcher.hpp"
#include "nn/dataLoaders/DataLoader.hpp"

class Experiment04
{
   public:
    explicit Experiment04(const Config& config);
    // Run the experiment; returns 0 on success, non-zero on failure.
    int run();

   private:
    Config config_;
    // Dataset and pipeline components are stored as members so they
    // persist across `run()` and can be inspected or reused.
    std::unique_ptr<DataLoader> loader_;
    std::unique_ptr<DemoProbeModel> model_;
    std::unique_ptr<BatchPrefetcher> prefetcher_;
    std::shared_ptr<Protocol101117Dataset> dataset_;

    std::size_t seen_batches_ = 0;
    std::size_t processed_samples_ = 0;
    std::size_t dataset_total_samples_ = 0;
};
