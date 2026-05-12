/**
 * @file DataLoader.cpp
 * @brief Implementation of the `DataLoader` batching/iteration utilities.
 */

#include "dataLoaders/runtime/DataLoader.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "dataLoaders/samplers/DistributedSampler.hpp"
#include "dataLoaders/samplers/RandomSampler.hpp"
#include "dataLoaders/samplers/SequentialSampler.hpp"
#include "dataLoaders/samplers/WeightedRandomSampler.hpp"

// Implementation notes:
// - The `DataLoader` delegates sample-index generation to an `ISampler`.
// - Each iterator owns a sampler-produced snapshot to avoid interference.
// - `Iterator::operator*()` delegates batching to `Dataset::collate()`, which
//   enables datasets to implement fast slicing/gather.

using std::invalid_argument;
using std::make_unique;
using std::move;
using std::optional;
using std::shared_ptr;
using std::size_t;
using std::unique_ptr;
using std::vector;

namespace
{

auto make_default_sampler(                           //
    size_t dataset_size,                             //
    const DataLoader::DefaultSamplerOptions& options //
    ) -> unique_ptr<ISampler>
{
    switch (options.type)
    {
        case DataLoader::DefaultSamplerType::Sequential:
            return make_unique<SequentialSampler>(dataset_size);
        case DataLoader::DefaultSamplerType::Random:
            return make_unique<RandomSampler>(dataset_size, options.seed);
        case DataLoader::DefaultSamplerType::WeightedRandom:
        {
            const size_t num_samples = options.weighted_num_samples.value_or(dataset_size);
            if (!options.weights.empty())
            {
                return make_unique<WeightedRandomSampler>(
                    options.weights, num_samples, options.seed);
            }

            // Fallback: uniform weights across the full dataset.
            vector<double> uniform_weights(dataset_size, 1.0);
            return make_unique<WeightedRandomSampler>(
                std::move(uniform_weights), num_samples, options.seed);
        }
        case DataLoader::DefaultSamplerType::Distributed:
            return make_unique<DistributedSampler>(dataset_size,
                options.num_replicas,
                options.rank,
                options.distributed_shuffle,
                options.distributed_drop_last,
                options.seed);
    }

    throw invalid_argument("DataLoader: unknown default sampler type."); //
}

} // namespace

DataLoader::DataLoader(          //
    shared_ptr<Dataset> dataset, //
    size_t batch_size,           //
    bool do_shuffle,             //
    optional<unsigned int> seed  //
    )
    : DataLoader(dataset,
          batch_size,
          DefaultSamplerOptions{
              .type = do_shuffle ? DefaultSamplerType::Random : DefaultSamplerType::Sequential,
              .seed = seed})
{
}

DataLoader::DataLoader(                  //
    shared_ptr<Dataset> dataset,         //
    size_t batch_size,                   //
    const DefaultSamplerOptions& options //
    )
    : DataLoader(                            //
          dataset,                           //
          batch_size,                        //
          make_default_sampler(              //
              dataset ? dataset->size() : 0, //
              options                        //
              )                              //
      )
{
}

DataLoader::DataLoader(          //
    shared_ptr<Dataset> dataset, //
    size_t batch_size,           //
    unique_ptr<ISampler> sampler //
    )
    :                               //
      dataset_(std::move(dataset)), //
      batch_size_(batch_size),      //
      sampler_(std::move(sampler))
{
    if (!dataset_)
    {
        throw invalid_argument("DataLoader: dataset cannot be null.");
    }
    if (!sampler_)
    {
        throw invalid_argument("DataLoader: sampler cannot be null.");
    }
    if (batch_size == 0)
    {
        throw invalid_argument("DataLoader: batch size cannot be zero.");
    }
    // Check for implicitly converted negative values (wrapped to very large size_t)
    // When -1 is passed to size_t, it becomes SIZE_MAX (typically 2^64-1 or 2^32-1)
    // Any batch_size > 1 billion is suspicious and likely a wrapped negative
    constexpr size_t MAX_REASONABLE_BATCH_SIZE = 1'000'000'000;
    if (batch_size > MAX_REASONABLE_BATCH_SIZE)
    {
        throw invalid_argument(
            "DataLoader: batch size is unreasonably "
            "large (possible negative value)." //
        );
    }

    // Precompute number of batches from sampler cardinality.
    const size_t n_samples = sampler_->index_count();

    // Calculate number of batches needed, rounding up for the last
    // batch if it doesn't divide evenly in order to partition samples
    // into batches
    num_batches_ = (n_samples + batch_size_ - 1) / batch_size_;

    if (dataset_->size() == 0 && n_samples > 0)
    {
        throw invalid_argument("DataLoader: sampler requested indices for an empty dataset.");
    }
}

auto DataLoader::begin() -> DataLoader::Iterator
{
    // Generate this epoch's sampled index list.
    auto snapshot = std::make_shared<vector<size_t>>(sampler_->index_count());
    sampler_->set_epoch(epoch_);
    sampler_->sample_into(*snapshot);

    ++epoch_;
    return {*this, 0, snapshot};
}

auto DataLoader::end() -> DataLoader::Iterator
{
    // End iterator doesn't need valid indices, just position
    return {*this, num_batches_, std::make_shared<vector<size_t>>()};
}
