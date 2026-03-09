#include "../include/SamplerOptionsFactory.hpp"

#include <stdexcept>

auto makeSamplerOptions(const Config& config) -> DataLoader::DefaultSamplerOptions
{
    DataLoader::DefaultSamplerOptions options{};
    options.seed = config.seed;

    if (config.sampler_type.empty())
    {
        options.type = config.shuffle ? DataLoader::DefaultSamplerType::Random
                                      : DataLoader::DefaultSamplerType::Sequential;
        return options;
    }

    if (config.sampler_type == "sequential")
    {
        options.type = DataLoader::DefaultSamplerType::Sequential;
        return options;
    }

    if (config.sampler_type == "random")
    {
        options.type = DataLoader::DefaultSamplerType::Random;
        return options;
    }

    if (config.sampler_type == "weighted")
    {
        options.type = DataLoader::DefaultSamplerType::WeightedRandom;
        options.weights = config.sampler_weights;
        options.weighted_num_samples = config.weighted_num_samples;
        return options;
    }

    if (config.sampler_type == "distributed")
    {
        options.type = DataLoader::DefaultSamplerType::Distributed;
        options.num_replicas = config.distributed_num_replicas;
        options.rank = config.distributed_rank;
        options.distributed_shuffle = config.distributed_shuffle;
        options.distributed_drop_last = config.distributed_drop_last;
        return options;
    }

    throw std::runtime_error("Unknown sampler type: " + config.sampler_type);
}
