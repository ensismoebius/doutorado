#include "nn/dataLoaders/SamplerOptionResolution.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

auto normalizeSamplerTypeToken(std::string token) -> std::string
{
    std::transform(token.begin(),
        token.end(),
        token.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return token;
}

auto resolveDefaultSamplerOptions(const SamplerOptionSelection& selection)
    -> DataLoader::DefaultSamplerOptions
{
    DataLoader::DefaultSamplerOptions options{};
    options.seed = selection.seed;

    const std::string sampler_type = normalizeSamplerTypeToken(selection.sampler_type);
    if (sampler_type.empty())
    {
        options.type = selection.shuffle ? DataLoader::DefaultSamplerType::Random
                                         : DataLoader::DefaultSamplerType::Sequential;
        return options;
    }

    if (sampler_type == "sequential")
    {
        options.type = DataLoader::DefaultSamplerType::Sequential;
        return options;
    }

    if (sampler_type == "random")
    {
        options.type = DataLoader::DefaultSamplerType::Random;
        return options;
    }

    if (sampler_type == "weighted")
    {
        options.type = DataLoader::DefaultSamplerType::WeightedRandom;
        options.weights = selection.weights;
        options.weighted_num_samples = selection.weighted_num_samples;
        return options;
    }

    if (sampler_type == "distributed")
    {
        options.type = DataLoader::DefaultSamplerType::Distributed;
        options.num_replicas = selection.distributed_num_replicas;
        options.rank = selection.distributed_rank;
        options.distributed_shuffle = selection.distributed_shuffle;
        options.distributed_drop_last = selection.distributed_drop_last;
        return options;
    }

    throw std::runtime_error("Unknown sampler type: " + sampler_type);
}
