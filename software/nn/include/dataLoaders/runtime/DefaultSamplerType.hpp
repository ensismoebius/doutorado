#ifndef NN_DATALOADERS_DEFAULT_SAMPLER_TYPE_HPP
#define NN_DATALOADERS_DEFAULT_SAMPLER_TYPE_HPP

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace nn::dataLoaders
{

enum class DefaultSamplerType
{
    Sequential,
    Random,
    WeightedRandom,
    Distributed,
};

struct DefaultSamplerOptions
{
    DefaultSamplerType type = DefaultSamplerType::Sequential;

    std::optional<unsigned int> seed = std::nullopt;

    std::vector<double> weights = {};
    std::optional<std::size_t> weighted_num_samples = std::nullopt;

    std::size_t num_replicas = 1;
    std::size_t rank = 0;
    bool distributed_shuffle = true;
    bool distributed_drop_last = false;
};

} // namespace nn::dataLoaders

#endif // NN_DATALOADERS_DEFAULT_SAMPLER_TYPE_HPP