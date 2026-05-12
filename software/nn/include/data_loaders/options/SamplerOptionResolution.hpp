#ifndef NN_DATALOADERS_SAMPLEROPTIONRESOLUTION_HPP
#define NN_DATALOADERS_SAMPLEROPTIONRESOLUTION_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "data_loaders/runtime/DataLoader.hpp"

/**
 * @brief User-facing sampler selection arguments independent from CLI libraries.
 */
struct SamplerOptionSelection
{
    std::string sampler_type;
    bool shuffle = true;
    std::optional<unsigned int> seed = std::nullopt;

    std::vector<double> weights = {};
    std::optional<std::size_t> weighted_num_samples = std::nullopt;

    std::size_t distributed_num_replicas = 1;
    std::size_t distributed_rank = 0;
    bool distributed_shuffle = true;
    bool distributed_drop_last = false;
};

/**
 * @brief Resolves a normalized sampler token from user input.
 */
auto normalizeSamplerTypeToken(std::string token) -> std::string;

/**
 * @brief Converts a user sampler selection into DataLoader default sampler options.
 */
auto resolveDefaultSamplerOptions(const SamplerOptionSelection& selection)
    -> DataLoader::DefaultSamplerOptions;

#endif // NN_DATALOADERS_SAMPLEROPTIONRESOLUTION_HPP
