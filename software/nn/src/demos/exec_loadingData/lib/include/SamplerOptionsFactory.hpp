#ifndef EXEC_LOADINGDATA_SAMPLEROPTIONSFACTORY_HPP
#define EXEC_LOADINGDATA_SAMPLEROPTIONSFACTORY_HPP

#include "cli.hpp"
#include "nn/dataLoaders/DataLoader.hpp"

auto makeSamplerOptions(const Config& config) -> DataLoader::DefaultSamplerOptions;

#endif // EXEC_LOADINGDATA_SAMPLEROPTIONSFACTORY_HPP
