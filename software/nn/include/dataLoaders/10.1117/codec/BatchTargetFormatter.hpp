#ifndef NN_DATALOADERS_10_1117_BATCHTARGETFORMATTER_HPP
#define NN_DATALOADERS_10_1117_BATCHTARGETFORMATTER_HPP

#include <string>

#include "utility/batching.hpp"

namespace nn::dataLoaders
{

/**
 * @brief Formats protocol 10.1117 batch target labels into a human-readable text block.
 *
 * The formatter is independent from any experiment executable and can be reused
 * by demos, diagnostics tools, and future experiments.
 */
auto formatProtocol101117BatchTargets(const Batch& batch) -> std::string;

} // namespace nn::dataLoaders

#endif // NN_DATALOADERS_10_1117_BATCHTARGETFORMATTER_HPP
