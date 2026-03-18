#include "nn/dataLoaders/10.1117/codec/BatchTargetFormatter.hpp"

#include <cstddef>
#include <sstream>

#include "nn/dataLoaders/10.1117/schema/NAMES.hpp"

namespace nn::dataLoaders
{
namespace
{

auto labelOrUnknown(const std::map<int, std::string>& labels, int key) -> std::string
{
    const auto it = labels.find(key);
    if (it != labels.end())
    {
        return it->second;
    }

    return "Unknown(" + std::to_string(key) + ")";
}

} // namespace

auto formatProtocol101117BatchTargets(const Batch& batch) -> std::string
{
    std::ostringstream out;

    for (std::size_t i = 0; i < batch.targets.rows(); ++i)
    {
        const int modality = static_cast<int>(batch.targets.at(i, 1));
        const int stimulus = static_cast<int>(batch.targets.at(i, 2));
        const int artifact = static_cast<int>(batch.targets.at(i, 3));

        out << "Sample " << i << ":\n";
        out << "  - Target subject ID: " << batch.targets.at(i, 0) << '\n';
        out << "  - Target modality: " << labelOrUnknown(MODALITY_NAMES, modality) << '\n';
        out << "  - Target stimulus: " << labelOrUnknown(ESTIMULUS_NAMES, stimulus) << '\n';
        out << "  - Target artifact: " << labelOrUnknown(ARTIFACT_NAMES, artifact) << '\n';
        out << "  - Target EEG index label: " << batch.targets.at(i, 4) << "\n\n";
    }

    return out.str();
}

} // namespace nn::dataLoaders
