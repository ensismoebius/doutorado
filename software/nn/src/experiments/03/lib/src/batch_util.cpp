#include "../include/batch_util.hpp"

#include <iostream>

#include "nn/dataLoaders/10.1117/NAMES.hpp"

using std::cout;
using std::size_t;

using nn::dataLoaders::ARTIFACT_NAMES;
using nn::dataLoaders::ESTIMULUS_NAMES;
using nn::dataLoaders::MODALITY_NAMES;

auto printData(const Batch& batch) -> void
{
    for (size_t i = 0; i < batch.inputs.rows(); ++i)
    {
        cout << "Sample " << i << ":\n";
        cout << "  - Target subject ID: " << batch.targets.at(i, 0) << '\n';
        cout << "  - Target modality: "
             << MODALITY_NAMES.at(static_cast<int>(batch.targets.at(i, 1))) << '\n';
        cout << "  - Target stimulus: "
             << ESTIMULUS_NAMES.at(static_cast<int>(batch.targets.at(i, 2))) << '\n';
        cout << "  - Target artifact: "
             << ARTIFACT_NAMES.at(static_cast<int>(batch.targets.at(i, 3))) << '\n';
        cout << "  - Target EEG index label: " << batch.targets.at(i, 4) << "\n\n";
    }
}