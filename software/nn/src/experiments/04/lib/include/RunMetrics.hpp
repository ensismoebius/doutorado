#pragma once

#include <cstddef>

namespace comparative_autoencoder_experiment
{

struct RunMetrics
{
    float mse = 0.0f;
    float mae = 0.0f;
    float r2 = 0.0f;

    float precision = 0.0f;
    float recall = 0.0f;
    float f1 = 0.0f;

    float spike_rate = 0.0f;
    float energy = 0.0f;

    float train_ms = 0.0f;
    float infer_ms = 0.0f;

    std::size_t parameter_count = 0;
    std::size_t macs = 0;
};

} // namespace comparative_autoencoder_experiment
