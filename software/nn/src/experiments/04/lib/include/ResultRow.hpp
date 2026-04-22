#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "RunMetrics.hpp"

namespace comparative_autoencoder_experiment
{

struct ResultRow
{
    std::string dataset;
    std::string model;
    std::string encoding;
    std::string architecture;
    int layers = 1;
    float v_th = 1.0f;
    float alpha = 0.9f;
    int run_id = 0;

    std::uint32_t seed = 0u;
    std::size_t config_hash = 0u;

    RunMetrics metrics;
};

} // namespace comparative_autoencoder_experiment
