#pragma once

#include <string>

namespace comparative_autoencoder_experiment
{

struct CliOptions
{
    std::string comparative_config;
    std::string dataset_root;
    bool help = false;
};

} // namespace comparative_autoencoder_experiment
