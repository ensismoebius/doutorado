#pragma once

#include <string>

namespace e04
{

struct CliOptions
{
    std::string comparative_config;
    std::string dataset_root;
    bool help = false;
};

} // namespace e04
