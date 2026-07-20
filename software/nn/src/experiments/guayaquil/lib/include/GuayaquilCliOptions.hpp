#pragma once

#include <string>

namespace guayaquil
{

struct CliOptions
{
    std::string comparative_config;
    std::string dataset_root;
    bool help = false;
};

} // namespace guayaquil
