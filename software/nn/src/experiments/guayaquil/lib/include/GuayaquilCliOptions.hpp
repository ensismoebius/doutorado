#pragma once

#include <string>

namespace guayaquil
{

struct CliOptions
{
    std::string comparative_config;
    std::string dataset_root;
    int cv_fold = -1; // -1 = use the profile's value; >= 0 overrides it (LOSO fold loop)
    bool cv_fold_set = false;
    bool help = false;
};

} // namespace guayaquil
