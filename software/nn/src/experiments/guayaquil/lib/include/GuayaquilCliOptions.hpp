#pragma once

#include <string>

namespace guayaquil
{

struct CliOptions
{
    std::string comparative_config;
    std::string dataset_root;
    std::string dataset; // empty = use the profile's evaluation.datasets; else run only this one
    int cv_fold = -1;    // -1 = use the profile's value; >= 0 overrides it (LOSO fold loop)
    bool cv_fold_set = false;
    bool no_tui = false; // force-disable the live progress TUI even on a terminal
    bool help = false;
};

} // namespace guayaquil
