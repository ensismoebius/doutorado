#pragma once

#include <string>

namespace meeting01
{

struct CliOptions
{
    std::string comparative_config;
    std::string dataset_root;
    std::string dataset; // empty = use the profile's evaluation.datasets; else run only this one
    int cv_fold = -1;    // -1 = use the profile's value; >= 0 overrides it (LOSO fold loop)
    bool cv_fold_set = false;
    bool no_tui = false; // force-disable the live progress TUI even on a terminal
    // Print the validated, resolved configuration (resolved_config_json, plus each dataset's
    // resolved source) as JSON on stdout and exit without running anything.
    bool print_resolved_config = false;
    bool help = false;
};

} // namespace meeting01
