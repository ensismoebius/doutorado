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
    // Build the split, write only the PCA / mean-frame reference inputs, exit (no training).
    bool dump_reference_inputs_only = false;
    bool help = false;
};

} // namespace meeting01
