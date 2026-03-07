#include <string>

#include "CLI/CLI.hpp"

using CLI::App;
using CLI::ExistingFile;
using std::string;

struct Config
{
    size_t row_index = 0;
    string eeg_mat;
    string audio_mat;
};

/**
 * @brief Parses command-line arguments and updates the configuration accordingly.
 *
 * @param argc Argument count from the command line.
 * @param argv Argument vector from the command line.
 * @param config Configuration structure to update.
 * @param default_config Default configuration structure.
 */
void parseCliParams(int argc, char* argv[], Config& config, const Config& default_config);
