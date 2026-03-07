#include <cstddef>
#include <optional>
#include <string>

#include "CLI/CLI.hpp"

using CLI::App;
using std::optional;
using std::string;

struct Config
{
    string subject_regex_pattern; // Subject Id
    string dataset_root;          // Path containing subject dirs

    size_t batch_size;  // Batch size for DataLoader
    size_t max_batches; // Max batches to iterate

    bool shuffle = true;         // Shuffle samples before batching?
    optional<unsigned int> seed; // Random seed for shuffling (ignored if shuffle=false)
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