#include "../include/LstmAutoencoderExperiment.hpp"

#include <string>
#include <vector>

#include "../include/Experiment04Args.hpp"
#include "../include/Experiment04Cli.hpp"

auto LstmAutoencoderExperiment::run(int argc, char* argv[]) -> int
{
    std::vector<char*> normalized_argv;
    std::vector<std::string> normalized_args;

    lstm_snn_autoencoder_comparisom::normalize_aliases(argc, argv, normalized_args);
    lstm_snn_autoencoder_comparisom::to_argv(normalized_args, normalized_argv);

    return lstm_snn_autoencoder_comparisom::run_comparative_experiment( //
        static_cast<int>(normalized_argv.size()),                       //
        normalized_argv.data()                                          //
    );
}
