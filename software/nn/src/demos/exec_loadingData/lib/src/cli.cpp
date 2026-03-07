#include "../include/cli.hpp"

void parseCliParams(int argc, char* argv[], Config& config, const Config& default_config)
{
    App app("Load one audio row and one EEG row from 10.1117 MATLAB files.");

    app.add_option("--audio-mat", config.audio_mat, "Path to audio MAT file")
        ->expected(0, 1)
        ->check(ExistingFile);

    app.add_option("--eeg-mat", config.eeg_mat, "Path to EEG MAT file")
        ->expected(0, 1)
        ->check(ExistingFile);

    app.add_option("--row", config.row_index, "Row index to load from both files")
        ->expected(1)
        ->check(CLI::NonNegativeNumber);

    try
    {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError& e)
    {
        app.exit(e);
    }

    config.audio_mat =             //
        config.audio_mat.empty() ? //
            default_config.audio_mat
                                 : //
            config.audio_mat;

    config.eeg_mat =             //
        config.eeg_mat.empty() ? //
            default_config.eeg_mat
                               : //
            config.eeg_mat;
}