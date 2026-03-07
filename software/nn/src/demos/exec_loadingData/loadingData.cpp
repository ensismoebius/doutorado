/**
 * @file loadingData.cpp
 * @brief Tiny utility demo: load one audio row and one EEG row via 10.1117 loaders.
 */

#include <array>
#include <iostream>
#include <string>
#include <tuple>

#include "lib/include/cli.hpp"
#include "nn/dataLoaders/10.1117/AudioLoader.h"
#include "nn/dataLoaders/10.1117/EEGLoader.h"

using nn::dataLoaders::EEG_CHANNELS_NAMES;
using nn::dataLoaders::loadAudioFromMat;
using nn::dataLoaders::loadEEGFromMat;

using std::cerr;
using std::cout;
using std::exception;
using std::string;

auto main(int argc, char* argv[]) -> int
{
    const string prefix =
        "/home/ensismoebius/Documentos/"
        "UNESP/doutorado/databases/"
        "BaseDeDatosHablaImaginada/";

    const Config dft_config{
        .row_index = 0,                           //
        .eeg_mat = prefix + "S01/S01_EEG.mat",    //
        .audio_mat = prefix + "S01/S01_Audio.mat" //
    };

    Config config;

    parseCliParams(argc, argv, config, dft_config);

    try
    {
        const auto [audio_tensor, stimulus, eeg_index] = loadAudioFromMat( //
            config.audio_mat,                                              //
            config.row_index                                               //
        );

        const auto [eeg_tensor, eeg_labels] = loadEEGFromMat( //
            config.eeg_mat,                                   //
            config.row_index                                  //
        );

        cout << "Audio loaded from: " << config.audio_mat << '\n';
        cout << "  shape: [" << audio_tensor.rows() << "x" << audio_tensor.cols() << "]\n";
        cout << "  stimulus: " << stimulus << '\n';
        cout << "  eeg_index: " << eeg_index << "\n\n";

        cout << "EEG loaded from: " << config.eeg_mat << '\n';
        cout << "  shape: [" << eeg_tensor.rows() << "x" << eeg_tensor.cols() << "]\n";
        cout << "  modality: " << eeg_labels[0] << '\n';
        cout << "  stimulus: " << eeg_labels[1] << '\n';
        cout << "  artifact: " << eeg_labels[2] << '\n';

        if (eeg_tensor.rows() == EEG_CHANNELS_NAMES.size())
        {
            cout << "  channels:";
            for (const auto& channel_name : EEG_CHANNELS_NAMES)
            {
                cout << ' ' << channel_name;
            }
            cout << '\n';
        }
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
