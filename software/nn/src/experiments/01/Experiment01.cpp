#include <iostream>

#include "dataLoaders/AudioLoader.h"

static void init()
{
    // Initialization code goes here
}

static void perform()
{
    // Experiment 01 perform code goes here
    auto [audioSamples, eegIndex] = nn::dataLoaders::loadAudioFromMat(
        "/home/ensismoebius/Documentos/UNESP/doutorado/databases/BaseDeDatosHablaImaginada/S01/"
        "S01_Audio.mat",
        0);

    // Use audioSamples and eegIndex as needed
    (void) audioSamples;
    (void) eegIndex;
    // For example, print the EEG index
    std::cout << "EEG Index: " << eegIndex << '\n';
}

auto main(int argc, char** argv) -> int
{
    init();
    // Experiment 01 code goes here
    perform();
    return 0;
}