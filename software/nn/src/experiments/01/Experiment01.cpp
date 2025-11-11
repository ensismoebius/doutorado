#include <iostream>

#include "core/dataLoaders/AudioLoader.h"
#include "core/dataLoaders/EEGLoader.h"
#include "core/dataLoaders/MatFileFlags.h" // Include the new header

static void init()
{
    // Initialization code goes here
}

static void perform()
{
    // Experiment 01 perform code goes here
    auto [audioSamples, audioStimulus, eegIndex] = nn::dataLoaders::loadAudioFromMat(
        "/home/ensismoebius/Documentos/UNESP/doutorado/"
        "databases/BaseDeDatosHablaImaginada/S01/"
        "S01_Audio.mat",
        0);

    // Use audioSamples, audioStimulus, and eegIndex as needed
    (void) audioSamples;
    (void) audioStimulus;
    (void) eegIndex;
    // For example, print the audio stimulus and EEG index
    std::cout << MatFileFlags::getAudioFlagName(MatFileFlags::AudioFlag::Stimulus) << ": " << audioStimulus << '\n';
    std::cout << MatFileFlags::getAudioFlagName(MatFileFlags::AudioFlag::EEG_Index) << ": " << eegIndex << '\n';

    auto [eegSamples, eegInfo] = nn::dataLoaders::loadEEGFromMat(
        "/home/ensismoebius/Documentos/UNESP/doutorado/"
        "databases/BaseDeDatosHablaImaginada/S01/"
        "S01_EEG.mat",
        0);
    // Use eegSamples and eegInfo as needed
    (void) eegSamples;
    (void) eegInfo;
    // For example, print the EEG info from the EEG file
    std::cout << "EEG Info from EEG file: "
              << MatFileFlags::getEEGFlagName(MatFileFlags::EEGFlag::Modality) << "=" << eegInfo[static_cast<int>(MatFileFlags::EEGFlag::Modality)] << ", "
              << MatFileFlags::getEEGFlagName(MatFileFlags::EEGFlag::Stimulus) << "=" << eegInfo[static_cast<int>(MatFileFlags::EEGFlag::Stimulus)] << ", "
              << MatFileFlags::getEEGFlagName(MatFileFlags::EEGFlag::Artifact) << "=" << eegInfo[static_cast<int>(MatFileFlags::EEGFlag::Artifact)] << '\n';
}

auto main(int argc, char** argv) -> int
{
    init();
    // Experiment 01 code goes here
    perform();
    return 0;
}