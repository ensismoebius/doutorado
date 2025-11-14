#include <iostream>
#include <filesystem> // Include for path manipulation

#include "core/dataLoaders/AudioLoader.h"
#include "core/dataLoaders/EEGLoader.h"
#include "core/dataLoaders/MatFileFlags.h" // Include the new header
#include "core/wave/Wav.h"

using std::cout;
using namespace MatFileFlags;
using namespace nn::dataLoaders;

static void init()
{
    // Initialization code goes here
}

static void perform(const std::string& inputAudioFilePath)
{
    // Experiment 01 perform code goes here
    auto [audioSamples, audioStimulus, eegIndex] = loadAudioFromMat(inputAudioFilePath, 0);

    // Convert eigen matrix to vector
    std::vector<float> audioSamplesVec(audioSamples.data(),
                                       audioSamples.data() + audioSamples.size());

    // Convert float vector to double vector
    std::vector<double> audioSamplesDoubleVec(audioSamplesVec.begin(), audioSamplesVec.end());

    Wav w(44100, 16, 1, audioSamplesDoubleVec.data(), audioSamplesDoubleVec.size());
    
    std::filesystem::path inputPath(inputAudioFilePath);
    std::string outputFilename = inputPath.stem().string() + "_Output.wav";
    std::string outputWavPath = (inputPath.parent_path() / outputFilename).string();
    
    w.write(outputWavPath);

    // Use audioSamples, audioStimulus, and eegIndex as needed
    (void) audioSamples;
    (void) audioStimulus;
    (void) eegIndex;
    // For example, print the audio stimulus and EEG index
    cout << getAudioFlagName(AudioFlag::Stimulus) << ": " << audioStimulus << '\n';
    cout << getAudioFlagName(AudioFlag::EEG_Index) << ": " << eegIndex << '\n';

    auto [eegSamples, eegInfo] = loadEEGFromMat(
        "/home/ensismoebius/Documentos/UNESP/doutorado/"
        "databases/BaseDeDatosHablaImaginada/S01/"
        "S01_EEG.mat",
        0);
    // Use eegSamples and eegInfo as needed
    (void) eegSamples;
    (void) eegInfo;
    // For example, print the EEG info from the EEG file
    cout << "EEG Info from EEG file: " << getEEGFlagName(EEGFlag::Modality) << "="
         << eegInfo[static_cast<int>(EEGFlag::Modality)] << ", "
         << getEEGFlagName(EEGFlag::Stimulus) << "=" << eegInfo[static_cast<int>(EEGFlag::Stimulus)]
         << ", " << getEEGFlagName(EEGFlag::Artifact) << "="
         << eegInfo[static_cast<int>(EEGFlag::Artifact)] << '\n';
}

auto main(int argc, char** argv) -> int
{
    init();
    // Experiment 01 code goes here
    std::string audioFilePath =
        "/home/ensismoebius/Documentos/UNESP/doutorado/"
        "databases/BaseDeDatosHablaImaginada/S01/"
        "S01_Audio.mat";
    perform(audioFilePath);
    return 0;
}