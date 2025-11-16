#include <filesystem> // Include for path manipulation
#include <iostream>
#include <map> // Added for std::map
#include <string>

#include "core/dataLoaders/10.1117/AudioLoader.h"
#include "core/dataLoaders/10.1117/EEGLoader.h"
#include "core/wave/Wav.h"

using std::cout;

using namespace nn::dataLoaders;

// Map stimulus IDs to their string representations
const std::map<int, std::string> stimulusNames = {{1, "A"},
                                                  {2, "E"},
                                                  {3, "I"},
                                                  {4, "O"},
                                                  {5, "U"},
                                                  {6, "Arriba"},
                                                  {7, "Abajo"},
                                                  {8, "Adelante"},
                                                  {9, "Atras"},
                                                  {10, "Derecha"},
                                                  {11, "Izquierda"}};

static void init()
{
    // Initialization code goes here
}

static void perform(const std::string& basePath)
{
    for (const auto& entry : std::filesystem::directory_iterator(basePath))
    {
        if (entry.is_directory())
        {
            std::string subjectPath = entry.path().string();
            std::string subjectName = entry.path().filename().string();
            std::string audioFilePath = subjectPath + "/" + subjectName + "_Audio.mat";
            std::string eegFilePath = subjectPath + "/" + subjectName + "_EEG.mat";

            if (std::filesystem::exists(audioFilePath) && std::filesystem::exists(eegFilePath))
            {
                cout << "Processing subject: " << subjectName << '\n';

                auto [audioSamples, audioStimulus, eegIndex] = loadAudioFromMat(audioFilePath, 0);

                // The eegIndex from the audio file is 1-based, so we subtract 1 for 0-based
                // indexing
                long eegRowIndex = static_cast<long>(eegIndex) - 1;

                auto [eegSamplesMatrix, eegInfo] = loadEEGFromMat(eegFilePath, eegRowIndex);

                // Split EEG data into channels
                constexpr int numChannels = 6;
                std::vector<Eigen::VectorXf> eegChannels(numChannels);

                for (int j = 0; j < numChannels; ++j)
                {
                    eegChannels[j] = eegSamplesMatrix.row(j);
                }

                // Now you have:
                // - audioSamples (Eigen::VectorXf)
                // - eegChannels (std::vector of 6 Eigen::VectorXf)
                // You can proceed with further processing...

                cout << "  - Loaded Audio Sample linked to EEG Sample " << eegRowIndex << '\n';

                // Get stimulus name for audio
                std::string audioStimulusName = "Unknown";
                if (stimulusNames.contains(audioStimulus))
                {
                    audioStimulusName = stimulusNames.at(audioStimulus);
                }

                // Example: Write audio to a WAV file
                std::vector<float> audioSamplesVec(audioSamples.data(),
                                                   audioSamples.data() + audioSamples.size());
                std::vector<double> audioSamplesDoubleVec(audioSamplesVec.begin(),
                                                          audioSamplesVec.end());

                Wav w(44100, 16, 1, audioSamplesDoubleVec.data(), audioSamplesDoubleVec.size());
                std::filesystem::path subjectDirPath(subjectPath);
                std::string outputFilename =
                    subjectName + "_AudioSample_" + audioStimulusName + ".wav";
                std::string outputWavPath = (subjectDirPath / outputFilename).string();
                w.write(outputWavPath);
                cout << "  - Wrote Audio Sample to " << outputWavPath << '\n';

                // Get EEG stimulus from eegInfo
                int eegStimulus = eegInfo[1]; // Assuming index 1 is stimulus

                // Get stimulus name for EEG
                std::string eegStimulusName = "Unknown";
                if (stimulusNames.contains(eegStimulus))
                {
                    eegStimulusName = stimulusNames.at(eegStimulus);
                }

                // Write EEG channels to WAV files
                std::array<std::string, numChannels> eegChannelNames = {
                    "F3", "F4", "C3", "C4", "P3", "P4"};
                for (int j = 0; j < numChannels; ++j)
                {
                    std::vector<double> eegChannelDoubleVec(
                        eegChannels[j].data(), eegChannels[j].data() + eegChannels[j].size());
                    // Assuming EEG sample rate is 1024 Hz, 16-bit, 1 channel
                    Wav eegWav(1024, 16, 1, eegChannelDoubleVec.data(), eegChannelDoubleVec.size());
                    std::string eegOutputFilename = subjectName + "_EEG_Channel_" +
                                                    eegChannelNames[j] + "_" + eegStimulusName +
                                                    ".wav";
                    std::string eegOutputWavPath =
                        (std::filesystem::path(subjectPath) / eegOutputFilename).string();
                    eegWav.write(eegOutputWavPath);
                    cout << "  - Wrote EEG Channel "
                         << eegChannelNames[j] + " to " + eegOutputWavPath << '\n';
                }
            }
        }
    }
}

auto main(int argc, char** argv) -> int
{
    init();

    std::string basePath =
        "/home/ensismoebius/Documentos/UNESP/doutorado/"
        "databases/BaseDeDatosHablaImaginada/";

    perform(basePath);

    return 0;
}