#include <filesystem>
#include <string>

#include "audioTypes.h"
#include "experiments/01/Experiment01_utils.h"

using std::string;

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
            string subjectPath = entry.path().string();
            string subjectName = entry.path().filename().string();
            string audioFilePath = subjectPath;
            audioFilePath += "/";
            audioFilePath += subjectName;
            audioFilePath += "_Audio.mat";
            string eegFilePath = subjectPath;
            eegFilePath += "/";
            eegFilePath += subjectName;
            eegFilePath += "_EEG.mat";

            if (std::filesystem::exists(audioFilePath) && std::filesystem::exists(eegFilePath))
            {
                const SubjectInfo subject = {.path = subjectPath,
                                             .name = subjectName,
                                             .audio_file_path = audioFilePath,
                                             .eeg_file_path = eegFilePath};
                processSubject(subject);
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