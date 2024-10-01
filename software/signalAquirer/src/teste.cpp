#include <iostream>
#include <portaudio.h>

class AudioRecorder {
public:
    AudioRecorder() : stream(nullptr) {}

    ~AudioRecorder() {
        Pa_Terminate();
    }

    bool initialize() {
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            std::cerr << "Erro ao inicializar o PortAudio: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }

        inputParameters.device = Pa_GetDeviceCount() > 1 ? 1 : Pa_GetDefaultInputDevice();
        if (inputParameters.device == paNoDevice) {
            std::cerr << "Nenhum dispositivo de áudio encontrado." << std::endl;
            return false;
        }

        inputParameters.channelCount = 1;  // Captura mono
        inputParameters.sampleFormat = paInt16;
        inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
        inputParameters.hostApiSpecificStreamInfo = nullptr;

        err = Pa_OpenStream(&stream, &inputParameters, nullptr, 44100, 256, paClipOff, audioCallback, this);
        if (err != paNoError) {
            std::cerr << "Erro ao abrir o stream: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }

        return true;
    }

    static int audioCallback(const void* input, void* output,
                             unsigned long frameCount,
                             const PaStreamCallbackTimeInfo* timeInfo,
                             PaStreamCallbackFlags statusFlags,
                             void* userData) {
        const int16_t* in = static_cast<const int16_t*>(input);
        AudioRecorder* recorder = static_cast<AudioRecorder*>(userData);

        // Processar o áudio capturado
        recorder->processAudio(in, frameCount);

        return paContinue;  // Continue capturando
    }

    void processAudio(const int16_t* input, unsigned long frameCount) {
        std::cout << "Frame capturado: ";
        for (unsigned long i = 0; i < frameCount; ++i) {
            std::cout << input[i] << " ";  // Exibe os primeiros valores capturados
        }
        std::cout << std::endl;
    }

    bool startRecording() {
        PaError err = Pa_StartStream(stream);
        if (err != paNoError) {
            std::cerr << "Erro ao iniciar o stream: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }
        return true;
    }

    void stopRecording() {
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
    }

private:
    PaStream* stream;
    PaStreamParameters inputParameters;
};

int main() {
    AudioRecorder recorder;
    if (!recorder.initialize()) {
        return -1;
    }

    std::cout << "Iniciando gravação..." << std::endl;
    if (!recorder.startRecording()) {
        return -1;
    }

    Pa_Sleep(5000);  // Grava por 5 segundos

    recorder.stopRecording();
    std::cout << "Gravação concluída." << std::endl;

    return 0;
}
