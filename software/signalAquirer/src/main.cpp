#include <iostream>
#include <vector>
#include <portaudio.h>

#define SAMPLE_RATE 44100
#define FRAMES_PER_BUFFER 256

class AudioCapture {
public:
    AudioCapture(int maxFrames) : maxFrameIndex(maxFrames), frameIndex(0) {
        buffer.resize(maxFrameIndex);
    }

    ~AudioCapture() {
        stop();
    }

    bool start() {
        PaError err;

        // Inicializa o PortAudio
        err = Pa_Initialize();
        if (err != paNoError) {
            std::cerr << "Erro ao inicializar o PortAudio: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }

        // Abre um stream de áudio para captura
        err = Pa_OpenStream(&stream, nullptr, nullptr, SAMPLE_RATE, FRAMES_PER_BUFFER, paClipOff, audioCallback, this);
        if (err != paNoError) {
            std::cerr << "Erro ao abrir o stream: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }

        // Inicia a captura
        err = Pa_StartStream(stream);
        if (err != paNoError) {
            std::cerr << "Erro ao iniciar o stream: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }

        frameIndex = 0; // Reseta o índice
        return true;
    }

    void stop() {
        if (stream) {
            Pa_StopStream(stream);
            Pa_CloseStream(stream);
            Pa_Terminate(); // Finaliza o PortAudio
            stream = nullptr;
        }
    }

    const std::vector<float>& getBuffer() const {
        return buffer;
    }

private:
    static int audioCallback(const void* inputBuffer, void* outputBuffer,
                             unsigned long framesPerBuffer,
                             const PaStreamCallbackTimeInfo* timeInfo,
                             PaStreamCallbackFlags statusFlags,
                             void* userData) {
        AudioCapture* data = static_cast<AudioCapture*>(userData);
        const float* in = static_cast<const float*>(inputBuffer);

        for (unsigned long i = 0; i < framesPerBuffer; i++) {
            if (data->frameIndex < data->maxFrameIndex) {
                data->buffer[data->frameIndex++] = *in++; // Armazena o dado de áudio no buffer
            }
        }

        return paContinue; // Continua a captura
    }

    PaStream* stream = nullptr;
    std::vector<float> buffer;
    int frameIndex;
    int maxFrameIndex;
};

int main() {
    const int maxFrames = SAMPLE_RATE; // Captura 1 segundo de áudio
    AudioCapture audioCapture(maxFrames);

    if (!audioCapture.start()) {
        return -1; // Saída se a captura falhar
    }

    std::cout << "Gravando... Pressione Enter para parar." << std::endl;
    std::cin.get(); // Espera pela entrada do usuário

    audioCapture.stop(); // Para a captura

    // Exibe alguns dados gravados
    const auto& buffer = audioCapture.getBuffer();
    std::cout << "Dados gravados: " << std::endl;
    for (size_t i = 0; i < audioCapture.getBuffer().size(); i++) {
        std::cout << buffer[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
