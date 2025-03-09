#include <lsl_cpp.h>
#include <portaudio.h>
#include <iostream>
#include <cmath>

// Configuration
const int SAMPLE_RATE = 44100;
const int FRAMES_PER_BUFFER = 256;
const int NUM_CHANNELS = 1;
const int CONSOLE_WIDTH = 50;

// LSL outlets (global for simplicity)
lsl::stream_outlet *audio_outlet = nullptr;
lsl::stream_outlet *sine_outlet = nullptr;

// PortAudio callback
int audioCallback(const void *data, void * /*output*/,
                  unsigned long frameCount,
                  const PaStreamCallbackTimeInfo * /*timeInfo*/,
                  PaStreamCallbackFlags /*statusFlags*/,
                  void * /*userData*/)
{
    const float *samples = static_cast<const float *>(data);

    // Push audio data to LSL
    if (audio_outlet)
    {
        for (unsigned i = 0; i < frameCount; i++)
            audio_outlet->push_sample(&samples[i]);
    }

    // Generate and push sine wave data to LSL
    if (sine_outlet)
    {
        static float phase = 0.0f;
        const float phase_inc = 2.0f * M_PI * 2.0f / SAMPLE_RATE; // 440 Hz sine wave

        for (unsigned i = 0; i < frameCount; i++)
        {
            float sine_sample = sin(phase);
            sine_outlet->push_sample(&sine_sample);
            phase += phase_inc;
            if (phase > 2.0f * M_PI)
                phase -= 2.0f * M_PI;
        }
    }

    return paContinue;
}

int main()
{
    // Create LSL streams
    lsl::stream_info audio_info("AudioStream_subject00", "Audio", NUM_CHANNELS, SAMPLE_RATE, lsl::cf_float32);
    audio_outlet = new lsl::stream_outlet(audio_info);

    lsl::stream_info sine_info("SineWave_subject00", "Synthetic", 1, SAMPLE_RATE, lsl::cf_float32);
    sine_outlet = new lsl::stream_outlet(sine_info);

    // Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError)
    {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }

    // Open stream
    PaStream *stream;
    err = Pa_OpenDefaultStream(&stream,
                               NUM_CHANNELS,
                               0,
                               paFloat32,
                               SAMPLE_RATE,
                               FRAMES_PER_BUFFER,
                               audioCallback,
                               nullptr);

    if (err != paNoError)
    {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return 1;
    }

    // Start stream
    err = Pa_StartStream(stream);
    if (err != paNoError)
    {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        Pa_Terminate();
        return 1;
    }

    std::cout << "Streaming audio and sine wave... Press Enter to stop." << std::endl;
    std::cin.get(); // Wait for user input

    // Cleanup
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    delete audio_outlet;
    delete sine_outlet;

    return 0;
}