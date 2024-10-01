#include "AudioCapturer.h"
#include <iostream>

AudioCapturer::AudioCapturer(unsigned int sampleRate, int channels) : sampleRate(sampleRate), channels(channels)
{
    int rc;
    snd_pcm_hw_params_t *params;
    snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;

    rc = snd_pcm_open(&handle, "default", SND_PCM_STREAM_CAPTURE, 0);
    if (rc < 0)
    {
        std::cerr << "Failed to open PCM device: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    snd_pcm_hw_params_alloca(&params);

    rc = snd_pcm_hw_params_any(handle, params);
    if (rc < 0)
    {
        std::cerr << "Failed to initialize hardware parameters: " << snd_strerror(rc) << std::endl;
        snd_pcm_close(handle);
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_hw_params_set_access(handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (rc < 0)
    {
        std::cerr << "Failed to set access type: " << snd_strerror(rc) << std::endl;
        snd_pcm_close(handle);
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_hw_params_set_format(handle, params, format);
    if (rc < 0)
    {
        std::cerr << "Unable to set format: " << snd_strerror(rc) << std::endl;
        snd_pcm_close(handle);
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_hw_params_set_channels(handle, params, channels);
    if (rc < 0)
    {
        std::cerr << "Failed to set channels: " << snd_strerror(rc) << std::endl;
        snd_pcm_close(handle);
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_hw_params_set_rate_near(handle, params, &sampleRate, 0);
    if (rc < 0)
    {
        std::cerr << "Failed to set rate: " << snd_strerror(rc) << std::endl;
        snd_pcm_close(handle);
        exit(EXIT_FAILURE);
    }

    unsigned int actualRate = sampleRate;
    rc = snd_pcm_hw_params_set_rate_near(handle, params, &actualRate, 0);
    if (rc < 0 || actualRate != sampleRate)
    {
        std::cerr << "Warning: Requested sample rate (" << sampleRate
                  << ") differs from actual (" << actualRate << ")" << std::endl;
        sampleRate = actualRate; // Optionally update the internal sampleRate to match
    }

    rc = snd_pcm_hw_params(handle, params);
    if (rc < 0)
    {
        std::cerr << "Failed to set hardware parameters: " << snd_strerror(rc) << std::endl;
        snd_pcm_close(handle);
        exit(EXIT_FAILURE);
    }
}

AudioCapturer::~AudioCapturer()
{
    snd_pcm_close(handle);
}

void AudioCapturer::captureAudio(float *buffer, int bufferSize)
{
    int rc = snd_pcm_readi(handle, buffer, bufferSize / channels);

    if (rc < 0)
    {
        std::cerr << "Error reading audio data: " << snd_strerror(rc) << std::endl;
        return;
    }
}
