#include "AudioCapturer.h"
#include <iostream>

AudioCapturer::AudioCapturer(unsigned int sampleRate, int channels) : sampleRate(sampleRate), channels(channels)
{
    int rc;
    snd_pcm_hw_params_t *params;
    snd_pcm_sw_params_t *sw_params;
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

    rc = snd_pcm_hw_params(handle, params);
    if (rc < 0)
    {
        std::cerr << "Failed to set hardware parameters: " << snd_strerror(rc) << std::endl;
        snd_pcm_close(handle);
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_sw_params_current(handle, sw_params);
    if (rc < 0)
    {
        std::cerr << "Failed to get software parameters: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_sw_params_set_start_threshold(handle, sw_params, 0);
    if (rc < 0)
    {
        std::cerr << "Failed to set start threshold: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }

    rc = snd_pcm_sw_params(handle, sw_params);
    if (rc < 0)
    {
        std::cerr << "Failed to set software parameters: " << snd_strerror(rc) << std::endl;
        exit(EXIT_FAILURE);
    }
}

AudioCapturer::~AudioCapturer()
{
    snd_pcm_close(handle);
}

std::vector<short> AudioCapturer::captureAudio(int bufferSize)
{
    std::vector<short> buffer(bufferSize);
    
    int rc = snd_pcm_readi(handle, buffer.data(), bufferSize / channels);
    if (rc < 0)
    {
        std::cerr << "Error reading audio data: " << snd_strerror(rc) << std::endl;
        // Handle error appropriately, e.g., return an empty vector
        return std::vector<short>();
    }
    return buffer;
}
