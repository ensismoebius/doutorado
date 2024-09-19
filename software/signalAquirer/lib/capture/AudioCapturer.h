#ifndef AUDIO_CAPTURER_H
#define AUDIO_CAPTURER_H

#include <alsa/asoundlib.h>
#include <vector>

class AudioCapturer
{
public:
    AudioCapturer(unsigned int sampleRate = 44100, int channels = 1);
    ~AudioCapturer();

    void captureAudio(std::vector<short> *buffer, int bufferSize);

private:
    snd_pcm_t *handle;
    unsigned int sampleRate;
    int channels;
};

#endif // AUDIO_CAPTURER_H