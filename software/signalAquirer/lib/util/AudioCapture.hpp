#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#include <portaudio.h>
#include <vector>
#include <atomic>
#include <string>
#include <cmath>

class AudioCapture
{
public:
    explicit AudioCapture(size_t buffer_size = 44100 * 5); // 5 seconds buffer
    ~AudioCapture();

    bool start();
    void stop();
    bool is_running() const;

    std::vector<float> get_available_samples();
    const std::string &last_error() const;

    static const int SAMPLE_RATE = 44100;
    static const int FRAMES_PER_BUFFER = 256;
    static const int NUM_CHANNELS = 1;

private:
    std::atomic<bool> running_;
    std::string last_error_;
    PaStream *stream_;

    // Ring buffer components
    std::vector<float> ring_buffer_;
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
    const size_t ring_buffer_size_;
    const size_t ring_buffer_mask_;

    static int static_audio_callback(
        const void *input, void *output,
        unsigned long frameCount,
        const PaStreamCallbackTimeInfo *timeInfo,
        PaStreamCallbackFlags statusFlags,
        void *userData);

    void handle_audio_data(const float *input, size_t frame_count);
    size_t available_samples() const;
};
#endif // AUDIO_CAPTURE_H