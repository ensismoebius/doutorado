#include "AudioCapture.hpp"
#include <iostream>

// Constructor: Initializes ring buffer size as the next power of two
AudioCapture::AudioCapture(size_t buffer_size)
    : ring_buffer_size_(1 << static_cast<size_t>(std::log2(buffer_size) + 1)),
      ring_buffer_mask_(ring_buffer_size_ - 1)
{
    ring_buffer_.resize(ring_buffer_size_);
}

// Destructor: Ensures audio capture stops before destruction
AudioCapture::~AudioCapture()
{
    stop();
}

// Starts audio capture
bool AudioCapture::start()
{
    if (running_)
        return true;

    if (!setup_stream())
        return false;

    PaError err = Pa_StartStream(stream_);
    if (err != paNoError)
    {
        last_error_ = "PortAudio error: " + std::string(Pa_GetErrorText(err));
        Pa_CloseStream(stream_);
        Pa_Terminate();
        return false;
    }

    running_ = true;
    return true;
}

// Stops audio capture
void AudioCapture::stop()
{
    if (running_)
    {
        Pa_StopStream(stream_);
        Pa_CloseStream(stream_);
        Pa_Terminate();
        running_ = false;
    }
}

// Checks if the audio capture is running
bool AudioCapture::is_running() const
{
    return running_;
}

// Retrieves the last error message
const std::string &AudioCapture::last_error() const
{
    return last_error_;
}

// Retrieves available audio samples from the ring buffer
std::vector<float> AudioCapture::get_available_samples()
{
    size_t avail = available_samples();
    std::vector<float> result;
    result.reserve(avail);

    size_t current_read = read_pos_;
    const size_t current_write = write_pos_;

    while (current_read != current_write)
    {
        result.push_back(ring_buffer_[current_read & ring_buffer_mask_]);
        current_read++;
    }

    read_pos_ = current_read;
    return result;
}

// Returns the number of available samples in the ring buffer
size_t AudioCapture::available_samples() const
{
    return write_pos_ - read_pos_;
}

// Handles audio data and writes to the ring buffer
void AudioCapture::handle_audio_data(const float *input, size_t frame_count)
{
    size_t current_write = write_pos_;

    for (size_t i = 0; i < frame_count; ++i)
    {
        ring_buffer_[current_write & ring_buffer_mask_] = input[i];
        current_write++;
    }

    write_pos_ = current_write;
}

// PortAudio callback function (static)
int AudioCapture::static_audio_callback(
    const void *input,
    void *output,
    unsigned long frameCount,
    const PaStreamCallbackTimeInfo *timeInfo,
    PaStreamCallbackFlags statusFlags,
    void *userData)
{
    static_cast<AudioCapture *>(userData)->handle_audio_data(
        static_cast<const float *>(input),
        frameCount);

    return paContinue;
}

// Initializes the PortAudio stream
bool AudioCapture::setup_stream()
{
    PaError err = Pa_Initialize();
    if (err != paNoError)
    {
        last_error_ = "PortAudio error: " + std::string(Pa_GetErrorText(err));
        return false;
    }

    err = Pa_OpenDefaultStream(
        &stream_,
        NUM_CHANNELS,      // Input channels
        0,                 // Output channels
        paFloat32,         // Sample format
        SAMPLE_RATE,       // Sample rate
        FRAMES_PER_BUFFER, // Frames per buffer
        &AudioCapture::static_audio_callback,
        this);

    if (err != paNoError)
    {
        last_error_ = "PortAudio error: " + std::string(Pa_GetErrorText(err));
        Pa_Terminate();
        return false;
    }

    return true;
}
