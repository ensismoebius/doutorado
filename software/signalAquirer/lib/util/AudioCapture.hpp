#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#include <portaudio.h>
#include <vector>
#include <atomic>
#include <string>
#include <cmath>
#include "ICapturer.hpp"

using namespace std;

/**
 * @brief AudioCapture class for handling real-time audio capture using PortAudio.
 */
class AudioCapture : public ICapturer
{
public:
    /**
     * @brief Constructs an AudioCapture instance with a specified buffer size.
     * @param buffer_size The desired size of the ring buffer (default: 5 seconds at 44.1kHz).
     */
    explicit AudioCapture(size_t buffer_size = 44100 * 5);

    /**
     * @brief Destructor, ensures audio capture stops before destruction.
     */
    ~AudioCapture();

    /**
     * @brief Starts the audio capture.
     * @return true if successful, false on failure.
     */
    bool start() override;

    /**
     * @brief Stops the audio capture.
     */
    void stop() override;

    /**
     * @brief Checks if the audio capture is running.
     * @return true if running, false otherwise.
     */
    bool isCapturing() const override;

    /**
     * @brief Retrieves the available audio samples from the ring buffer.
     * @return A vector containing the captured audio samples.
     */
    const vector<float> getAvailableSamples() override;

    /**
     * @brief Retrieves the last error message.
     * @return A string containing the last encountered error.
     */
    const string &last_error() const override;

    // Constants
    static constexpr int SAMPLE_RATE = 44100;
    static constexpr int FRAMES_PER_BUFFER = 2048;
    static constexpr int NUM_CHANNELS = 1;

private:
    atomic<bool> capturing{false}; ///< Indicates if audio capture is running.
    string last_error_;            ///< Stores the last error message.
    PaStream *stream_{nullptr};    ///< PortAudio stream handle.

    // Ring buffer
    vector<float> ring_buffer_;     ///< Circular buffer for storing audio samples.
    atomic<size_t> write_pos_{0};   ///< Write position in the ring buffer.
    atomic<size_t> read_pos_{0};    ///< Read position in the ring buffer.
    const size_t ring_buffer_size_; ///< Size of the ring buffer (power of two).
    const size_t ring_buffer_mask_; ///< Mask for efficient index wrapping.

    /**
     * @brief PortAudio callback function (static).
     */
    static int static_audio_callback(
        const void *input, void *output,
        unsigned long frameCount,
        const PaStreamCallbackTimeInfo *timeInfo,
        PaStreamCallbackFlags statusFlags,
        void *userData);

    /**
     * @brief Processes incoming audio data and stores it in the ring buffer.
     */
    void handle_audio_data(const float *input, size_t frame_count);

    /**
     * @brief Returns the number of available samples in the ring buffer.
     */
    size_t available_samples() const;

    /**
     * @brief Initializes the PortAudio stream.
     * @return true if successful, false otherwise.
     */
    bool setup_stream();
};

#endif // AUDIO_CAPTURE_H
