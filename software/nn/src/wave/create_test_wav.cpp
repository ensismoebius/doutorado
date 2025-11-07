#include <cmath>

#include "Wav.h"

int main()
{
    Wav wavFile;
    std::vector<long double> data;

    // Generate a simple sine wave
    const double frequency = 440.0; // Hz
    const double sampleRate = 44100.0;
    const double duration = 0.1; // seconds

    for (int i = 0; i < sampleRate * duration; i++)
    {
        double t = i / sampleRate;
        data.push_back(std::sin(2.0 * M_PI * frequency * t));
    }

    wavFile.write("test.wav");
    return 0;
}