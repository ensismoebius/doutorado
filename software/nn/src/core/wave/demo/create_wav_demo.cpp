#include <cmath>

#include "../Wav.h"
#include "../filtersOperations.h"

auto main() -> int
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

    // This is used for nothing but to test the createAlpha function
    double alpha = createAlpha(sampleRate, 2000.0);

    wavFile.write("test.wav");
    return 0;
}