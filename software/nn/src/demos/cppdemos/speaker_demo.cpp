#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "nn/layers/Leaky.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/Sequential.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/wave/Wav.h"
#include "nn/wave/filter_operations.hpp"
#include "nn/wavelet/waveletOperations.h"

using namespace std;
using namespace nn;

// Simple helper: parse int suffix for wavelet name like "db4" -> 4
static int parse_wavelet_order(const string& wavelet_name)
{
    for (size_t i = 0; i < wavelet_name.size(); ++i)
    {
        if (isdigit(wavelet_name[i]))
        {
            return stoi(wavelet_name.substr(i));
        }
    }
    return 4; // default to 4
}

// Compute RMS energies per subband using wavelet malat + extract_subband_energies
static vector<double> extract_subband_rms(const vector<double>& window, int sampling_rate,
                                          const string& wavelet_name)
{
    int order = parse_wavelet_order(wavelet_name);
    // Create a lowpass filter; choose a conservative final frequency (nyquist)
    double finalFreq = sampling_rate / 2.0;
    auto lp = createLowPassFilter(order, sampling_rate, finalFreq);

    auto res = wavelets::malat(window,
                               std::span<const double>(lp.data(), lp.size()),
                               wavelets::PACKET_WAVELET,
                               /*level=*/1);

    auto energies = wavelets::extract_subband_energies(res, /*level=*/1);
    return energies;
}

// Poisson encode normalized [0,1] vector into a sequence of spike tensors
static vector<Tensor> poisson_encode(const vector<double>& features, int steps_per_window,
                                     float time_step = 0.1f)
{
    std::mt19937 gen(static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    int n_bands = static_cast<int>(features.size());
    vector<Tensor> seq;
    seq.reserve(steps_per_window);

    for (int t = 0; t < steps_per_window; ++t)
    {
        Tensor spike(1, n_bands);
        spike.setZero();
        for (int j = 0; j < n_bands; ++j)
        {
            float p = static_cast<float>(features[j]) * time_step; // simple mapping
            p = std::clamp(p, 0.0f, 1.0f);
            if (dist(gen) < p) spike.at(0, j) = 1.0f;
        }
        seq.emplace_back(spike);
    }
    return seq;
}

int main(int argc, char** argv)
{
    ios::sync_with_stdio(false);

    // Simple CLI: speaker_demo demo --input sample.wav --wavelet db4 --steps 10
    if (argc < 2)
    {
        cout << "Usage: " << argv[0]
             << " demo [--input file.wav] [--wavelet db4] [--num-bands N] [--steps S]" << endl;
        return 1;
    }

    string cmd = argv[1];
    if (cmd != string("demo"))
    {
        cerr << "Only 'demo' subcommand is implemented in this C++ port." << endl;
        return 1;
    }

    string input_wav = "src/demos/pydemos/dados/vozes/alice/amostra_20260118_201843.wav";
    string wavelet = "db4";
    int num_bands = 100;
    int steps = 10;

    for (int i = 2; i < argc; ++i)
    {
        string a = argv[i];
        if (a == "--input" && i + 1 < argc)
            input_wav = argv[++i];
        else if (a == "--wavelet" && i + 1 < argc)
            wavelet = argv[++i];
        else if (a == "--num-bands" && i + 1 < argc)
            num_bands = stoi(argv[++i]);
        else if (a == "--steps" && i + 1 < argc)
            steps = stoi(argv[++i]);
    }

    try
    {
        // 1) Read WAV
        Wav w;
        w.read(input_wav);
        const auto& data = w.get_data_left();
        if (data.empty())
        {
            cerr << "No audio data read from " << input_wav << endl;
            return 1;
        }
        int fs = static_cast<int>(w.get_path().empty() ? 44100 : w.get_path().length()); // fallback

        // NOTE: Wav doesn't expose sampling rate getter directly; hack: read header via object
        // internals if needed. For the demo assume standard sampling rate 44100
        fs = 44100;

        // 2) Windowing: fixed window length equal to entire file or subwindows (simple: one window)
        vector<double> window(data.begin(), data.end());

        // 3) Extract subband RMS energies via wavelet
        auto energies = extract_subband_rms(window, fs, wavelet);

        // Reduce or interpolate energies to requested num_bands
        vector<double> bands;
        if (static_cast<int>(energies.size()) == num_bands)
        {
            bands = energies;
        }
        else if (static_cast<int>(energies.size()) > num_bands)
        {
            bands.assign(energies.begin(), energies.begin() + num_bands);
        }
        else
        {
            // upsample by repeating
            bands.reserve(num_bands);
            for (int i = 0; i < num_bands; ++i)
            {
                bands.push_back(energies[i % energies.size()]);
            }
        }

        // Normalize bands to [0,1]
        double maxv = *max_element(bands.begin(), bands.end());
        if (maxv <= 0.0) maxv = 1.0;
        for (double& b : bands) b = b / maxv;

        // 4) Poisson encode
        float time_step = 1.0f / static_cast<float>(steps);
        auto spike_seq = poisson_encode(bands, steps, time_step);

        // 5) Build a tiny SNN: Linear -> Leaky -> Linear -> Leaky
        int hidden = num_bands; // keep same dimensionality for didactic plots
        auto seq_model = Sequential({
            make_shared<Linear>(num_bands, hidden),
            make_shared<Leaky>(1.0F, 1.0F, 1.0F, 1.0F),
            make_shared<Linear>(hidden, num_bands),
            make_shared<Leaky>(1.0F, 1.0F, 1.0F, 1.0F),
        });

        // 6) Run through SNN timesteps
        Tensor sum_output(1, num_bands);
        sum_output.setZero();

        for (int t = 0; t < steps; ++t)
        {
            Tensor spike_in = spike_seq[t];
            Tensor out = seq_model.forward(spike_in, /*requires_grad=*/false);
            // For interpretability we treat "out" as spikes (it already is from Leaky)
            sum_output = sum_output.add(out);
        }

        // 7) Print aggregated output (sum of spikes per band)
        cout << "Aggregated output spikes (per band):\n";
        for (int j = 0; j < sum_output.cols(); ++j)
        {
            cout << sum_output.at(0, j);
            if (j + 1 < sum_output.cols()) cout << ",";
        }
        cout << "\n";
    }
    catch (const std::exception& ex)
    {
        cerr << "Error: " << ex.what() << endl;
        return 2;
    }

    return 0;
}
