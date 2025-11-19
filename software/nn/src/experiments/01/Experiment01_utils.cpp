#include "Experiment01_utils.hpp"

#include <fftw3.h>
#include <sndfile.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "core/dataLoaders/10.1117/AudioLoader.h"
#include "core/dataLoaders/10.1117/EEGLoader.h"
#include "core/optimizers/Adam.hpp"

using nn::dataLoaders::loadAudioFromMat;
using nn::dataLoaders::loadEEGFromMat;
using std::cout;
using std::make_shared;
using std::string;

using std::min;
using std::size_t;
using std::vector;

// ------------------ Parâmetros (igual ao Python) ------------------
const int FS_TARGET = 44100;
const double FRAME_MS = 25.0;
const double FRAME_SHIFT_MS = 10.0;
const int NFFT = 512;
const int N_FILTERS = 24;
const int N_CEPS = 19;
const double PREEMPH = 0.97;
const int DELTA_WINDOW = 2;

static inline void pre_emphasis_inplace(vector<double>& x, double a = PREEMPH)
{
    for (size_t i = x.size() - 1; i >= 1; --i)
    {
        x[i] = x[i] - a * x[i - 1];
    }
    // x[0] permanece
}

auto framing_and_window(const vector<double>& x, int fs, int& frame_len, int& frame_step)
    -> vector<vector<double>>
{
    frame_len = (int) round(FRAME_MS * fs / 1000.0);
    frame_step = (int) round(FRAME_SHIFT_MS * fs / 1000.0);
    int N = (int) x.size();
    int num_frames = 1 + std::max(0, (N - frame_len) / frame_step);
    int pad_len = (num_frames * frame_step) + frame_len;
    vector<double> xpad = x;
    xpad.resize(pad_len, 0.0);
    vector<double> win(frame_len);

    for (int n = 0; n < frame_len; ++n)
    {
        win[n] = 0.54 - 0.46 * cos(2 * M_PI * n / (frame_len - 1)); // Hamming
    }

    vector<vector<double>> frames(num_frames, vector<double>(frame_len));
    for (int i = 0; i < num_frames; ++i)
    {
        int start = i * frame_step;
        for (int j = 0; j < frame_len; ++j)
        {
            frames[i][j] = xpad[start + j] * win[j];
        }
    }
    return frames;
}

auto rfft_power(const vector<vector<double>>& frames, int nfft) -> vector<vector<double>>
{
    size_t num_frames = frames.size();
    size_t nbin = (nfft / 2) + 1;

    vector<vector<double>> P(num_frames, vector<double>(nbin, 0.0));

    // usar FFTW
    auto* in = (double*) fftw_malloc(sizeof(double) * nfft);

    auto* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (nfft / 2 + 1));

    fftw_plan p = fftw_plan_dft_r2c_1d(nfft, in, out, FFTW_ESTIMATE);

    for (int f = 0; f < num_frames; ++f)
    {
        // copiar frame e zerar o resto
        size_t L = frames[f].size();
        for (size_t i = 0; i < L; ++i)
        {
            in[i] = frames[f][i];
        }
        for (size_t i = L; i < nfft; ++i)
        {
            in[i] = 0.0;
        }
        fftw_execute(p);
        for (size_t k = 0; k < nbin; ++k)
        {
            double re = out[k][0];
            double im = out[k][1];
            P[f][k] = (re * re + im * im) / (double) nfft; // potência
        }
    }
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return P;
}

void build_linear_filterbank(int nfft, int fs, int n_filters, vector<vector<double>>& fb,
                             vector<double>& centers)
{
    int nbin = (nfft / 2) + 1;
    fb.assign(n_filters, vector<double>(nbin, 0.0));
    double fmax = fs / 2.0;
    centers.resize(n_filters + 2);
    for (int i = 0; i < n_filters + 2; ++i)
    {
        centers[i] = (fmax) * (double) i / (n_filters + 1);
    }
    vector<int> bins(n_filters + 2);
    for (int i = 0; i < (int) centers.size(); ++i)
    {
        bins[i] = (int) floor((nfft + 1) * centers[i] / fs);
    }
    for (int m = 1; m <= n_filters; ++m)
    {
        int f_m_minus = bins[m - 1];
        int f_m = bins[m];
        int f_m_plus = bins[m + 1];
        if (f_m > f_m_minus)
        {
            for (int k = f_m_minus; k < f_m; ++k)
            {
                fb[m - 1][k] = (double) (k - f_m_minus) / (double) (f_m - f_m_minus);
            }
        }
        if (f_m_plus > f_m)
        {
            for (int k = f_m; k < f_m_plus; ++k)
            {
                fb[m - 1][k] = (double) (f_m_plus - k) / (double) (f_m_plus - f_m);
            }
        }
    }
}

auto dot_power_filterbank(const vector<vector<double>>& P, const vector<vector<double>>& fb)
    -> vector<vector<double>>
{
    size_t frames = P.size();
    size_t n_filters = fb.size();
    vector<vector<double>> E(frames, vector<double>(n_filters, 0.0));
    size_t nbin = P[0].size();
    for (size_t t = 0; t < frames; ++t)
    {
        for (size_t m = 0; m < n_filters; ++m)
        {
            double sum = 0.0;
            for (size_t k = 0; k < nbin; ++k)
            {
                sum += P[t][k] * fb[m][k];
            }
            sum = std::max(sum, 1e-12);
            E[t][m] = log(sum);
        }
    }
    return E;
}

// DCT-II (normalized "ortho") simples via definição
auto dct2(const vector<vector<double>>& X, int ncep) -> vector<vector<double>>
{
    size_t frames = X.size();
    size_t M = X[0].size();
    vector<vector<double>> C(frames, vector<double>(ncep, 0.0));
    for (int t = 0; t < frames; ++t)
    {
        for (int i = 0; i < ncep; ++i)
        {
            double sum = 0.0;
            for (int m = 0; m < M; ++m)
            {
                sum += X[t][m] * cos(M_PI * i * (m + 0.5) / M);
            }
            // normalização ortho
            C[t][i] = sum * sqrt(2.0 / static_cast<double>(M));
            if (i == 0)
            {
                C[t][i] *= 1.0 / std::numbers::sqrt2;
            }
        }
    }
    return C;
}

// Delta (derivada temporal)
auto compute_deltas(const vector<vector<double>>& feat, int N = DELTA_WINDOW)
    -> vector<vector<double>>
{
    size_t T = feat.size();
    size_t D = feat[0].size();
    vector<vector<double>> pad(T + (static_cast<size_t>(2 * N)), vector<double>(D));
    // pad edges
    for (int t = 0; t < N; ++t)
    {
        pad[t] = feat[0];
    }
    for (int t = 0; t < T; ++t)
    {
        pad[t + N] = feat[t];
    }
    for (int t = 0; t < N; ++t)
    {
        pad[T + N + t] = feat[T - 1];
    }
    double denom = 0.0;
    for (int i = 1; i <= N; ++i)
    {
        denom += i * i;
    }
    denom *= 2.0;
    vector<vector<double>> del(T, vector<double>(D, 0.0));
    for (int t = 0; t < T; ++t)
    {
        for (int d = 0; d < D; ++d)
        {
            double num = 0.0;
            for (int n = 1; n <= N; ++n)
            {
                num += n * (pad[t + N + n][d] - pad[t + N - n][d]);
            }
            del[t][d] = num / denom;
        }
    }
    return del;
}

static auto loadAndProcessAudio(const std::string& audioFilePath, float window_size_sec,
                                float overlap_ratio, int sampling_rate)
    -> std::vector<Eigen::MatrixXf>
{
    auto [audioSamples, audioStimulus, eegIndex] = loadAudioFromMat(audioFilePath, 0);
    Eigen::MatrixXf audioMatrix = audioSamples.transpose(); // Convert to 1xN matrix for windowing

    SF_INFO sfinfo;
    SNDFILE* snd = sf_open(audioFilePath.c_str(), SFM_READ, &sfinfo);

    if (snd == nullptr)
    {
        std::cerr << "Erro abrindo " << audioFilePath << "\n";
        return {};
    }

    if (sfinfo.samplerate != FS_TARGET)
    {
        std::cerr << "Amostragem diferente de " << FS_TARGET << " Hz. Reamostrar externamente.\n";
        sf_close(snd);
        return {};
    }

    vector<double> input_data(sfinfo.frames * sfinfo.channels);
    sf_readf_double(snd, input_data.data(), sfinfo.frames);

    // se estéreo -> média
    if (sfinfo.channels > 1)
    {
        vector<double> mono(sfinfo.frames);
        for (int i = 0; i < sfinfo.frames; ++i)
        {
            double s = 0.0;
            for (int c = 0; c < sfinfo.channels; ++c)
            {
                s += input_data[(i * sfinfo.channels) + c];
            }
            mono[i] = s / sfinfo.channels;
        }
        input_data.swap(mono);
    }
    sf_close(snd);

    // 1) pré-ênfase
    pre_emphasis_inplace(input_data);

    // 2) framing+janelamento
    int frame_len;
    int frame_step;
    auto frames = framing_and_window(input_data, FS_TARGET, frame_len, frame_step);

    // 3-4) FFT + potência
    auto P = rfft_power(frames, NFFT);

    // 5) banco linear
    vector<vector<double>> fb;
    vector<double> centers;
    build_linear_filterbank(NFFT, FS_TARGET, N_FILTERS, fb, centers);

    // 6) energy por banda + log
    auto logE = dot_power_filterbank(P, fb);

    // 7) DCT
    auto ceps = dct2(logE, N_CEPS);

    // 8) deltas
    auto delta = compute_deltas(ceps);
    auto delta2 = compute_deltas(delta);

    // 9) concatenar e imprimir (simples: print primeiros 5 frames)
    size_t T = ceps.size();
    for (size_t t = 0; t < min(T, static_cast<size_t>(5)); ++t)
    {
        for (int i = 0; i < N_CEPS; ++i)
        {
            cout << ceps[t][i] << " ";
        }
        for (int i = 0; i < N_CEPS; ++i)
        {
            cout << delta[t][i] << " ";
        }
        for (int i = 0; i < N_CEPS; ++i)
        {
            cout << delta2[t][i] << " ";
        }
        cout << "\n";
    }

    return {}; // Retornar vetor vazio por enquanto
}

void processSubject(const std::string& subjectPath, const std::string& subjectName,
                    const std::string& audioFilePath, const std::string& eegFilePath)
{
    cout << "Processing subject: " << subjectName << '\n';

    // Define experiment parameters
    const float window_size_sec = 1.5F;
    const float overlap_ratio = 0.5F;
    const int audio_sampling_rate = 44100; // Assuming 44.1 kHz for audio
    const int eeg_sampling_rate = 1024;    // Assuming 1024 Hz for EEG

    // Load, normalize, and window audio data
    std::vector<Eigen::MatrixXf> audioWindows =
        loadAndProcessAudio(audioFilePath, window_size_sec, overlap_ratio, audio_sampling_rate);
    cout << "  - Loaded and processed " << audioWindows.size() << " audio windows.\n";
}
