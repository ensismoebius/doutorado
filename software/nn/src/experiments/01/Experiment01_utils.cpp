#include "Experiment01_utils.hpp"

#include <fftw3.h>

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

/**
 * @brief Etapa 1: Pré-ênfase.
 *
 * Aplica um filtro de pré-ênfase para amplificar os componentes de alta frequência do sinal.
 * A fórmula é: y[n] = x[n] - α * x[n-1].
 * Isso compensa o decaimento natural do espectro da voz humana.
 *
 * @param signal Sinal de áudio (modificado in-place).
 * @param coefficient Coeficiente de pré-ênfase.
 */
static inline void pre_emphasis_inplace(vector<double>& signal, double coefficient)
{
    for (size_t i = signal.size() - 1; i >= 1; --i)
    {
        signal[i] = signal[i] - coefficient * signal[i - 1];
    }
    // signal[0] permanece
}

/**
 * @brief Etapa 2: Janelamento.
 *
 * Divide o sinal em frames curtos e aplica uma janela (Hamming) para reduzir o vazamento espectral.
 * O sinal é considerado estacionário dentro de cada frame.
 *
 * @param signal Sinal de entrada.
 * @param sampling_rate Frequência de amostragem.
 * @param frame_duration_ms Duração do frame em milissegundos.
 * @param frame_shift_ms Passo do frame em milissegundos.
 * @param frame_length Comprimento do frame em amostras (saída).
 * @param frame_step Passo do frame em amostras (saída).
 * @return Vector de frames janelados.
 */
auto framing_and_window(const vector<double>& signal, int sampling_rate, double frame_duration_ms,
                        double frame_shift_ms, int& frame_length, int& frame_step)
    -> vector<vector<double>>
{
    frame_length = (int) round(frame_duration_ms * sampling_rate / 1000.0);
    frame_step = (int) round(frame_shift_ms * sampling_rate / 1000.0);
    int signal_length = (int) signal.size();
    int number_of_frames = 1 + std::max(0, (signal_length - frame_length) / frame_step);
    int padded_length = (number_of_frames * frame_step) + frame_length;
    vector<double> padded_signal = signal;
    padded_signal.resize(padded_length, 0.0);
    vector<double> window_function(frame_length);

    // Janela de Hamming: w[n] = 0.54 - 0.46 * cos(2πn / (N-1))
    for (int i = 0; i < frame_length; ++i)
    {
        window_function[i] = 0.54 - 0.46 * cos(2 * M_PI * i / (frame_length - 1)); // Hamming
    }

    vector<vector<double>> frames(number_of_frames, vector<double>(frame_length));
    for (int i = 0; i < number_of_frames; ++i)
    {
        int start_index = i * frame_step;
        for (int j = 0; j < frame_length; ++j)
        {
            frames[i][j] = padded_signal[start_index + j] * window_function[j];
        }
    }
    return frames;
}

/**
 * @brief Etapas 3 & 4: STFT e Espectro de Potência.
 *
 * Calcula a Transformada Rápida de Fourier de Tempo Curto (STFT) para cada frame
 * e, em seguida, o espectro de potência.
 * A STFT é calculada usando a biblioteca FFTW.
 * O espectro de potência é |X[k]|^2, onde X[k] é a STFT.
 *
 * @param frames Frames janelados do sinal.
 * @param fft_points Número de pontos da FFT.
 * @return Espectro de potência para cada frame.
 */
auto rfft_power(const vector<vector<double>>& frames, int fft_points) -> vector<vector<double>>
{
    if (frames.empty())
    {
        throw std::invalid_argument("Input frames cannot be empty.");
    }

    size_t number_of_frames = frames.size();
    size_t number_of_bins = (fft_points / 2) + 1;

    vector<vector<double>> power_spectrum(number_of_frames, vector<double>(number_of_bins, 0.0));

    auto* fftw_input = (double*) fftw_malloc(sizeof(double) * fft_points);
    auto* fftw_output = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (fft_points / 2 + 1));
    fftw_plan fftw_plan = fftw_plan_dft_r2c_1d(fft_points, fftw_input, fftw_output, FFTW_ESTIMATE);

    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        size_t frame_length = frames[frame_index].size();
        auto fft_points_size_t = static_cast<size_t>(fft_points);
        size_t copy_length = std::min(frame_length, fft_points_size_t);

        for (size_t i = 0; i < copy_length; ++i)
        {
            fftw_input[i] = frames[frame_index][i];
        }
        for (size_t i = copy_length; i < fft_points_size_t; ++i)
        {
            fftw_input[i] = 0.0;
        }

        fftw_execute(fftw_plan);

        for (size_t bin_index = 0; bin_index < number_of_bins; ++bin_index)
        {
            double real_part = fftw_output[bin_index][0];
            double imaginary_part = fftw_output[bin_index][1];
            power_spectrum[frame_index][bin_index] =
                (real_part * real_part + imaginary_part * imaginary_part) / (double) fft_points;
        }
    }

    fftw_destroy_plan(fftw_plan);
    fftw_free(fftw_input);
    fftw_free(fftw_output);

    return power_spectrum;
}

/**
 * @brief Constrói o banco de filtros triangulares lineares.
 *
 * Diferente do MFCC, que usa a escala Mel, o LFCC usa filtros espaçados linearmente
 * em frequência.
 *
 * @param fft_points Número de pontos da FFT.
 * @param sampling_rate Frequência de amostragem.
 * @param number_of_filters Número de filtros no banco.
 * @param filterbank Matriz do banco de filtros (saída).
 * @param center_frequencies Frequências centrais dos filtros (saída).
 */
void build_linear_filterbank(int fft_points, int sampling_rate, int number_of_filters,
                             vector<vector<double>>& filterbank,
                             vector<double>& center_frequencies)
{
    int number_of_bins = (fft_points / 2) + 1;
    filterbank.assign(number_of_filters, vector<double>(number_of_bins, 0.0));
    double max_frequency = sampling_rate / 2.0;
    center_frequencies.resize(number_of_filters + 2);

    for (int i = 0; i < number_of_filters + 2; ++i)
    {
        center_frequencies[i] = (max_frequency) * (double) i / (number_of_filters + 1);
    }

    vector<int> bin_indices(number_of_filters + 2);
    for (int i = 0; i < (int) center_frequencies.size(); ++i)
    {
        bin_indices[i] = (int) floor((fft_points + 1) * center_frequencies[i] / sampling_rate);
    }

    for (int filter_index = 1; filter_index <= number_of_filters; ++filter_index)
    {
        int previous_bin_index = bin_indices[filter_index - 1];
        int current_bin_index = bin_indices[filter_index];
        int next_bin_index = bin_indices[filter_index + 1];

        if (current_bin_index > previous_bin_index)
        {
            for (int bin_index = previous_bin_index; bin_index < current_bin_index; ++bin_index)
            {
                filterbank[filter_index - 1][bin_index] =
                    (double) (bin_index - previous_bin_index) /
                    (double) (current_bin_index - previous_bin_index);
            }
        }
        if (next_bin_index > current_bin_index)
        {
            for (int bin_index = current_bin_index; bin_index < next_bin_index; ++bin_index)
            {
                filterbank[filter_index - 1][bin_index] =
                    (double) (next_bin_index - bin_index) /
                    (double) (next_bin_index - current_bin_index);
            }
        }
    }
}

/**
 * @brief Aplica o banco de filtros e a compressão logarítmica.
 *
 * Multiplica o espectro de potência de cada frame pelo banco de filtros para obter
 * a energia em cada banda. Em seguida, aplica o logaritmo natural.
 * Fórmula: E = log(Σ |X[k]|^2 * H_m[k])
 *
 * @param power_spectrum Espectro de potência.
 * @param filterbank Banco de filtros lineares.
 * @return Energias logarítmicas das bandas.
 */
auto dot_power_filterbank(const vector<vector<double>>& power_spectrum,
                          const vector<vector<double>>& filterbank) -> vector<vector<double>>
{
    size_t number_of_frames = power_spectrum.size();
    size_t number_of_filters = filterbank.size();
    vector<vector<double>> log_energies(number_of_frames, vector<double>(number_of_filters, 0.0));
    size_t number_of_bins = power_spectrum[0].size();

    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        for (size_t filter_index = 0; filter_index < number_of_filters; ++filter_index)
        {
            double sum = 0.0;
            for (size_t bin_index = 0; bin_index < number_of_bins; ++bin_index)
            {
                sum += power_spectrum[frame_index][bin_index] * filterbank[filter_index][bin_index];
            }
            sum = std::max(sum, 1e-12); // Evita log(0)
            log_energies[frame_index][filter_index] = log(sum);
        }
    }
    return log_energies;
}

/**
 * @brief Calcula a Transformada Cosseno Discreta (DCT-II).
 *
 * A DCT é aplicada para decorrelacionar as energias das bandas de filtro,
 * resultando nos coeficientes cepstrais (LFCC).
 *
 * @param log_energies Energias logarítmicas das bandas.
 * @param number_of_cepstra Número de coeficientes cepstrais a serem mantidos.
 * @return Coeficientes LFCC.
 */
auto dct2(const vector<vector<double>>& log_energies, int number_of_cepstra)
    -> vector<vector<double>>
{
    size_t number_of_frames = log_energies.size();
    size_t number_of_filters = log_energies[0].size();
    vector<vector<double>> cepstral_coefficients(number_of_frames,
                                                 vector<double>(number_of_cepstra, 0.0));

    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        for (int cepstrum_index = 0; cepstrum_index < number_of_cepstra; ++cepstrum_index)
        {
            double sum = 0.0;
            for (size_t filter_index = 0; filter_index < number_of_filters; ++filter_index)
            {
                sum += log_energies[frame_index][filter_index] *
                       cos(M_PI * cepstrum_index * (filter_index + 0.5) / number_of_filters);
            }
            // normalização ortho
            cepstral_coefficients[frame_index][cepstrum_index] =
                sum * sqrt(2.0 / static_cast<double>(number_of_filters));
            if (cepstrum_index == 0)
            {
                cepstral_coefficients[frame_index][cepstrum_index] *= 1.0 / std::numbers::sqrt2;
            }
        }
    }
    return cepstral_coefficients;
}

/**
 * @brief Calcula os deltas (derivada temporal) dos features.
 *
 * @param features Matriz de features (e.g., LFCCs).
 * @param window_span Extensão da janela para o cálculo do delta.
 * @return Matriz com os coeficientes delta.
 */
auto compute_deltas(const vector<vector<double>>& features, int window_span)
    -> vector<vector<double>>
{
    size_t number_of_frames = features.size();
    if (number_of_frames == 0)
    {
        return {};
    }
    size_t number_of_features = features[0].size();
    vector<vector<double>> padded_features(
        number_of_frames + (static_cast<size_t>(2 * window_span)),
        vector<double>(number_of_features));

    // pad edges
    for (int i = 0; i < window_span; ++i)
    {
        padded_features[i] = features[0];
    }
    for (size_t i = 0; i < number_of_frames; ++i)
    {
        padded_features[i + window_span] = features[i];
    }
    for (int i = 0; i < window_span; ++i)
    {
        padded_features[number_of_frames + window_span + i] = features[number_of_frames - 1];
    }

    double denominator = 0.0;
    for (int i = 1; i <= window_span; ++i)
    {
        denominator += i * i;
    }
    denominator *= 2.0;

    vector<vector<double>> delta_features(number_of_frames,
                                          vector<double>(number_of_features, 0.0));
    for (size_t t = 0; t < number_of_frames; ++t)
    {
        for (size_t d = 0; d < number_of_features; ++d)
        {
            double numerator = 0.0;
            for (int n = 1; n <= window_span; ++n)
            {
                numerator +=
                    n * (padded_features[t + window_span + n][d] - padded_features[t + window_span - n][d]);
            }
            delta_features[t][d] = numerator / denominator;
        }
    }
    return delta_features;
}

/**
 * @brief Função principal que carrega e processa o áudio para extrair LFCCs.
 *
 * Orquestra todas as etapas de pré-processamento:
 * 1. Pré-ênfase
 * 2. Janelamento
 * 3. STFT (FFT)
 * 4. Espectro de Potência
 * 5. Aplicação do banco de filtros lineares
 * 6. Compressão logarítmica
 * 7. DCT para obter os coeficientes cepstrais
 * 8. Cálculo dos deltas e delta-deltas
 */
static auto loadAndProcessAudio(const std::string& audioFilePath, int target_sampling_rate,
                                double preemphasis_coefficient, double frame_duration_ms,
                                double frame_shift_ms, int number_of_filters,
                                int number_of_cepstrals, int delta_window_span)
    -> std::vector<Eigen::MatrixXf>
{
    auto [audioSamples, audioStimulus, eegIndex] = loadAudioFromMat(audioFilePath, 0);
    Eigen::MatrixXf audioMatrix = audioSamples.transpose(); // Convert to 1xN matrix for windowing

    if (audio_sampling_rate != target_sampling_rate)
    {
        std::cerr << "Amostragem diferente de " << target_sampling_rate
                  << " Hz. Reamostrar externamente.\n";
        return {};
    }

    vector<double> input_data(audioMatrix.size());
    Eigen::Map<Eigen::VectorXd>(input_data.data(), audioMatrix.size()) = audioMatrix.cast<double>();
    audioMatrix.resize(0, 0);

    // Etapa 1: Pré-ênfase
    pre_emphasis_inplace(input_data, preemphasis_coefficient);

    // Etapa 2: Framing e Janelamento (Hamming)
    int frame_length;
    int frame_step;
    auto frames = framing_and_window(input_data, target_sampling_rate, frame_duration_ms,
                                     frame_shift_ms, frame_length, frame_step);

    // Etapas 3 & 4: STFT (via FFT) e cálculo do Espectro de Potência
    auto power_spectrum = rfft_power(frames, frame_length);

    // Etapa 5: Construção do banco de filtros lineares
    vector<vector<double>> filterbank;
    vector<double> center_frequencies;
    build_linear_filterbank(frame_length, target_sampling_rate, number_of_filters, filterbank,
                            center_frequencies);

    // Etapa 6: Aplicação do banco de filtros e compressão logarítmica
    auto log_energies = dot_power_filterbank(power_spectrum, filterbank);

    // Etapa 7: DCT para obter os coeficientes cepstrais (LFCC)
    auto cepstral_coefficients = dct2(log_energies, number_of_cepstrals);

    // (Extra) Etapa 8: Cálculo dos deltas (derivadas temporais)
    auto delta_coefficients = compute_deltas(cepstral_coefficients, delta_window_span);
    auto delta_delta_coefficients = compute_deltas(delta_coefficients, delta_window_span);

    // Etapa 9: Concatenação e normalização (aqui apenas imprime para depuração)
    size_t number_of_frames = cepstral_coefficients.size();
    for (size_t t = 0; t < min(number_of_frames, static_cast<size_t>(5)); ++t)
    {
        for (int i = 0; i < number_of_cepstrals; ++i)
        {
            cout << cepstral_coefficients[t][i] << " ";
        }
        for (int i = 0; i < number_of_cepstrals; ++i)
        {
            cout << delta_coefficients[t][i] << " ";
        }
        for (int i = 0; i < number_of_cepstrals; ++i)
        {
            cout << delta_delta_coefficients[t][i] << " ";
        }
        cout << "\n";
    }

    return {}; // Retornar vetor vazio por enquanto
}

void processSubject(const std::string& subjectPath, const std::string& subjectName,
                    const std::string& audioFilePath, const std::string& eegFilePath)
{
    cout << "Processing subject: " << subjectName << '\n';

    // Parâmetros de extração de features de áudio
    const int target_sampling_rate = 44100;
    const double preemphasis_coefficient = 0.97;
    const double frame_duration_ms = 25.0;
    const double frame_shift_ms = 10.0;
    const int number_of_filters = 24;
    const int number_of_cepstrals = 19;
    const int delta_window_span = 2;

    // Load, normalize, and window audio data
    std::vector<Eigen::MatrixXf> audioWindows =
        loadAndProcessAudio(audioFilePath, target_sampling_rate, preemphasis_coefficient,
                            frame_duration_ms, frame_shift_ms, number_of_filters,
                            number_of_cepstrals, delta_window_span);
    cout << "  - Loaded and processed " << audioWindows.size() << " audio windows.\n";
}

