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
        signal[i] = signal[i] - (coefficient * signal[i - 1]);
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
 * @param params Parâmetros de processamento de áudio.
 * @param frame_length Comprimento do frame em amostras (saída).
 * @param frame_step Passo do frame em amostras (saída).
 * @return Vector de frames janelados.
 */
auto framing_and_window(const vector<double>& signal, const AudioProcessingParameters& params,
                        int& frame_length, int& frame_step) -> vector<vector<double>>
{
    /// @brief Comprimento de cada frame em amostras. (Parâmetro de saída)
    frame_length = (int) round(params.frame_duration_ms * params.target_sampling_rate / 1000.0);

    /// @brief Deslocamento entre frames consecutivos em amostras. (Parâmetro de saída)
    frame_step = (int) round(params.frame_shift_ms * params.target_sampling_rate / 1000.0);

    /// @brief Comprimento total do sinal de entrada.
    const int signal_length = (int) signal.size();

    /// @brief Número total de frames que serão extraídos do sinal.
    const int number_of_frames = 1 + std::max(0, (signal_length - frame_length) / frame_step);

    /// @brief Comprimento do sinal após o padding para conter todos os frames.
    const int padded_length = (number_of_frames * frame_step) + frame_length;

    /// @brief Cópia do sinal com padding de zeros no final.
    vector<double> padded_signal = signal;

    /// @brief Vetor para armazenar a função de janelamento (Hamming).
    vector<double> window_function(frame_length);

    /// @brief Matriz para armazenar os frames resultantes após o janelamento.
    vector<vector<double>> frames(number_of_frames, vector<double>(frame_length));

    padded_signal.resize(padded_length, 0.0);

    // Janela de Hamming: w[n] = 0.54 - 0.46 * cos(2πn / (N-1))
    for (int i = 0; i < frame_length; ++i)
    {
        /// @brief Índice atual para a função de janela.
        window_function[i] = 0.54 - (0.46 * cos(2 * M_PI * i / (frame_length - 1))); // Hamming
    }

    for (int i = 0; i < number_of_frames; ++i)
    {
        /// @brief Índice do frame atual.
        const int start_index = i * frame_step;
        for (int j = 0; j < frame_length; ++j)
        {
            /// @brief Índice da amostra dentro do frame atual.
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

    /// @brief Número de frames de entrada.
    const size_t number_of_frames = frames.size();

    /// @brief Número de bins de frequência resultantes da FFT (N/2 + 1).
    const size_t number_of_bins = (fft_points / 2) + 1;

    /// @brief Matriz para armazenar o espectro de potência de cada frame.
    vector<vector<double>> power_spectrum(number_of_frames, vector<double>(number_of_bins, 0.0));

    /// @brief Buffer de entrada para a FFTW (real).
    auto* fftw_input = (double*) fftw_malloc(sizeof(double) * fft_points);

    /// @brief Buffer de saída para a FFTW (complexo).
    auto* fftw_output = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (fft_points / 2 + 1));

    /// @brief Plano da FFTW para a transformada de real para complexo.
    fftw_plan fftw_plan = fftw_plan_dft_r2c_1d(fft_points, fftw_input, fftw_output, FFTW_ESTIMATE);

    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        /// @brief Comprimento do frame atual.
        const size_t frame_length = frames[frame_index].size();

        /// @brief Tamanho da FFT em `size_t` para comparações.
        const auto fft_points_size_t = static_cast<size_t>(fft_points);

        /// @brief Comprimento a ser copiado para o buffer da FFT, o mínimo entre o comprimento do
        /// frame e o tamanho da FFT.
        const size_t copy_length = std::min(frame_length, fft_points_size_t);

        for (size_t i = 0; i < copy_length; ++i)
        {
            /// @brief Índice da amostra dentro do frame para cópia.
            fftw_input[i] = frames[frame_index][i];
        }
        for (size_t i = copy_length; i < fft_points_size_t; ++i)
        {
            /// @brief Índice para preenchimento com zeros no buffer da FFT.
            fftw_input[i] = 0.0;
        }

        fftw_execute(fftw_plan);

        for (size_t bin_index = 0; bin_index < number_of_bins; ++bin_index)
        {
            /// @brief Parte real do resultado da FFT para o bin atual.
            const double real_part = fftw_output[bin_index][0];

            /// @brief Parte imaginária do resultado da FFT para o bin atual.
            const double imaginary_part = fftw_output[bin_index][1];

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
 * @param params Parâmetros de processamento de áudio.
 * @param filterbank Matriz do banco de filtros (saída).
 * @param center_frequencies Frequências centrais dos filtros (saída).
 */
void build_linear_filterbank(int fft_points, const AudioProcessingParameters& params,
                             vector<vector<double>>& filterbank, vector<double>& center_frequencies)
{
    /// @brief Número de bins de frequência resultantes da FFT (N/2 + 1).
    const int number_of_bins = (fft_points / 2) + 1;

    /// @brief Frequência máxima representada (Metade da frequência de amostragem).
    const double max_frequency = params.target_sampling_rate / 2.0;

    /// @brief Vetor de índices dos bins correspondentes às frequências centrais.
    vector<int> bin_indices(params.number_of_filters + 2);

    filterbank.assign(params.number_of_filters, vector<double>(number_of_bins, 0.0));
    center_frequencies.resize(params.number_of_filters + 2);

    for (int i = 0; i < params.number_of_filters + 2; ++i)
    {
        /// @brief Índice para iterar sobre as frequências centrais.
        center_frequencies[i] = (max_frequency) * (double) i / (params.number_of_filters + 1);
    }

    for (size_t i = 0; i < center_frequencies.size(); ++i)
    {
        /// @brief Índice para mapear frequências centrais para bins da FFT.
        bin_indices[i] =
            (int) floor((fft_points + 1) * center_frequencies[i] / params.target_sampling_rate);
    }

    for (int filter_index = 1; filter_index <= params.number_of_filters; ++filter_index)
    {
        /// @brief Índice do bin anterior ao filtro atual.
        const int previous_bin_index = bin_indices[filter_index - 1];

        /// @brief Índice do bin central do filtro atual.
        const int current_bin_index = bin_indices[filter_index];

        /// @brief Índice do bin posterior ao filtro atual.
        const int next_bin_index = bin_indices[filter_index + 1];

        if (current_bin_index > previous_bin_index)
        {
            for (int bin_index = previous_bin_index; bin_index < current_bin_index; ++bin_index)
            {
                /// @brief Índice do bin de frequência para a parte ascendente do filtro triangular.
                filterbank[filter_index - 1][bin_index] =
                    (double) (bin_index - previous_bin_index) /
                    (double) (current_bin_index - previous_bin_index);
            }
        }
        if (next_bin_index > current_bin_index)
        {
            for (int bin_index = current_bin_index; bin_index < next_bin_index; ++bin_index)
            {
                /// @brief Índice do bin de frequência para a parte descendente do filtro
                /// triangular.
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
    /// @brief Número de frames no espectro de potência.
    const size_t number_of_frames = power_spectrum.size();
    /// @brief Número de filtros no banco de filtros.
    const size_t number_of_filters = filterbank.size();
    /// @brief Número de bins de frequência por frame.
    const size_t number_of_bins = power_spectrum[0].size();
    /// @brief Matriz para armazenar as energias logarítmicas resultantes.
    vector<vector<double>> log_energies(number_of_frames, vector<double>(number_of_filters, 0.0));

    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        /// @brief Índice do frame atual.
        for (size_t filter_index = 0; filter_index < number_of_filters; ++filter_index)
        {
            /// @brief Índice do filtro atual.
            /// @brief Variável temporária para acumular a soma ponderada do espectro de potência.
            double sum = 0.0;
            for (size_t bin_index = 0; bin_index < number_of_bins; ++bin_index)
            {
                /// @brief Índice do bin de frequência atual.
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
    /// @brief Número de frames (vetores de energia) de entrada.
    const size_t number_of_frames = log_energies.size();

    /// @brief Número de filtros, que corresponde ao número de energias por frame.
    const size_t number_of_filters = log_energies[0].size();

    /// @brief Matriz para armazenar os coeficientes cepstrais resultantes.
    vector<vector<double>> cepstral_coefficients(number_of_frames,
                                                 vector<double>(number_of_cepstra, 0.0));

    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        /// @brief Índice do frame atual.
        for (int cepstrum_index = 0; cepstrum_index < number_of_cepstra; ++cepstrum_index)
        {
            /// @brief Índice do coeficiente cepstral atual.
            /// @brief Variável temporária para acumular a soma ponderada para o cálculo do
            /// coeficiente cepstral.
            double sum = 0.0;
            for (size_t filter_index = 0; filter_index < number_of_filters; ++filter_index)
            {
                /// @brief Índice do filtro atual.
                sum += log_energies[frame_index][filter_index] *
                       cos(M_PI * cepstrum_index * (static_cast<double>(filter_index) + 0.5) /
                           static_cast<double>(number_of_filters));
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
    /// @brief Número de frames (vetores de features) de entrada.
    const size_t number_of_frames = features.size();
    if (number_of_frames == 0)
    {
        return {};
    }

    /// @brief Dimensionalidade do vetor de features.
    const size_t number_of_features = features[0].size();

    /// @brief Matriz de features com padding nas bordas para o cálculo dos deltas.
    vector<vector<double>> padded_features(
        number_of_frames + (static_cast<size_t>(2 * window_span)),
        vector<double>(number_of_features));

    /// @brief Denominador da fórmula de cálculo dos deltas, pré-calculado.
    double denominator = 0.0;

    /// @brief Matriz para armazenar os coeficientes delta resultantes.
    vector<vector<double>> delta_features(number_of_frames,
                                          vector<double>(number_of_features, 0.0));

    // pad edges
    for (int i = 0; i < window_span; ++i)
    {
        /// @brief Índice para preencher o padding inicial.
        padded_features[i] = features[0];
    }
    for (size_t i = 0; i < number_of_frames; ++i)
    {
        /// @brief Índice para copiar as features originais para o centro do vetor com padding.
        padded_features[i + window_span] = features[i];
    }
    for (int i = 0; i < window_span; ++i)
    {
        /// @brief Índice para preencher o padding final.
        padded_features[number_of_frames + window_span + i] = features[number_of_frames - 1];
    }

    for (int i = 1; i <= window_span; ++i)
    {
        /// @brief Índice para calcular o denominador da fórmula de deltas.
        denominator += i * i;
    }
    denominator *= 2.0;

    for (size_t t = 0; t < number_of_frames; ++t)
    {
        /// @brief Índice do frame atual.
        for (size_t d = 0; d < number_of_features; ++d)
        {
            /// @brief Índice da feature atual.
            /// @brief Variável temporária para acumular o numerador da fórmula de deltas.
            double numerator = 0.0;
            for (int n = 1; n <= window_span; ++n)
            {
                /// @brief Deslocamento para calcular a diferença entre frames.
                numerator += n * (padded_features[t + window_span + n][d] -
                                  padded_features[t + window_span - n][d]);
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
static auto loadAndProcessAudio(const std::string& audioFilePath,
                                const AudioProcessingParameters& params)
    -> std::vector<Eigen::MatrixXf>
{
    /// @brief Amostras de áudio carregadas do arquivo .mat.
    auto [audioSamples, audioStimulus, eegIndex] = loadAudioFromMat(audioFilePath, 0);

    /// @brief Matriz Eigen para manipulação inicial dos dados de áudio.
    Eigen::MatrixXf audioMatrix = audioSamples.transpose(); // Convert to 1xN matrix for windowing

    /// @brief Vetor de double para conter os dados de áudio para processamento.
    vector<double> input_data(audioMatrix.size());

    /// @brief Comprimento do frame em amostras, calculado pelo janelamento.
    int frame_length;

    /// @brief Passo do frame em amostras, calculado pelo janelamento.
    int frame_step;

    /// @brief Frames do sinal após janelamento.
    auto frames = framing_and_window(input_data, params, frame_length, frame_step);

    /// @brief Espectro de potência de cada frame.
    auto power_spectrum = rfft_power(frames, frame_length);

    /// @brief Matriz do banco de filtros lineares.
    vector<vector<double>> filterbank;

    /// @brief Frequências centrais de cada filtro do banco.
    vector<double> center_frequencies;

    /// @brief Energias logarítmicas após aplicação do banco de filtros.
    auto log_energies = dot_power_filterbank(power_spectrum, filterbank);

    /// @brief Coeficientes cepstrais (LFCC) calculados.
    auto cepstral_coefficients = dct2(log_energies, params.number_of_cepstrals);

    /// @brief Coeficientes delta (primeira derivada).
    auto delta_coefficients = compute_deltas(cepstral_coefficients, params.delta_window_span);

    /// @brief Coeficientes delta-delta (segunda derivada).
    auto delta_delta_coefficients = compute_deltas(delta_coefficients, params.delta_window_span);

    /// @brief Número total de frames processados.
    size_t number_of_frames = cepstral_coefficients.size();

    if (44100 != params.target_sampling_rate)
    {
        std::cerr << "Amostragem diferente de " << params.target_sampling_rate
                  << " Hz. Reamostrar externamente.\n";
        return {};
    }

    Eigen::Map<Eigen::VectorXd>(input_data.data(), audioMatrix.size()) = audioMatrix.cast<double>();
    audioMatrix.resize(0, 0);

    // Etapa 1: Pré-ênfase
    pre_emphasis_inplace(input_data, params.preemphasis_coefficient);

    // Etapa 2: Framing e Janelamento (Hamming)
    frames = framing_and_window(input_data, params, frame_length, frame_step);

    // Etapas 3 & 4: STFT (via FFT) e cálculo do Espectro de Potência
    power_spectrum = rfft_power(frames, frame_length);

    // Etapa 5: Construção do banco de filtros lineares
    build_linear_filterbank(frame_length, params, filterbank, center_frequencies);

    // Etapa 6: Aplicação do banco de filtros e compressão logarítmica
    log_energies = dot_power_filterbank(power_spectrum, filterbank);

    // Etapa 7: DCT para obter os coeficientes cepstrais (LFCC)
    cepstral_coefficients = dct2(log_energies, params.number_of_cepstrals);

    // (Extra) Etapa 8: Cálculo dos deltas (derivadas temporais)
    delta_coefficients = compute_deltas(cepstral_coefficients, params.delta_window_span);
    delta_delta_coefficients = compute_deltas(delta_coefficients, params.delta_window_span);

    // Etapa 9: Concatenação e normalização (aqui apenas imprime para depuração)
    number_of_frames = cepstral_coefficients.size();
    for (size_t t = 0; t < min(number_of_frames, static_cast<size_t>(5)); ++t)
    {
        /// @brief Índice do frame atual para depuração.
        for (int i = 0; i < params.number_of_cepstrals; ++i)
        {
            /// @brief Índice do coeficiente cepstral para depuração.
            cout << cepstral_coefficients[t][i] << " ";
        }
        for (int i = 0; i < params.number_of_cepstrals; ++i)
        {
            /// @brief Índice do coeficiente delta para depuração.
            cout << delta_coefficients[t][i] << " ";
        }
        for (int i = 0; i < params.number_of_cepstrals; ++i)
        {
            /// @brief Índice do coeficiente delta-delta para depuração.
            cout << delta_delta_coefficients[t][i] << " ";
        }
        cout << "\n";
    }

    return {}; // Retornar vetor vazio por enquanto
}

void processSubject(const SubjectInfo& subject)
{
    cout << "Processing subject: " << subject.name << '\n';

    /// @brief Parâmetros para a extração de features LFCC do áudio.
    const AudioProcessingParameters params = {.target_sampling_rate = 44100,
                                              .preemphasis_coefficient = 0.97,
                                              .frame_duration_ms = 25.0,
                                              .frame_shift_ms = 10.0,
                                              .number_of_filters = 24,
                                              .number_of_cepstrals = 19,
                                              .delta_window_span = 2};

    /// @brief Janelas de áudio processadas (atualmente não utilizado).
    std::vector<Eigen::MatrixXf> audioWindows =
        loadAndProcessAudio(subject.audio_file_path, params);
    cout << "  - Loaded and processed " << audioWindows.size() << " audio windows.\n";
}
