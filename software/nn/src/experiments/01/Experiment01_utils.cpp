#include "Experiment01_utils.hpp"
#define _USE_MATH_DEFINES
#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers> // Added for std::numbers::sqrt2_v
#include <vector>

#include "core/dataLoaders/10.1117/AudioLoader.h"
#include "core/dataLoaders/10.1117/EEGLoader.h"
#include "core/optimizers/Adam.hpp"
#include "core/tensor/Tensor.hpp"

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
static inline void pre_emphasis_inplace(vector<float>& signal, float coefficient)
{
    // Aplica o filtro de pré-ênfase ao sinal.
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
auto framing_and_window(const vector<float>& signal, const AudioProcessingParameters& params,
                        int& frame_length, int& frame_step) -> vector<vector<float>>
{
    // Comprimento de cada frame em amostras. (Parâmetro de saída)
    frame_length = (int) round(params.frame_duration_ms * params.target_sampling_rate / 1000.0F);

    // Deslocamento entre frames consecutivos em amostras. (Parâmetro de saída)
    frame_step = (int) round(params.frame_shift_ms * params.target_sampling_rate / 1000.0F);

    // Comprimento total do sinal de entrada.
    const int signal_length = (int) signal.size();
    // Número total de frames que serão extraídos do sinal.
    const int number_of_frames = 1 + std::max(0, (signal_length - frame_length) / frame_step);
    // Comprimento do sinal após o padding para conter todos os frames.
    const int padded_length = (number_of_frames * frame_step) + frame_length;
    // Cópia do sinal com padding de zeros no final.
    vector<float> padded_signal = signal;
    // Vetor para armazenar a função de janelamento (Hamming).
    vector<float> window_function(frame_length);
    // Matriz para armazenar os frames resultantes após o janelamento.
    vector<vector<float>> frames(number_of_frames, vector<float>(frame_length));

    padded_signal.resize(padded_length, 0.0F);

    // Janela de Hamming: w[n] = 0.54 - 0.46 * cos(2πn / (N-1))
    for (float i = 0; i < frame_length; ++i)
    {
        window_function[i] =
            0.54F - (0.46F * cosf(2 * std::numbers::pi_v<float> * i /
                                  (static_cast<float>(frame_length) - 1))); // Hamming
    }

    // Aplicação do janelamento em cada frame.
    for (int i = 0; i < number_of_frames; ++i)
    {
        // Índice do frame atual.
        const int start_index = i * frame_step;

        // Preenchimento do frame atual com a janela aplicada.
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
auto rfft_power(const vector<vector<float>>& frames, int fft_points) -> Tensor
{
    if (frames.empty())
    {
        throw std::invalid_argument("Input frames cannot be empty.");
    }

    // Número de frames de entrada.
    const size_t number_of_frames = frames.size();
    // Número de bins de frequência resultantes da FFT (N/2 + 1).
    const size_t number_of_bins = (fft_points / 2) + 1;
    // Matriz para armazenar o espectro de potência de cada frame.
    Tensor power_spectrum(static_cast<int>(number_of_frames), static_cast<int>(number_of_bins));
    // Buffer de entrada para a FFTW (real).
    auto* fftw_input = (float*) fftwf_malloc(sizeof(float) * fft_points);
    // Buffer de saída para a FFTW (complexo).
    auto* fftw_output = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (fft_points / 2 + 1));
    // Plano da FFTW para a transformada de real para complexo.
    fftwf_plan fftw_plan =
        fftwf_plan_dft_r2c_1d(fft_points, fftw_input, fftw_output, FFTW_ESTIMATE);

    // Processamento de cada frame.
    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        // Comprimento do frame atual.
        const size_t frame_length = frames[frame_index].size();
        // Tamanho da FFT em `size_t` para comparações.
        const auto fft_points_size_t = static_cast<size_t>(fft_points);
        // Comprimento a ser copiado para o buffer da FFT, o mínimo entre o comprimento do
        /// frame e o tamanho da FFT.
        const size_t copy_length = std::min(frame_length, fft_points_size_t);

        // Cópia do frame atual para o buffer de entrada da FFT, com zero-padding se necessário.
        for (size_t i = 0; i < copy_length; ++i)
        {
            fftw_input[i] = frames[frame_index][i];
        }
        // Preenchimento com zeros se o frame for menor que o tamanho da FFT.
        for (size_t i = copy_length; i < fft_points_size_t; ++i)
        {
            fftw_input[i] = 0.0F;
        }

        // Execução da FFT.
        fftwf_execute(fftw_plan);

        // Cálculo do espectro de potência para o frame atual.
        for (size_t bin_index = 0; bin_index < number_of_bins; ++bin_index)
        {
            // Parte real do resultado da FFT para o bin atual.
            const float real_part = fftw_output[bin_index][0];
            // Parte imaginária do resultado da FFT para o bin atual.
            const float imaginary_part = fftw_output[bin_index][1];
            // Cálculo do espectro de potência normalizado.
            power_spectrum.data(frame_index, bin_index) =
                (real_part * real_part + imaginary_part * imaginary_part) / (float) fft_points;
        }
    }

    // Liberação dos recursos da FFTW.
    fftwf_destroy_plan(fftw_plan);
    fftwf_free(fftw_input);
    fftwf_free(fftw_output);

    // Retorno do espectro de potência calculado.
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
                             Tensor& filterbank, vector<float>& center_frequencies)
{
    // Número de bins de frequência resultantes da FFT (N/2 + 1).
    const int number_of_bins = (fft_points / 2) + 1;
    // Frequência máxima representada (Metade da frequência de amostragem).
    const float max_frequency = params.target_sampling_rate / 2.0F;
    // Vetor de índices dos bins correspondentes às frequências centrais.
    vector<int> bin_indices(params.number_of_filters + 2);

    // Inicialização do banco de filtros e das frequências centrais.
    filterbank = Tensor(params.number_of_filters, number_of_bins);

    // Inicialização das frequências centrais.
    center_frequencies.resize(params.number_of_filters + 2);

    // Cálculo das frequências centrais dos filtros.
    for (int i = 0; i < params.number_of_filters + 2; ++i)
    {
        center_frequencies[i] = (max_frequency) * (float) i / (params.number_of_filters + 1);
    }

    // Cálculo dos índices dos bins para as frequências centrais.
    for (size_t i = 0; i < center_frequencies.size(); ++i)
    {
        // Índice para mapear frequências centrais para bins da FFT.
        bin_indices[i] =
            (int) floor((fft_points + 1) * center_frequencies[i] / params.target_sampling_rate);
    }

    // Construção dos filtros triangulares.
    for (int filter_index = 1; filter_index <= params.number_of_filters; ++filter_index)
    {
        // Índice do bin anterior ao filtro atual.
        const int previous_bin_index = bin_indices[filter_index - 1];
        // Índice do bin central do filtro atual.
        const int current_bin_index = bin_indices[filter_index];
        // Índice do bin posterior ao filtro atual.
        const int next_bin_index = bin_indices[filter_index + 1];

        // Construção da parte ascendente e descendente do filtro triangular.
        if (current_bin_index > previous_bin_index)
        {
            // Parte ascendente do filtro triangular.
            for (int bin_index = previous_bin_index; bin_index < current_bin_index; ++bin_index)
            {
                filterbank.data(filter_index - 1, bin_index) =
                    (float) (bin_index - previous_bin_index) /
                    (float) (current_bin_index - previous_bin_index);
            }
        }
        // Parte descendente do filtro triangular.
        if (next_bin_index > current_bin_index)
        {
            // Parte descendente do filtro triangular.
            for (int bin_index = current_bin_index; bin_index < next_bin_index; ++bin_index)
            {
                filterbank.data(filter_index - 1, bin_index) =
                    (float) (next_bin_index - bin_index) /
                    (float) (next_bin_index - current_bin_index);
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
auto dot_power_filterbank(const Tensor& power_spectrum, const Tensor& filterbank) -> Tensor
{
    // Número de frames no espectro de potência.
    const size_t number_of_frames = power_spectrum.data.rows();
    // Número de filtros no banco de filtros.
    const size_t number_of_filters = filterbank.data.rows();
    // Número de bins de frequência por frame.
    const size_t number_of_bins = power_spectrum.data.cols();
    // Matriz para armazenar as energias logarítmicas resultantes.
    Tensor log_energies(number_of_frames, number_of_filters);

    // Cálculo das energias logarítmicas para cada frame e filtro.
    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        // Índice do frame atual.
        for (size_t filter_index = 0; filter_index < number_of_filters; ++filter_index)
        {
            // Índice do filtro atual.
            // Variável temporária para acumular a soma ponderada do espectro de potência.
            float sum = 0.0F;
            for (size_t bin_index = 0; bin_index < number_of_bins; ++bin_index)
            {
                // Índice do bin de frequência atual.
                sum += power_spectrum.data(frame_index, bin_index) *
                       filterbank.data(filter_index, bin_index);
            }
            sum = std::max(sum, 1e-12F); // Evita log(0)
            log_energies.data(frame_index, filter_index) = log(sum);
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
auto dct2(const Tensor& log_energies, int number_of_cepstra) -> Tensor
{
    // Número de frames (vetores de energia) de entrada.
    const size_t number_of_frames = log_energies.data.rows();
    // Número de filtros, que corresponde ao número de energias por frame.
    const size_t number_of_filters = log_energies.data.cols();
    // Matriz para armazenar os coeficientes cepstrais resultantes.
    Tensor cepstral_coefficients(number_of_frames, number_of_cepstra);
    // Cálculo dos coeficientes cepstrais para cada frame.
    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        // Cria os coeficientes cepstrais para o frame atual.
        for (int cepstrum_index = 0; cepstrum_index < number_of_cepstra; ++cepstrum_index)
        {
            // Variável temporária para acumular a soma ponderada para o cálculo do
            float sum = 0.0F;

            // Cálculo do coeficiente cepstral atual.
            for (size_t filter_index = 0; filter_index < number_of_filters; ++filter_index)
            {
                // Índice do filtro atual.
                sum += log_energies.data(frame_index, filter_index) *
                       cosf(std::numbers::pi_v<float> * cepstrum_index *
                            (static_cast<float>(filter_index) + 0.5F) /
                            static_cast<float>(number_of_filters));
            }

            // normalização ortho
            cepstral_coefficients.data(frame_index, cepstrum_index) =
                sum * sqrt(2.0F / static_cast<float>(number_of_filters));

            // Ajuste do primeiro coeficiente cepstral.
            if (cepstrum_index == 0)
            {
                cepstral_coefficients.data(frame_index, cepstrum_index) *=
                    1.0F / std::numbers::sqrt2_v<float>;
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
auto compute_deltas(const Tensor& features, int window_span) -> Tensor
{
    // Número de frames (vetores de features) de entrada.
    const size_t number_of_frames = features.data.rows();
    if (number_of_frames == 0)
    {
        return {};
    }
    // Dimensionalidade do vetor de features.
    const size_t number_of_features = features.data.cols();
    // Matriz de features com padding nas bordas para o cálculo dos deltas.
    Tensor padded_features(number_of_frames + (static_cast<size_t>(2 * window_span)),
                           number_of_features);
    // Denominador da fórmula de cálculo dos deltas, pré-calculado.
    float denominator = 0.0F;
    // Matriz para armazenar os coeficientes delta resultantes.
    Tensor delta_features(number_of_frames, number_of_features);

    // Preenchimento do tensor de features com padding.
    for (int i = 0; i < window_span; ++i)
    {
        padded_features.data.row(i) = features.data.row(0);
    }

    // Cópia dos features originais para o centro do tensor com padding.
    for (size_t i = 0; i < number_of_frames; ++i)
    {
        padded_features.data.row(i + window_span) = features.data.row(i);
    }

    // Preenchimento do padding final.
    for (int i = 0; i < window_span; ++i)
    {
        padded_features.data.row(number_of_frames + window_span + i) =
            features.data.row(number_of_frames - 1);
    }

    // Cálculo do denominador da fórmula de deltas.
    for (int i = 1; i <= window_span; ++i)
    {
        denominator += (float) i * i;
    }
    denominator *= 2.0F;

    // Cálculo dos coeficientes delta para cada frame e feature.
    for (size_t t = 0; t < number_of_frames; ++t)
    {
        // Itera sobre cada dimensão do feature vector.
        for (size_t d = 0; d < number_of_features; ++d)
        {
            // Variável temporária para acumular o numerador da fórmula de deltas.
            float numerator = 0.0F;

            // Cálculo do numerador para o delta da feature atual.
            for (int n = 1; n <= window_span; ++n)
            {
                // Deslocamento para calcular a diferença entre frames.
                numerator += (float) n * (padded_features.data(t + window_span + n, d) -
                                          padded_features.data(t + window_span - n, d));
            }
            delta_features.data(t, d) = numerator / denominator;
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
                                const AudioProcessingParameters& params) -> std::vector<Tensor>
{
    // Amostras de áudio carregadas do arquivo .mat.
    auto [audioSamples, audioStimulus, eegIndex] = loadAudioFromMat(audioFilePath, 0);
    // Matriz Eigen para manipulação inicial dos dados de áudio.
    Eigen::MatrixXf audioMatrix = audioSamples.transpose(); // Convert to 1xN matrix for windowing
    // Vetor de float para conter os dados de áudio para processamento.
    vector<float> input_data(audioMatrix.size());
    // Comprimento do frame em amostras, calculado pelo janelamento.
    int frame_length;
    // Passo do frame em amostras, calculado pelo janelamento.
    int frame_step;
    // Frames do sinal após janelamento.
    vector<vector<float>> frames;
    // Espectro de potência de cada frame.
    Tensor power_spectrum;
    // Matriz do banco de filtros lineares.
    Tensor filterbank;
    // Frequências centrais de cada filtro do banco.
    vector<float> center_frequencies;
    // Energias logarítmicas após aplicação do banco de filtros.
    Tensor log_energies;
    // Coeficientes cepstrais (LFCC) calculados.
    Tensor cepstral_coefficients;
    // Coeficientes delta (primeira derivada).
    Tensor delta_coefficients;
    // Coeficientes delta-delta (segunda derivada).
    Tensor delta_delta_coefficients;
    // Número total de frames processados.
    size_t number_of_frames;

    if (44100 != params.target_sampling_rate)
    {
        std::cerr << "Amostragem diferente de " << params.target_sampling_rate
                  << " Hz. Reamostrar externamente.\n";
        return {};
    }

    Eigen::Map<Eigen::VectorXf>(input_data.data(), audioMatrix.size()) = audioMatrix;
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
    number_of_frames = cepstral_coefficients.data.rows();
    for (size_t t = 0; t < min(number_of_frames, static_cast<size_t>(5)); ++t)
    {
        // Índice do frame atual para depuração.
        for (int i = 0; i < params.number_of_cepstrals; ++i)
        {
            // Índice do coeficiente cepstral para depuração.
            cout << cepstral_coefficients.data(t, i) << " ";
        }
        for (int i = 0; i < params.number_of_cepstrals; ++i)
        {
            // Índice do coeficiente delta para depuração.
            cout << delta_coefficients.data(t, i) << " ";
        }
        for (int i = 0; i < params.number_of_cepstrals; ++i)
        {
            // Índice do coeficiente delta-delta para depuração.
            cout << delta_delta_coefficients.data(t, i) << " ";
        }
        cout << "\n";
    }

    return {}; // Retornar vetor vazio por enquanto
}

void processSubject(const SubjectInfo& subject)
{
    cout << "Processing subject: " << subject.name << '\n';

    // Parâmetros para a extração de features LFCC do áudio.
    const AudioProcessingParameters params = {.target_sampling_rate = 44100,
                                              .preemphasis_coefficient = 0.97,
                                              .frame_duration_ms = 25.0,
                                              .frame_shift_ms = 10.0,
                                              .number_of_filters = 24,
                                              .number_of_cepstrals = 19,
                                              .delta_window_span = 2};

    // Janelas de áudio processadas (atualmente não utilizado).
    std::vector<Tensor> audioWindows = loadAndProcessAudio(subject.audio_file_path, params);
    cout << "  - Loaded and processed " << audioWindows.size() << " audio windows.\n";
}
