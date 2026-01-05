#include "nn/wave/audioFeatureExtraction.h"

#include <fftw3.h> // For FFTW library functions

#include <algorithm> // For std::min, std::max
#include <cmath>     // For cosf, roundf, floorf, logf, sqrtf
#include <cstddef>   // For size_t
#include <numbers>   // For std::numbers::pi_v, std::numbers::sqrt2_v
#include <vector>    // For std::vector

#include "nn/optimizers/Adam.hpp" // For Adam optimizer
#include "nn/tensor/Tensor.hpp"   // For Tensor

using std::size_t;
using std::vector;
using namespace std; // Resolve cout errors

namespace nn::core::wave
{

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
void pre_emphasis_inplace(vector<float>& signal, float coefficient)
{
    if (signal.empty())
    {
        return; // Nothing to do for an empty signal
    }
    // Aplica o filtro de pré-ênfase ao sinal.
    for (size_t i = signal.size() - 1; i > 0; --i)
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
 * @param signal Sinal de áudio.
 * @param context Contexto contendo parâmetros e saídas.
 * @return Frames do sinal após janelamento.
 */
auto framing_and_window(const vector<float>& signal, FramingConfig& context)
    -> vector<vector<float>>
{
    if (signal.empty())
    {
        return {}; // Return empty frames for an empty signal
    }
    // Comprimento de cada frame em amostras. (Parâmetro de saída)
    context.frame_length = static_cast<int>(
        roundf(context.loading_params.audio_params.frame_duration_ms *
               static_cast<float>(context.loading_params.audio_params.target_sampling_rate) /
               context.loading_params.constants.ms_to_seconds_factor));

    // Deslocamento entre frames consecutivos em amostras. (Parâmetro de saída)
    context.frame_step = static_cast<int>(
        roundf(context.loading_params.audio_params.frame_shift_ms *
               static_cast<float>(context.loading_params.audio_params.target_sampling_rate) /
               context.loading_params.constants.ms_to_seconds_factor));

    // Comprimento total do sinal de entrada.
    const int signal_length = static_cast<int>(signal.size());

    // Número total de frames que serão extraídos do sinal.
    const int number_of_frames =
        1 + std::max(0, (signal_length - context.frame_length) / context.frame_step);

    // Comprimento do sinal após o padding para conter todos os frames.
    const int padded_length = (number_of_frames * context.frame_step) + context.frame_length;

    // Cópia do sinal com padding de zeros no final.
    vector<float> padded_signal(padded_length, 0.0F);
    std::copy(signal.begin(), signal.end(), padded_signal.begin());

    // Vetor para armazenar a função de janelamento (Hamming).
    vector<float> window_function(context.frame_length);

    // Matriz para armazenar os frames resultantes após o janelamento.
    vector<vector<float>> frames(number_of_frames, vector<float>(context.frame_length));

    // Janela de Hamming: w[n] = 0.54 - 0.46 * cos(2πn / (N-1))
    for (int i = 0; i < context.frame_length; ++i)
    {
        window_function[i] = context.loading_params.hamming_window_config.alpha -
                             (context.loading_params.hamming_window_config.beta *
                              cosf(2 * std::numbers::pi_v<float> * static_cast<float>(i) /
                                   (static_cast<float>(context.frame_length) - 1))); // Hamming
    }

    // Aplicação do janelamento em cada frame.
    for (int i = 0; i < number_of_frames; ++i)
    {
        // Índice do frame atual.
        const int start_index = i * context.frame_step;

        // Preenchimento do frame atual com a janela aplicada.
        for (int j = 0; j < context.frame_length; ++j)
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
 * @param frames Frames do sinal após janelamento.
 * @param fft_points Número de pontos da FFT (igual ao comprimento do frame).
 * @return Espectro de potência de cada frame.
 */
auto rfft_power(const vector<vector<float>>& frames, int fft_points) -> nn::Tensor
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
    nn::Tensor power_spectrum(static_cast<int>(number_of_frames), static_cast<int>(number_of_bins));

    // Buffer de entrada para a FFTW (real).
    auto* fftw_input = static_cast<float*>(fftwf_malloc(sizeof(float) * fft_points));

    // Buffer de saída para a FFTW (complexo).
    auto* fftw_output =
        static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex) * (fft_points / 2 + 1)));

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

        // Comprimento a ser copiado para o buffer da FFT, o mínimo entre o comprimento do frame e o
        // tamanho da FFT.
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
            power_spectrum(static_cast<long>(frame_index), static_cast<long>(bin_index)) =
                (real_part * real_part + imaginary_part * imaginary_part) /
                static_cast<float>(fft_points);
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
 * @param context Contexto contendo parâmetros e saídas.
 */
void build_linear_filterbank(int fft_points, FilterbankConfig& context)
{
    // Número de bins de frequência resultantes da FFT (N/2 + 1).
    // Um "bin" representa uma pequena faixa de frequência na saída da FFT.
    const int number_of_bins = (fft_points / 2) + 1;

    // Vetor de índices dos bins correspondentes às frequências centrais.
    vector<int> bin_indices(context.loading_params.audio_params.number_of_filters + 2);

    // Frequência máxima representada (Metade da frequência de amostragem).
    const float max_frequency =
        static_cast<float>(context.loading_params.constants.default_sampling_rate) / 2.0F;

    // Inicialização do banco de filtros e das frequências centrais.
    context.filterbank =
        nn::Tensor(context.loading_params.audio_params.number_of_filters, number_of_bins);

    // Inicialização das frequências centrais.
    context.center_frequencies.resize(context.loading_params.audio_params.number_of_filters + 2);

    // Cálculo das frequências centrais dos filtros.
    for (int i = 0; i < context.loading_params.audio_params.number_of_filters + 2; ++i)
    {
        context.center_frequencies[i] =
            (max_frequency) * static_cast<float>(i) /
            static_cast<float>(context.loading_params.audio_params.number_of_filters + 1);
    }

    // Cálculo dos índices dos bins para as frequências centrais.
    for (size_t i = 0; i < context.center_frequencies.size(); ++i)
    {
        // Índice para mapear frequências centrais para bins da FFT.
        bin_indices[i] = static_cast<int>(
            floorf((static_cast<float>(fft_points) + 1.0F) * context.center_frequencies[i] /
                   static_cast<float>(context.loading_params.audio_params.target_sampling_rate)));
    }

    // Construção dos filtros triangulares.
    for (int filter_index = 1;
         filter_index <= context.loading_params.audio_params.number_of_filters;
         ++filter_index)
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
                context.filterbank(filter_index - 1, bin_index) =
                    static_cast<float>(bin_index - previous_bin_index) /
                    static_cast<float>(current_bin_index - previous_bin_index);
            }
        }
        // Parte descendente do filtro triangular.
        if (next_bin_index > current_bin_index)
        {
            // Parte descendente do filtro triangular.
            for (int bin_index = current_bin_index; bin_index < next_bin_index; ++bin_index)
            {
                context.filterbank(filter_index - 1, bin_index) =
                    static_cast<float>(next_bin_index - bin_index) /
                    static_cast<float>(next_bin_index - current_bin_index);
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
 * @param power_spectrum Espectro de potência de cada frame.
 * @param context Contexto contendo parâmetros e o banco de filtros.
 * @return Energias logarítmicas das bandas de filtro.
 */
auto dot_power_filterbank(const nn::Tensor& power_spectrum, const PowerFilterbankConfig& context)
    -> nn::Tensor
{
    // Número de frames no espectro de potência.
    const long number_of_frames = static_cast<long>(power_spectrum.rows());

    // Número de filtros no banco de filtros.
    const long number_of_filters = static_cast<long>(context.filterbank.rows());

    // Número de bins de frequência por frame.
    const long number_of_bins = static_cast<long>(power_spectrum.cols());

    // Matriz para armazenar as energias logarítmicas resultantes.
    nn::Tensor log_energies(number_of_frames, number_of_filters);

    // Cálculo das energias logarítmicas para cada frame e filtro.
    for (long frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        // Índice do frame atual.
        for (long filter_index = 0; filter_index < number_of_filters; ++filter_index)
        {
            // Variável temporária para acumular a soma ponderada do espectro de potência.
            float sum = 0.0F;
            for (long bin_index = 0; bin_index < number_of_bins; ++bin_index)
            {
                // Índice do bin de frequência atual.
                sum += power_spectrum(frame_index, bin_index) *
                       context.filterbank(filter_index, bin_index);
            }
            sum = std::max(sum,
                           context.loading_params.constants.min_log_energy); // Evita log(0)
            log_energies(frame_index, filter_index) = logf(sum);
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
 * @param log_energies Energias logarítmicas das bandas de filtro.
 * @param loading_params Parâmetros de carregamento e processamento.
 * @return Coeficientes cepstrais (LFCC).
 */
auto dct2(const nn::Tensor& log_energies, const LoadingAndProcessingParameters& loading_params)
    -> nn::Tensor
{
    // Número de frames (vetores de energia) de entrada.
    const size_t number_of_frames = log_energies.rows();

    // Número de filtros, que corresponde ao número de energias por frame.
    const size_t number_of_filters = log_energies.cols();

    // Matriz para armazenar os coeficientes cepstrais resultantes.
    nn::Tensor cepstral_coefficients((long) number_of_frames,
                                     loading_params.audio_params.number_of_cepstrals);

    // Cálculo dos coeficientes cepstrais para cada frame.
    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        // Cria os coeficientes cepstrais para o frame atual.
        for (size_t cepstrum_index = 0;
             cepstrum_index < loading_params.audio_params.number_of_cepstrals;
             ++cepstrum_index)
        {
            // Variável temporária para acumular a soma ponderada para o cálculo do coeficiente
            // cepstral.
            float sum = 0.0F;

            // Cálculo do coeficiente cepstral atual.
            for (long filter_index = 0; filter_index < static_cast<long>(number_of_filters);
                 ++filter_index)
            {
                // Acumula a soma ponderada usando a fórmula da DCT-II.
                sum += log_energies(    // Energia logarítmica
                           frame_index, // Índice do frame
                           filter_index // Índice do filtro
                           ) *
                       cosf(std::numbers::pi_v<float> * static_cast<float>(cepstrum_index) *
                            (static_cast<float>(filter_index) +
                             loading_params.dct_config.filter_index_offset) /
                            static_cast<float>(number_of_filters) // Índice do filtro normalizado
                       );
            }

            // Normalização ortogonal, armazenando o coeficiente cepstral calculado.
            cepstral_coefficients(frame_index, cepstrum_index) =
                sum * sqrtf(loading_params.dct_config.normalization_factor_sqrt /
                            static_cast<float>(number_of_filters));

            // Ajuste do primeiro coeficiente cepstral.
            if (cepstrum_index == 0)
            {
                cepstral_coefficients(frame_index, cepstrum_index) *=
                    1.0F / std::numbers::sqrt2_v<float>;
            }
        }
    }
    return cepstral_coefficients;
}

/**
 * @brief Calcula os deltas (derivada temporal) dos features.
 * Os deltas capturam a dinâmica temporal dos coeficientes cepstrais.
 *
 * @param features Matriz de features (LFCCs).
 * @param loading_params Parâmetros de carregamento e processamento.
 * @return Matriz com os coeficientes delta.
 */
auto compute_deltas(const nn::Tensor& features,
                    const LoadingAndProcessingParameters& loading_params) -> nn::Tensor
{
    // Número de frames (vetores de features) de entrada.
    const long number_of_frames = features.rows();
    if (number_of_frames == 0)
    {
        // Return an explicitly empty tensor (0x0)
        return nn::Tensor(0, 0);
    }

    // Dimensionalidade do vetor de features.
    const size_t number_of_features = features.cols();

    // Denominador da fórmula de cálculo dos deltas, pré-calculado.
    float denominator = 0.0F;

    // Matriz para armazenar os coeficientes delta resultantes.
    nn::Tensor delta_features(number_of_frames, (long) number_of_features);

    // Cálculo do denominador da fórmula de deltas.
    for (int i = 1; i <= loading_params.audio_params.delta_window_span; ++i)
    {
        denominator += static_cast<float>(i) * static_cast<float>(i);
    }

    // Essa multiplicação é feita pois o denominador é usado duas vezes na fórmula de deltas.
    denominator *= loading_params.delta_config.denominator_factor;

    // Cálculo dos coeficientes delta para cada frame e feature.
    for (size_t frame_index = 0; frame_index < number_of_frames; ++frame_index)
    {
        // Itera sobre cada dimensão do feature vector.
        for (size_t feature_index = 0; feature_index < number_of_features; ++feature_index)
        {
            // Variável temporária para acumular o numerador da fórmula de deltas.
            float numerator = 0.0F;

            // Cálculo do numerador para o delta da feature atual.
            for (size_t delta_span = 1; delta_span <= loading_params.audio_params.delta_window_span;
                 ++delta_span)
            {
                // Determine indices for c_{t+n} and c_{t-n} with boundary handling
                size_t index_plus_n = frame_index + delta_span;
                long index_minus_n = frame_index - delta_span;

                // Clamp indices to valid range
                if (index_plus_n >= number_of_frames)
                {
                    index_plus_n = number_of_frames - 1;
                }
                if (index_minus_n < 0)
                {
                    index_minus_n = 0;
                }

                // Deslocamento para calcular a diferença entre frames.
                numerator += static_cast<float>(delta_span) * // Peso baseado na distância temporal
                             (features(index_plus_n, feature_index) -
                              features(index_minus_n, feature_index));
            }

            // Cálculo do coeficiente delta para o frame e feature atuais.
            delta_features(frame_index, feature_index) = numerator / denominator;
        }
    }

    // Retorno dos coeficientes delta calculados.
    return delta_features;
}

// Windowing functions
auto hanning_window(int length) -> std::vector<double>
{
    if (length <= 0) return {};
    if (length == 1) return {1.0};
    std::vector<double> window(length);
    for (int i = 0; i < length; ++i)
    {
        window[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (length - 1)));
    }
    return window;
}

auto apply_window(const std::vector<double>& signal, const std::vector<double>& window)
    -> std::vector<double>
{
    if (signal.size() != window.size())
    {
        throw std::invalid_argument("Signal and window must have the same size");
    }
    std::vector<double> result(signal.size());
    for (size_t i = 0; i < signal.size(); ++i)
    {
        result[i] = signal[i] * window[i];
    }
    return result;
}

} // namespace nn::core::wave
