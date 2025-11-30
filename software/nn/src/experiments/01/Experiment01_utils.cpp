#define USE_MATH_DEFINES

#include "Experiment01_utils.h"

#include <fftw3.h> // For FFTW library functions

#include <cstddef>  // For size_t
#include <iostream> // For std::cout (if debugging)
#include <vector>   // For std::vector

#include "core/dataLoaders/10.1117/AudioLoader.h" // For loadAudioFromMat
#include "core/optimizers/Adam.hpp"
#include "core/tensor/Tensor.hpp"             // For Tensor
#include "core/wave/audioFeatureExtraction.h" // Include the new header

// Include Eigen for Eigen::Map and Eigen::VectorXf
#include <Eigen/Dense>

using nn::dataLoaders::loadAudioFromMat;
using std::size_t;
using std::vector;
using namespace std; // Use standard namespace

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
 * @param audioFilePath Caminho do arquivo de áudio.
 * @param loading_params Parâmetros de carregamento e processamento.
 * @return Vetor de tensores contendo os LFCCs e seus deltas.
 */
auto loadAndProcessAudio(const std::string& audioFilePath,
                         const LoadingAndProcessingParameters& loading_params)
    -> std::vector<Tensor>
{
    // Amostras de áudio carregadas do arquivo .mat.
    auto [audioSamples, audioStimulus, eegIndex] = loadAudioFromMat(audioFilePath, 0);

    // Vetor de float para conter os dados de áudio para processamento.
    vector<float> input_data(static_cast<size_t>(audioSamples.size()));

    // Frames do sinal após janelamento.
    vector<vector<float>> frames;

    // Espectro de potência de cada frame.
    Tensor power_spectrum;

    // Matriz do banco de filtros lineares.
    Tensor filterbank_local;                // Local variable for filterbank
    vector<float> center_frequencies_local; // Local variable for center_frequencies

    // Energias logarítmicas após aplicação do banco de filtros.
    Tensor log_energies;

    // Coeficientes cepstrais (LFCC) calculados.
    Tensor cepstral_coeff;

    // Coeficientes delta (primeira derivada).
    Tensor delta_coeff;

    // Coeficientes delta-delta (segunda derivada).
    Tensor delta_delta_coeff;

    // Transpõe a matriz de amostras para facilitar o processamento.
    audioSamples = audioSamples.transpose();

    // Copia os dados do áudio para o vetor de processamento.
    Eigen::Map<Eigen::VectorXf>(input_data.data(), audioSamples.size()) = audioSamples;

    // Etapa 1: Pré-ênfase (now nn::core::wave::pre_emphasis_inplace)
    nn::core::wave::pre_emphasis_inplace(
        input_data,                                         // Input audio data
        loading_params.audio_params.preemphasis_coefficient // Pre-emphasis coefficient
    );

    // Etapa 2: Framing e Janelamento (Hamming)
    FramingConfig framing_context = {
        .frame_length = 0,               // Initialized to 0, will be set by framing_and_window
        .frame_step = 0,                 // Initialized to 0, will be set by framing_and_window
        .loading_params = loading_params // Parâmetros de carregamento e processamento
    };
    frames = nn::core::wave::framing_and_window(input_data, framing_context);

    // Etapas 3 & 4: STFT (via FFT) e cálculo do Espectro de Potência
    power_spectrum = nn::core::wave::rfft_power(frames, framing_context.frame_length);

    // Etapa 5: Construção do banco de filtros lineares
    FilterbankConfig filterbank_context = {
        .filterbank = filterbank_local,                 // Matriz do banco de filtros (saída)
        .center_frequencies = center_frequencies_local, // Frequências centrais (saída)
        .loading_params = loading_params // Parâmetros de carregamento e processamento
    };
    nn::core::wave::build_linear_filterbank(framing_context.frame_length, filterbank_context);

    // Etapa 6: Aplicação do banco de filtros e compressão logarítmica
    PowerFilterbankConfig power_filterbank_context = {
        .filterbank = filterbank_local,  // Banco de filtros
        .loading_params = loading_params // Parâmetros de carregamento e processamento
    };
    log_energies = nn::core::wave::dot_power_filterbank(power_spectrum, power_filterbank_context);

    // Etapa 7: DCT para obter os coeficientes cepstrais (LFCC)
    cepstral_coeff = nn::core::wave::dct2(log_energies, loading_params);

    // (Extra) Etapa 8: Cálculo dos deltas (derivadas temporais)
    delta_coeff = nn::core::wave::compute_deltas(cepstral_coeff, loading_params);
    delta_delta_coeff = nn::core::wave::compute_deltas(delta_coeff, loading_params);

    return {cepstral_coeff, delta_coeff, delta_delta_coeff}; // Return calculated tensors
}

/**
 * @brief Processa um sujeito específico, carregando e processando seu áudio.
 *
 * @param subject Informações do sujeito a ser processado.
 */
void processSubject(const SubjectInfo& subject)
{
    cout << "Processing subject: " << subject.name << '\n';

    const AudioProcessingParams audio_processing_params = {.target_sampling_rate = 44100,
                                                           .preemphasis_coefficient = 0.97F,
                                                           .frame_duration_ms = 25.0,
                                                           .frame_shift_ms = 10.0,
                                                           .number_of_filters = 24,
                                                           .number_of_cepstrals = 19,
                                                           .delta_window_span = 2};

    constexpr HammingWindowConfig hamming_window_config = {.alpha = 0.54F, .beta = 0.46F};

    constexpr DctConfig dct_config = {.normalization_factor_sqrt = 2.0F,
                                      .filter_index_offset = 0.5F};

    constexpr DeltaConfig delta_config = {.denominator_factor = 2.0F};

    constexpr GeneralConstants general_constants = {.ms_to_seconds_factor = 1000.0F,
                                                    .min_log_energy = 1e-12F,
                                                    .default_sampling_rate = 44100,
                                                    .debug_frame_limit = 5};

    const LoadingAndProcessingParameters loading_params = {
        .audio_params = audio_processing_params,
        .hamming_window_config = hamming_window_config,
        .dct_config = dct_config,
        .delta_config = delta_config,
        .constants = general_constants};

    std::vector<Tensor> audio_windows =
        loadAndProcessAudio(subject.audio_file_path, loading_params);
    cout << "  - Loaded and processed " << audio_windows.size() << " audio windows.\n";

    // Extract MGDF cepstral from frames
    inline Eigen::VectorXd extract_mgdf_from_frame(
        const Eigen::VectorXf& frame, 
        const LFCCConfig& lfc_p, 
        const MGDFParams& mg_p, 
        const Eigen::MatrixXd& H)
    {
        Eigen::VectorXd mgdf_cepstral;
        nn::core::wave::extract_mgdf_cepstral_from_frame(frame, lfc_p, mg_p, H, mgdf_cepstral);
        return mgdf_cepstral;
    }
}