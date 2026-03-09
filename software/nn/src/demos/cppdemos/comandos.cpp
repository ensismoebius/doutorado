#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Use project headers from include/nn
#include "codificacao.hpp"
#include "nn/initializers/kaiming_snn.hpp"
#include "nn/layers/Leaky.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/Sequential.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/testing.hpp"
#include "nn/wave/Wav.h"
#include "nn/wave/audioFeatureExtraction.h"

namespace demo
{

using nn::TensorImpl;

/**
 * Calcula uma frequência máxima (probabilidade) para o codificador Poisson.
 *
 * Ideia:
 * - Se a média das características (em [0,1]) for alta, diminuímos a frequência
 *   máxima para evitar saturação.
 * - Se a média for baixa, aumentamos para evitar neurônios "mortos".
 *
 * Retorna um escalar (float) para uso como `frequencia_max`.
 *
 * Comportamento (resumido):
 * - Calcula a média das features em [0,1].
 * - Define `frequencia = qtde_de_spikes_esperada_por_passo / mean_val` (aproximação)
 *   e limita em [frequencia_max_min, frequencia_max_max].
 * - Objetivo: ajustar a probabilidade por passo de forma que a expectativa de
 *   spikes por neurônio por passo fique próxima a `qtde_de_spikes_esperada_por_passo`.
 *
 * Exemplo numérico:
 * - `mean_val = 0.5`, `qtde_de_spikes_esperada_por_passo = 0.1` -> frequencia ~= 0.2
 *   (dado p = x * frequencia, a expectativa E[p] ~ frequencia * mean_val = 0.2 * 0.5 = 0.1).
 * - Com `passos = 10`, espera-se ~1 spike por neurônio por janela (10 * 0.1).
 */
// `calcular_taxa_max_adaptativa` and `codificar_poisson` are implemented in
// `codificacao.cpp` and declared in `codificacao.hpp`.

// Forward-declare the new signature so the old-wrapper can call it.
void cmd_demo(double duration, int sample_rate, int window_size, int hop_size,
              const std::string& wavelet, int num_bands, int steps_per_window, int depth,
              const std::string& plot_output, unsigned int random_seed /* = 0 */);

// Backwards-compatible wrapper (old callers expect no seed parameter).
void cmd_demo(                     // old signature
    double duration,               //
    int sample_rate,               //
    int window_size,               //
    int hop_size,                  //
    const std::string& wavelet,    //
    int num_bands,                 //
    int steps_per_window,          //
    int depth,                     //
    const std::string& plot_output //
)
{
    // Forward to the new signature with default seed = 0 (nondeterministic)
    cmd_demo(duration,
             sample_rate,
             window_size,
             hop_size,
             wavelet,
             num_bands,
             steps_per_window,
             depth,
             plot_output,
             0u);
}

// New signature with explicit seed for reproducibility.
void cmd_demo(                      //
    double duration,                //
    int sample_rate,                //
    int window_size,                //
    int hop_size,                   //
    const std::string& wavelet,     //
    int num_bands,                  //
    int steps_per_window,           //
    int depth,                      //
    const std::string& plot_output, //
    unsigned int random_seed = 0    // optional seed for reproducibility (0 => nondeterministic)
)
{
    // RNG for Poisson encoder (matches Python's uniform sampling behavior)
    std::mt19937 rng;
    if (random_seed != 0)
    {
        rng.seed(random_seed);
        std::cerr << "Using fixed random seed: " << random_seed << '\n';
    }
    else
    {
        std::random_device rd;
        rng.seed(rd());
    }
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    auto extrair_janelas_caracteristicas = [&](std::vector<float>& audio)
    {
        LoadingAndProcessingParameters loading_params{};
        loading_params.audio_params.target_sampling_rate = sample_rate;
        loading_params.audio_params.preemphasis_coefficient = 0.97f;
        loading_params.audio_params.frame_duration_ms =
            static_cast<float>(window_size) * 1000.0f / sample_rate;
        loading_params.audio_params.frame_shift_ms =
            static_cast<float>(hop_size) * 1000.0f / sample_rate;
        loading_params.audio_params.number_of_filters = num_bands;
        loading_params.constants.default_sampling_rate = sample_rate;

        FramingConfig framing_cfg{window_size, hop_size, loading_params};

        std::vector<float> center_frequencies;
        nn::Tensor filterbank(1, 1);
        FilterbankConfig fb_cfg{filterbank, center_frequencies, loading_params};
        nn::core::wave::build_linear_filterbank(512, fb_cfg);

        nn::core::wave::pre_emphasis_inplace(audio,
                                             loading_params.audio_params.preemphasis_coefficient);
        auto frames = nn::core::wave::framing_and_window(audio, framing_cfg);
        auto power = nn::core::wave::rfft_power(frames, 512);
        PowerFilterbankConfig pf_cfg{fb_cfg.filterbank, loading_params};
        auto filt = nn::core::wave::dot_power_filterbank(power, pf_cfg);
        auto features = nn::core::wave::dct2(filt, loading_params);
        return features;
    };

    auto criar_modelo_snn = [&](int in_dim) -> std::shared_ptr<Sequential>
    {
        int hidden = std::max(8, num_bands);
        auto m = std::make_shared<Sequential>();
        auto l1 = std::make_shared<Linear>(in_dim, hidden);
        auto lk = std::make_shared<Leaky>(1.0f, 1.0f, 1.0f, 1.0f, true);
        auto l2 = std::make_shared<Linear>(hidden, in_dim);
        m->add_module(l1);
        m->add_module(lk);
        m->add_module(l2);
        kaimingSNNInitializer(l1, nn::testing::SEED);
        kaimingSNNInitializer(l2, nn::testing::SEED);
        return m;
    };

    auto executar_inferencia = [&](std::shared_ptr<Sequential> model, const nn::Tensor& features)
    {
        std::vector<nn::Tensor> lista_spikes;
        int in_dim = static_cast<int>(features.cols());
        model->reset_state();
        for (int fi = 0; fi < features.rows(); ++fi)
        {
            nn::Tensor frm(1, features.cols());
            for (int j = 0; j < features.cols(); ++j) frm.at(0, j) = features.at(fi, j);
            nn::Tensor spikes = codificacao::codificar_poisson(frm, steps_per_window, rng);
            nn::Tensor accum(1, in_dim);
            accum.setZero();
            for (int t = 0; t < static_cast<int>(spikes.rows()); ++t)
            {
                auto step = spikes.row(static_cast<nn::Index>(t));
                auto out = model->forward(step, /*requires_grad=*/false);
                accum = accum.add(out);
            }
            lista_spikes.push_back(accum);
        }
        return lista_spikes;
    };

    auto plotar_resultados = [&](const std::string& outpath,
                                 const std::vector<nn::Tensor>& lista_spikes,
                                 const std::vector<float>& audio,
                                 int fs)
    {
        std::ofstream fout(outpath);
        int in_dim = lista_spikes.empty() ? 0 : lista_spikes[0].cols();
        fout << "frame,";
        for (int j = 0; j < in_dim; ++j)
        {
            if (j) fout << ',';
            fout << "band_" << j;
        }
        fout << '\n';
        for (size_t i = 0; i < lista_spikes.size(); ++i)
        {
            fout << i << ',';
            for (int j = 0; j < lista_spikes[i].cols(); ++j)
            {
                if (j) fout << ',';
                fout << lista_spikes[i].at(0, j);
            }
            fout << '\n';
        }
        fout.close();

        Wav w;
        std::string wav_out = outpath + ".wav";
        w.write(wav_out, audio, fs);
        std::cerr << "Wrote demo CSV to " << outpath << " and WAV to " << wav_out << '\n';
    };

    // --- main pipeline using the helper lambdas above ---
    size_t total_samples = static_cast<size_t>(std::round(duration * sample_rate));
    std::vector<float> audio(total_samples);
    for (size_t i = 0; i < total_samples; ++i)
    {
        audio[i] = 0.1f * std::sin(2.0 * M_PI * 440.0 * static_cast<double>(i) / sample_rate);
    }

    auto features = extrair_janelas_caracteristicas(audio);
    int in_dim = static_cast<int>(features.cols());
    auto model = criar_modelo_snn(in_dim);
    auto lista_spikes_saida = executar_inferencia(model, features);
    plotar_resultados(plot_output, lista_spikes_saida, audio, sample_rate);
}

} // namespace demo
