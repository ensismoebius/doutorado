// Voice biometrics demo in C++ following system.md architecture: capture/load audio,
// window + WPT energy, log-normalize, Poisson encode, stateful SNN forward, CSV output.

#include <algorithm>
#include <argparse/argparse.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "codificacao.hpp"
#include "nn/initializers/kaiming_snn.hpp"
#include "nn/layers/Leaky.hpp"
#include "nn/layers/Linear.hpp"
#include "nn/layers/Module.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/testing.hpp"
#include "nn/wave/Wav.h"
#include "nn/wavelet/waveletOperations.h"

using ::Leaky;
using ::Linear;
using argparse::ArgumentParser;
using nn::Index;
using nn::Tensor;
using std::cout;
using std::exception;
using std::llround;
using std::mt19937;
using std::ofstream;
using std::runtime_error;
using std::shared_ptr;
using std::sin;
using std::string;
using std::vector;

struct ConfigExtracao
{
    int taxa_amostragem{44100};
    int tamanho_janela{512};
    int tamanho_passo{256};
    int num_bandas{100};
    int nivel_wpt{6};
};

struct ConfigSNN
{
    int passos_por_janela{10};
    float alvo_spikes_por_passo{0.10F};
    int profundidade{3};
    int tamanho_camada_oculta{128};
};

// --- Signal helpers ---

auto compute_wpt_level(int window_size, int num_bandas) -> int
{
    const int nivel_max_por_tamanho =
        static_cast<int>(std::floor(std::log2(std::max(1, window_size))));
    const int nivel_necessario = static_cast<int>(std::ceil(std::log2(std::max(1, num_bandas))));
    return std::max(1, std::min(nivel_max_por_tamanho, nivel_necessario));
}

auto generate_hann_window(int tamanho) -> std::vector<double>
{
    std::vector<double> w(static_cast<size_t>(tamanho));
    for (int i = 0; i < tamanho; ++i)
    {
        constexpr double pi = std::numbers::pi;
        w[static_cast<size_t>(i)] =
            0.5 *
            (1.0 - std::cos(2.0 * pi * static_cast<double>(i) / static_cast<double>(tamanho - 1)));
    }
    return w;
}

auto apply_windowing(const std::vector<double>& sinal, const ConfigExtracao& cfg)
    -> std::vector<std::vector<double>>
{
    std::vector<std::vector<double>> janelas;
    if (sinal.size() < static_cast<size_t>(cfg.tamanho_janela))
    {
        return janelas;
    }

    const auto janela = generate_hann_window(cfg.tamanho_janela);
    for (int inicio = 0; inicio + cfg.tamanho_janela <= static_cast<int>(sinal.size());
         inicio += cfg.tamanho_passo)
    {
        std::vector<double> segmento(static_cast<size_t>(cfg.tamanho_janela));
        for (int i = 0; i < cfg.tamanho_janela; ++i)
        {
            segmento[static_cast<size_t>(i)] =
                sinal[static_cast<size_t>(inicio + i)] * janela[static_cast<size_t>(i)];
        }
        janelas.push_back(std::move(segmento));
    }
    return janelas;
}

auto interpolate_to_size(const std::vector<double>& src, int destino) -> std::vector<float>
{
    if (destino <= 0)
    {
        return {};
    }

    if (static_cast<int>(src.size()) == destino)
    {
        return std::vector<float>(src.begin(), src.end());
    }

    std::vector<float> out(static_cast<size_t>(destino));

    for (int i = 0; i < destino; ++i)
    {
        double pos = (static_cast<double>(i) / static_cast<double>(destino - 1)) *
                     static_cast<double>(src.size() - 1);
        auto idx0 = static_cast<size_t>(std::floor(pos));
        auto idx1 = static_cast<size_t>(std::ceil(pos));
        double frac = pos - static_cast<double>(idx0);
        double v0 = src[idx0];
        double v1 = src[idx1];
        out[static_cast<size_t>(i)] = static_cast<float>((1.0 - frac) * v0 + frac * v1);
    }

    return out;
}

auto compute_wpt_energy(const std::vector<double>& janela, int num_bandas, int nivel_wpt)
    -> std::vector<float>
{
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const std::vector<double> haar{inv_sqrt2, inv_sqrt2};

    // Pad to next power of two if needed (mallat expects power-of-two length)
    int alvo = wavelets::get_next_power_of_two(static_cast<double>(janela.size()));
    std::vector<double> sinal_padd(janela.begin(), janela.end());
    sinal_padd.resize(static_cast<size_t>(alvo), 0.0);

    auto transform = wavelets::malat(sinal_padd,
                                     std::span<const double>(haar.data(), haar.size()),
                                     wavelets::TransformMode::PACKET_WAVELET,
                                     static_cast<unsigned int>(nivel_wpt));
    auto energias = wavelets::extract_subband_energies(transform, nivel_wpt);
    return interpolate_to_size(energias, num_bandas);
}

auto preprocess_energy(const std::vector<float>& energia) -> std::vector<float>
{
    std::vector<float> out(energia.size());
    float max_v = 0.0F;
    for (size_t i = 0; i < energia.size(); ++i)
    {
        float v = std::log1pf(std::max(0.0F, energia[i]));
        out[i] = v;
        max_v = std::max(max_v, v);
    }
    if (max_v < 1e-8F)
    {
        std::fill(out.begin(), out.end(), 0.0F);
        return out;
    }
    for (auto& v : out)
    {
        v = std::clamp(v / max_v, 0.0F, 1.0F);
    }
    return out;
}

auto build_features(const std::vector<double>& audio, ConfigExtracao cfg)
    -> std::vector<std::vector<float>>
{
    cfg.nivel_wpt = compute_wpt_level(cfg.tamanho_janela, cfg.num_bandas);
    auto janelas = apply_windowing(audio, cfg);
    std::vector<std::vector<float>> features;
    features.reserve(janelas.size());
    for (const auto& j : janelas)
    {
        auto energia = compute_wpt_energy(j, cfg.num_bandas, cfg.nivel_wpt);
        features.push_back(preprocess_energy(energia));
    }
    return features;
}

// --- SNN model (Linear + Leaky with optional residual blocks) ---

struct ResidualBlock : public Module
{
    std::shared_ptr<Linear> fc1;
    std::shared_ptr<Leaky> lif1;
    std::shared_ptr<Linear> fc2;
    std::shared_ptr<Leaky> lif2;

    explicit ResidualBlock(int dim)
    {
        fc1 = std::make_shared<Linear>(dim, dim);
        lif1 = std::make_shared<Leaky>();
        fc2 = std::make_shared<Linear>(dim, dim);
        lif2 = std::make_shared<Leaky>();
    }

    auto forward(const Tensor& x, bool requires_grad) -> Tensor override
    {
        auto h1 = fc1->forward(x, requires_grad);
        auto s1 = lif1->forward(h1, requires_grad);
        auto h2 = fc2->forward(s1, requires_grad);
        auto s2 = lif2->forward(h2, requires_grad);
        return s2.add(x);
    }

    auto backward(const Tensor& grad_out) -> Tensor override
    {
        auto g2 = lif2->backward(grad_out);
        auto g1 = fc2->backward(g2);
        auto g0 = lif1->backward(g1);
        auto gin = fc1->backward(g0);
        return gin.add(grad_out);
    }

    auto params() -> std::vector<Tensor*> override
    {
        auto p = fc1->params();
        auto p2 = fc2->params();
        p.insert(p.end(), p2.begin(), p2.end());
        auto sg1 = lif1->params();
        auto sg2 = lif2->params();
        p.insert(p.end(), sg1.begin(), sg1.end());
        p.insert(p.end(), sg2.begin(), sg2.end());
        return p;
    }

    void reset_state() override
    {
        lif1->reset_state();
        lif2->reset_state();
    }
};

struct ModeloSNN : public Module
{
    std::shared_ptr<Linear> fc_in;
    std::shared_ptr<Leaky> lif_in;
    std::vector<shared_ptr<ResidualBlock>> residual_blocks;
    std::shared_ptr<Linear> fc_out;
    std::shared_ptr<Leaky> lif_out;

    ModeloSNN(int entradas, int saidas, int profundidade, int ocultos, unsigned int seed)
    {
        fc_in = std::make_shared<Linear>(entradas, ocultos);
        lif_in = std::make_shared<Leaky>();

        for (int i = 0; i < profundidade; ++i)
        {
            residual_blocks.push_back(std::make_shared<ResidualBlock>(ocultos));
        }

        fc_out = std::make_shared<Linear>(ocultos, saidas);
        lif_out = std::make_shared<Leaky>();

        const unsigned int base_seed = seed == 0 ? nn::testing::SEED : seed;

        kaimingSNNInitializer(fc_in, base_seed + 1U);
        kaimingSNNInitializer(fc_out, base_seed + 2U);

        unsigned int offset = 3U;

        for (auto& b : residual_blocks)
        {
            kaimingSNNInitializer(b->fc1, base_seed + offset);
            kaimingSNNInitializer(b->fc2, base_seed + offset + 1U);
            offset += 2U;
        }
    }

    auto forward(const Tensor& x, bool requires_grad) -> Tensor override
    {
        auto h = fc_in->forward(x, requires_grad);
        h = lif_in->forward(h, requires_grad);
        for (auto& b : residual_blocks)
        {
            h = b->forward(h, requires_grad);
        }
        h = fc_out->forward(h, requires_grad);
        return lif_out->forward(h, requires_grad);
    }

    auto backward(const Tensor& grad_out) -> Tensor override
    {
        auto g = lif_out->backward(grad_out);
        g = fc_out->backward(g);
        for (auto it = residual_blocks.rbegin(); it != residual_blocks.rend(); ++it)
        {
            g = (*it)->backward(g);
        }
        g = lif_in->backward(g);
        return fc_in->backward(g);
    }

    auto params() -> std::vector<Tensor*> override
    {
        auto p = fc_in->params();
        auto outp = fc_out->params();
        p.insert(p.end(), outp.begin(), outp.end());
        auto lifp = lif_in->params();
        auto lifout = lif_out->params();
        p.insert(p.end(), lifp.begin(), lifp.end());
        p.insert(p.end(), lifout.begin(), lifout.end());
        for (auto& b : residual_blocks)
        {
            auto bp = b->params();
            p.insert(p.end(), bp.begin(), bp.end());
        }
        return p;
    }

    void reset_state() override
    {
        lif_in->reset_state();
        lif_out->reset_state();
        for (auto& b : residual_blocks)
        {
            b->reset_state();
        }
    }
};

// --- Pipeline ---

auto load_audio_or_synthetic(const string& caminho, double duracao, int sample_rate)
    -> vector<double>
{
    if (!caminho.empty())
    {
        if (!std::filesystem::exists(caminho) || !std::filesystem::is_regular_file(caminho))
        {
            throw runtime_error("Arquivo de audio invalido: " + caminho);
        }

        Wav w;
        w.read(caminho);

        auto data = w.get_data();
        if (data.empty())
        {
            throw runtime_error("Arquivo de audio vazio ou invalido: " + caminho);
        }

        return data;
    }

    const size_t total = static_cast<size_t>(llround(duracao * static_cast<double>(sample_rate)));
    vector<double> out(total);
    constexpr double f1 = 440.0;
    constexpr double pi = std::numbers::pi;
    for (size_t i = 0; i < total; ++i)
    {
        out[i] =
            0.1 * sin(2.0 * pi * f1 * static_cast<double>(i) / static_cast<double>(sample_rate));
    }
    return out;
}

auto save_spikes_csv(const string& caminho, const vector<Tensor>& spikes) -> void
{
    ofstream out(caminho);
    if (!out.is_open())
    {
        throw runtime_error("Nao foi possivel abrir o arquivo de saida.");
    }
    int cols = spikes.empty() ? 0 : static_cast<int>(spikes.front().cols());
    out << "frame";
    for (int j = 0; j < cols; ++j)
    {
        out << ",band_" << j;
    }
    out << '\n';
    for (size_t i = 0; i < spikes.size(); ++i)
    {
        out << i;
        for (int j = 0; j < cols; ++j)
        {
            out << ',' << spikes[i].at(0, j);
        }
        out << '\n';
    }
}

auto run_pipeline(                      //
    const string& wav_path,             //
    double duracao_sintetica,           //
    const ConfigExtracao& cfg_extracao, //
    const ConfigSNN& cfg_snn,           //
    unsigned int seed,                  //
    const string& saida_csv             //
    ) -> void
{
    auto audio = load_audio_or_synthetic( //
        wav_path,                    //
        duracao_sintetica,           //
        cfg_extracao.taxa_amostragem //
    );

    auto features = build_features(audio, cfg_extracao);

    if (features.empty())
    {
        throw runtime_error("Nenhuma janela gerada.");
    }

    ModeloSNN modelo(                              //
        static_cast<int>(features.front().size()), //
        static_cast<int>(features.front().size()), //
        cfg_snn.profundidade,                      //
        cfg_snn.tamanho_camada_oculta,             //
        seed                                       //
    );

    mt19937 rng(seed == 0 ? nn::testing::SEED : seed);

    vector<Tensor> saida_por_janela;
    saida_por_janela.reserve(features.size());

    for (const auto& feat : features)
    {
        Tensor entrada(1, static_cast<Index>(feat.size()));

        for (Index j = 0; j < entrada.cols(); ++j)
        {
            entrada.at(0, j) = feat[static_cast<size_t>(j)];
        }

        auto spikes_in = codificacao::codificar_poisson( //
            entrada,                                     //
            cfg_snn.passos_por_janela,                   //
            rng,                                         //
            -1.0F,                                       //
            true,                                        //
            cfg_snn.alvo_spikes_por_passo                //
        );

        Tensor acumulado(1, static_cast<Index>(feat.size()));
        acumulado.setZero();
        modelo.reset_state();

        for (Index t = 0; t < spikes_in.rows(); ++t)
        {
            auto step = spikes_in.row(t);
            auto spk_out = modelo.forward(step, false);
            acumulado = acumulado.add(spk_out);
        }

        saida_por_janela.push_back(acumulado);
    }

    save_spikes_csv(saida_csv, saida_por_janela);
    cout << "Pipeline concluido. Saida em " << saida_csv << "\n";
}

int main(int argc, char** argv)
{
    ArgumentParser parser("voice_biometrics_cpp");
    parser.add_description("Pipeline WPT -> log-normalizacao -> codificacao Poisson -> SNN.");

    parser
        .add_argument("--entrada-wav") //
        .default_value(string(""))     //
        .help("WAV de entrada (opcional)");

    parser
        .add_argument("--saida-csv")                    //
        .default_value(string("resultado_spikes.csv")); //

    parser
        .add_argument("--duracao")                    //
        .default_value(1.0)                           //
        .help("Duracao sintetica se nao houver WAV"); //

    parser
        .add_argument("--taxa-amostragem") //
        .default_value(44100)              //
        .help("Taxa alvo (Hz)");           //

    parser
        .add_argument("--tamanho-janela") //
        .default_value(512);              //

    parser
        .add_argument("--tamanho-passo") //
        .default_value(256);             //

    parser
        .add_argument("--num-bandas") //
        .default_value(100);          //

    parser
        .add_argument("--passos-por-janela") //
        .default_value(10);                  //

    parser
        .add_argument("--profundidade") //
        .default_value(3);              //

    parser
        .add_argument("--hidden") //
        .default_value(128);      //

    parser
        .add_argument("--seed") //
        .default_value(42u);    //

    try
    {
        parser.parse_args(argc, argv);
    }
    catch (const exception& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }

    ConfigExtracao cfg_extr;
    cfg_extr.taxa_amostragem = parser.get<int>("--taxa-amostragem");
    cfg_extr.tamanho_janela = parser.get<int>("--tamanho-janela");
    cfg_extr.tamanho_passo = parser.get<int>("--tamanho-passo");
    cfg_extr.num_bandas = parser.get<int>("--num-bandas");

    ConfigSNN cfg_snn;
    cfg_snn.passos_por_janela = parser.get<int>("--passos-por-janela");
    cfg_snn.profundidade = parser.get<int>("--profundidade");
    cfg_snn.tamanho_camada_oculta = parser.get<int>("--hidden");

    auto seed = parser.get<unsigned int>("--seed");
    auto entrada = parser.get<std::string>("--entrada-wav");
    auto saida_csv = parser.get<std::string>("--saida-csv");
    auto duracao = parser.get<double>("--duracao");

    try
    {
        run_pipeline(entrada, duracao, cfg_extr, cfg_snn, seed, saida_csv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Erro: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
