// Voice biometrics demo in C++ following system.md architecture: capture/load audio,
// window + WPT energy, log-normalize, Poisson encode, stateful SNN forward, CSV output.

#include <algorithm>
#include <argparse/argparse.hpp>
#include <cmath>
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

auto calcular_nivel_wpt(int window_size, int num_bandas) -> int
{
    const int nivel_max_por_tamanho =
        static_cast<int>(std::floor(std::log2(std::max(1, window_size))));
    const int nivel_necessario = static_cast<int>(std::ceil(std::log2(std::max(1, num_bandas))));
    return std::max(1, std::min(nivel_max_por_tamanho, nivel_necessario));
}

auto gerar_janela_hann(int tamanho) -> std::vector<double>
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

auto aplicar_janelamento(const std::vector<double>& sinal, const ConfigExtracao& cfg)
    -> std::vector<std::vector<double>>
{
    std::vector<std::vector<double>> janelas;
    if (sinal.size() < static_cast<size_t>(cfg.tamanho_janela))
    {
        return janelas;
    }

    const auto janela = gerar_janela_hann(cfg.tamanho_janela);
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

auto interpolar(const std::vector<double>& src, int destino) -> std::vector<float>
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

auto calcular_energia_wpt(const std::vector<double>& janela, int num_bandas, int nivel_wpt)
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
    return interpolar(energias, num_bandas);
}

auto preprocessar_energia(const std::vector<float>& energia) -> std::vector<float>
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

auto construir_features(const std::vector<double>& audio, ConfigExtracao cfg)
    -> std::vector<std::vector<float>>
{
    cfg.nivel_wpt = calcular_nivel_wpt(cfg.tamanho_janela, cfg.num_bandas);
    auto janelas = aplicar_janelamento(audio, cfg);
    std::vector<std::vector<float>> features;
    features.reserve(janelas.size());
    for (const auto& j : janelas)
    {
        auto energia = calcular_energia_wpt(j, cfg.num_bandas, cfg.nivel_wpt);
        features.push_back(preprocessar_energia(energia));
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
    std::vector<std::shared_ptr<ResidualBlock>> blocks;
    std::shared_ptr<Linear> fc_out;
    std::shared_ptr<Leaky> lif_out;

    ModeloSNN(int entradas, int saidas, int profundidade, int ocultos, unsigned int seed)
    {
        fc_in = std::make_shared<Linear>(entradas, ocultos);
        lif_in = std::make_shared<Leaky>();
        for (int i = 0; i < profundidade; ++i)
        {
            blocks.push_back(std::make_shared<ResidualBlock>(ocultos));
        }
        fc_out = std::make_shared<Linear>(ocultos, saidas);
        lif_out = std::make_shared<Leaky>();

        const unsigned int base_seed = seed == 0 ? nn::testing::SEED : seed;
        kaimingSNNInitializer(fc_in, base_seed + 1U);
        kaimingSNNInitializer(fc_out, base_seed + 2U);
        unsigned int offset = 3U;
        for (auto& b : blocks)
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
        for (auto& b : blocks)
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
        for (auto it = blocks.rbegin(); it != blocks.rend(); ++it)
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
        for (auto& b : blocks)
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
        for (auto& b : blocks)
        {
            b->reset_state();
        }
    }
};

// --- Pipeline ---

auto carregar_audio(const std::string& caminho, double duracao, int sample_rate)
    -> std::vector<double>
{
    if (!caminho.empty())
    {
        Wav w;
        w.read(caminho);
        return w.get_data();
    }

    const size_t total =
        static_cast<size_t>(std::llround(duracao * static_cast<double>(sample_rate)));
    std::vector<double> out(total);
    constexpr double f1 = 440.0;
    constexpr double pi = std::numbers::pi;
    for (size_t i = 0; i < total; ++i)
    {
        out[i] = 0.1 * std::sin(2.0 * pi * f1 * static_cast<double>(i) /
                                static_cast<double>(sample_rate));
    }
    return out;
}

auto salvar_csv(const std::string& caminho, const std::vector<Tensor>& spikes) -> void
{
    std::ofstream out(caminho);
    if (!out.is_open())
    {
        throw std::runtime_error("Nao foi possivel abrir o arquivo de saida.");
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

auto executar_pipeline(const std::string& wav_path, double duracao_sintetica,
                       const ConfigExtracao& cfg_extracao, const ConfigSNN& cfg_snn,
                       unsigned int seed, const std::string& saida_csv) -> void
{
    auto audio = carregar_audio(wav_path, duracao_sintetica, cfg_extracao.taxa_amostragem);
    auto features = construir_features(audio, cfg_extracao);
    if (features.empty())
    {
        throw std::runtime_error("Nenhuma janela gerada.");
    }

    ModeloSNN modelo(static_cast<int>(features.front().size()),
                     static_cast<int>(features.front().size()),
                     cfg_snn.profundidade,
                     cfg_snn.tamanho_camada_oculta,
                     seed);

    std::mt19937 rng(seed == 0 ? nn::testing::SEED : seed);

    std::vector<Tensor> saida_por_janela;
    saida_por_janela.reserve(features.size());

    for (const auto& feat : features)
    {
        Tensor entrada(1, static_cast<Index>(feat.size()));
        for (Index j = 0; j < entrada.cols(); ++j)
        {
            entrada.at(0, j) = feat[static_cast<size_t>(j)];
        }

        auto spikes_in = codificacao::codificar_poisson(
            entrada, cfg_snn.passos_por_janela, rng, -1.0F, true, cfg_snn.alvo_spikes_por_passo);

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

    salvar_csv(saida_csv, saida_por_janela);
    std::cout << "Pipeline concluido. Saida em " << saida_csv << "\n";
}

int main(int argc, char** argv)
{
    ArgumentParser program("voice_biometrics_cpp");
    program.add_description("Pipeline WPT -> log-normalizacao -> codificacao Poisson -> SNN.");

    program.add_argument("--entrada-wav")
        .default_value(std::string(""))
        .help("WAV de entrada (opcional)");
    program.add_argument("--saida-csv").default_value(std::string("resultado_spikes.csv"));
    program.add_argument("--duracao")
        .default_value(1.0)
        .help("Duracao sintetica se nao houver WAV");
    program.add_argument("--taxa-amostragem").default_value(44100).help("Taxa alvo (Hz)");
    program.add_argument("--tamanho-janela").default_value(512);
    program.add_argument("--tamanho-passo").default_value(256);
    program.add_argument("--num-bandas").default_value(100);
    program.add_argument("--passos-por-janela").default_value(10);
    program.add_argument("--profundidade").default_value(3);
    program.add_argument("--hidden").default_value(128);
    program.add_argument("--seed").default_value(42u);

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }

    ConfigExtracao cfg_extr;
    cfg_extr.taxa_amostragem = program.get<int>("--taxa-amostragem");
    cfg_extr.tamanho_janela = program.get<int>("--tamanho-janela");
    cfg_extr.tamanho_passo = program.get<int>("--tamanho-passo");
    cfg_extr.num_bandas = program.get<int>("--num-bandas");

    ConfigSNN cfg_snn;
    cfg_snn.passos_por_janela = program.get<int>("--passos-por-janela");
    cfg_snn.profundidade = program.get<int>("--profundidade");
    cfg_snn.tamanho_camada_oculta = program.get<int>("--hidden");

    auto seed = program.get<unsigned int>("--seed");
    auto entrada = program.get<std::string>("--entrada-wav");
    auto saida_csv = program.get<std::string>("--saida-csv");
    auto duracao = program.get<double>("--duracao");

    try
    {
        if (entrada.empty())
        {
            // Generate synthetic audio with desired duration
            auto audio = carregar_audio("", duracao, cfg_extr.taxa_amostragem);
            auto features = construir_features(audio, cfg_extr);
            if (features.empty())
            {
                throw std::runtime_error("Nenhuma janela gerada.");
            }
        }

        executar_pipeline(entrada, duracao, cfg_extr, cfg_snn, seed, saida_csv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Erro: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
