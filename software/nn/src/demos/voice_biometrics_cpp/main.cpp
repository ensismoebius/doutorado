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

struct ExtractionConfig
{
    int sample_rate{44100};
    int window_size{512};
    int hop_size{256};
    int num_bands{100};
    int wpt_level{6};
};

struct SnnConfig
{
    int steps_per_window{10};
    float target_spikes_per_step{0.10F};
    int depth{3};
    int hidden_size{128};
};

// --- Signal helpers ---

auto compute_wpt_level(int window_size, int num_bandas) -> int
{
    const int nivel_max_por_tamanho =
        static_cast<int>(std::floor(std::log2(std::max(1, window_size))));
    const int nivel_necessario = static_cast<int>(std::ceil(std::log2(std::max(1, num_bandas))));
    return std::max(1, std::min(nivel_max_por_tamanho, nivel_necessario));
}

auto generate_hann_window(int size) -> std::vector<double>
{
    std::vector<double> w(static_cast<size_t>(size));
    for (int i = 0; i < size; ++i)
    {
        constexpr double pi = std::numbers::pi;
        w[static_cast<size_t>(i)] =
            0.5 *
            (1.0 - std::cos(2.0 * pi * static_cast<double>(i) / static_cast<double>(size - 1)));
    }
    return w;
}

auto apply_windowing(const std::vector<double>& signal, const ExtractionConfig& cfg)
    -> std::vector<std::vector<double>>
{
    std::vector<std::vector<double>> windows;
    if (signal.size() < static_cast<size_t>(cfg.window_size))
    {
        return windows;
    }

    const auto window = generate_hann_window(cfg.window_size);
    for (int start = 0; start + cfg.window_size <= static_cast<int>(signal.size());
        start += cfg.hop_size)
    {
        std::vector<double> segment(static_cast<size_t>(cfg.window_size));
        for (int i = 0; i < cfg.window_size; ++i)
        {
            segment[static_cast<size_t>(i)] =
                signal[static_cast<size_t>(start + i)] * window[static_cast<size_t>(i)];
        }
        windows.push_back(std::move(segment));
    }
    return windows;
}

auto interpolate_to_size(const std::vector<double>& src, int destination_size) -> std::vector<float>
{
    if (destination_size <= 0)
    {
        return {};
    }

    if (static_cast<int>(src.size()) == destination_size)
    {
        return std::vector<float>(src.begin(), src.end());
    }

    std::vector<float> out(static_cast<size_t>(destination_size));

    for (int i = 0; i < destination_size; ++i)
    {
        double pos = (static_cast<double>(i) / static_cast<double>(destination_size - 1)) *
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

auto compute_wpt_energy(const std::vector<double>& window, int num_bands, int wpt_level)
    -> std::vector<float>
{
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const std::vector<double> haar{inv_sqrt2, inv_sqrt2};

    // Pad to next power of two if needed (mallat expects power-of-two length)
    int target_size = wavelets::get_next_power_of_two(static_cast<double>(window.size()));
    std::vector<double> padded_signal(window.begin(), window.end());
    padded_signal.resize(static_cast<size_t>(target_size), 0.0);

    auto transform = wavelets::malat(padded_signal,
        std::span<const double>(haar),
        wavelets::TransformMode::PACKET_WAVELET,
        static_cast<unsigned int>(wpt_level));
    auto energies = wavelets::extract_subband_energies(transform, wpt_level);
    return interpolate_to_size(energies, num_bands);
}

auto preprocess_energy(const std::vector<float>& energy) -> std::vector<float>
{
    std::vector<float> out(energy.size());
    float max_value = 0.0F;
    for (size_t i = 0; i < energy.size(); ++i)
    {
        float v = std::log1pf(std::max(0.0F, energy[i]));
        out[i] = v;
        max_value = std::max(max_value, v);
    }
    if (max_value < 1e-8F)
    {
        std::fill(out.begin(), out.end(), 0.0F);
        return out;
    }
    for (auto& v : out)
    {
        v = std::clamp(v / max_value, 0.0F, 1.0F);
    }
    return out;
}

auto build_features(const std::vector<double>& audio, ExtractionConfig cfg)
    -> std::vector<std::vector<float>>
{
    cfg.wpt_level = compute_wpt_level(cfg.window_size, cfg.num_bands);
    auto windows = apply_windowing(audio, cfg);
    std::vector<std::vector<float>> features;
    features.reserve(windows.size());
    for (const auto& window : windows)
    {
        auto energy = compute_wpt_energy(window, cfg.num_bands, cfg.wpt_level);
        features.push_back(preprocess_energy(energy));
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
    std::vector<Tensor*> param_ptrs_;

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

    auto params() -> std::span<Tensor*> override
    {
        param_ptrs_.clear();
        auto p = fc1->params();
        param_ptrs_.insert(param_ptrs_.end(), p.begin(), p.end());
        auto p2 = fc2->params();
        param_ptrs_.insert(param_ptrs_.end(), p2.begin(), p2.end());
        auto sg1 = lif1->params();
        param_ptrs_.insert(param_ptrs_.end(), sg1.begin(), sg1.end());
        auto sg2 = lif2->params();
        param_ptrs_.insert(param_ptrs_.end(), sg2.begin(), sg2.end());
        return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
    }

    void reset_state() override
    {
        lif1->reset_state();
        lif2->reset_state();
    }
};

struct SnnModel : public Module
{
    std::shared_ptr<Linear> fc_in;
    std::shared_ptr<Leaky> lif_in;
    std::vector<shared_ptr<ResidualBlock>> residual_blocks;
    std::shared_ptr<Linear> fc_out;
    std::shared_ptr<Leaky> lif_out;
    std::vector<Tensor*> param_ptrs_;

    SnnModel(int input_size, int output_size, int depth, int hidden_size, unsigned int seed)
    {
        fc_in = std::make_shared<Linear>(input_size, hidden_size);
        lif_in = std::make_shared<Leaky>();

        for (int i = 0; i < depth; ++i)
        {
            residual_blocks.push_back(std::make_shared<ResidualBlock>(hidden_size));
        }

        fc_out = std::make_shared<Linear>(hidden_size, output_size);
        lif_out = std::make_shared<Leaky>();

        const unsigned int base_seed = seed == 0 ? nn::testing::kSeed : seed;

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

    auto params() -> std::span<Tensor*> override
    {
        param_ptrs_.clear();
        auto p = fc_in->params();
        param_ptrs_.insert(param_ptrs_.end(), p.begin(), p.end());
        auto outp = fc_out->params();
        param_ptrs_.insert(param_ptrs_.end(), outp.begin(), outp.end());
        auto lifp = lif_in->params();
        param_ptrs_.insert(param_ptrs_.end(), lifp.begin(), lifp.end());
        auto lifout = lif_out->params();
        param_ptrs_.insert(param_ptrs_.end(), lifout.begin(), lifout.end());
        for (auto& b : residual_blocks)
        {
            auto bp = b->params();
            param_ptrs_.insert(param_ptrs_.end(), bp.begin(), bp.end());
        }
        return std::span<Tensor*>{param_ptrs_.data(), param_ptrs_.size()};
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

auto load_audio_or_synthetic(const string& audio_path, double duration, int sample_rate)
    -> vector<double>
{
    if (!audio_path.empty())
    {
        if (!std::filesystem::exists(audio_path) || !std::filesystem::is_regular_file(audio_path))
        {
            throw runtime_error("Arquivo de audio invalido: " + audio_path);
        }

        Wav w;
        w.read(audio_path);

        auto data = w.get_data();
        if (data.empty())
        {
            throw runtime_error("Arquivo de audio vazio ou invalido: " + audio_path);
        }

        return data;
    }

    const size_t total = static_cast<size_t>(llround(duration * static_cast<double>(sample_rate)));
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

auto save_spikes_csv(const string& output_path, const vector<Tensor>& spikes) -> void
{
    ofstream out(output_path);
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

auto run_pipeline(                          //
    const string& wav_path,                 //
    double synthetic_duration,              //
    const ExtractionConfig& extraction_cfg, //
    const SnnConfig& snn_cfg,               //
    unsigned int seed,                      //
    const string& output_csv                //
    ) -> void
{
    auto audio = load_audio_or_synthetic( //
        wav_path,                         //
        synthetic_duration,               //
        extraction_cfg.sample_rate        //
    );

    auto features = build_features(audio, extraction_cfg);

    if (features.empty())
    {
        throw runtime_error("Nenhuma janela gerada.");
    }

    SnnModel model(                                //
        static_cast<int>(features.front().size()), //
        static_cast<int>(features.front().size()), //
        snn_cfg.depth,                             //
        snn_cfg.hidden_size,                       //
        seed                                       //
    );

    mt19937 rng(seed == 0 ? nn::testing::kSeed : seed);

    vector<Tensor> output_per_window;
    output_per_window.reserve(features.size());

    for (const auto& feat : features)
    {
        Tensor input_tensor(1, static_cast<Index>(feat.size()));

        for (Index j = 0; j < input_tensor.cols(); ++j)
        {
            input_tensor.at(0, j) = feat[static_cast<size_t>(j)];
        }

        auto spikes_in = codificacao::encode_poisson( //
            input_tensor,                             //
            snn_cfg.steps_per_window,                 //
            rng,                                      //
            -1.0F,                                    //
            true,                                     //
            snn_cfg.target_spikes_per_step            //
        );

        Tensor accumulated(1, static_cast<Index>(feat.size()));
        accumulated.setZero();
        model.reset_state();

        for (Index t = 0; t < spikes_in.rows(); ++t)
        {
            auto step = spikes_in.row(t);
            auto spk_out = model.forward(step, false);
            accumulated = accumulated.add(spk_out);
        }

        output_per_window.push_back(accumulated);
    }

    save_spikes_csv(output_csv, output_per_window);
    cout << "Pipeline concluido. Saida em " << output_csv << "\n";
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

    ExtractionConfig extraction_cfg;
    extraction_cfg.sample_rate = parser.get<int>("--taxa-amostragem");
    extraction_cfg.window_size = parser.get<int>("--tamanho-janela");
    extraction_cfg.hop_size = parser.get<int>("--tamanho-passo");
    extraction_cfg.num_bands = parser.get<int>("--num-bandas");

    SnnConfig snn_cfg;
    snn_cfg.steps_per_window = parser.get<int>("--passos-por-janela");
    snn_cfg.depth = parser.get<int>("--profundidade");
    snn_cfg.hidden_size = parser.get<int>("--hidden");

    auto seed = parser.get<unsigned int>("--seed");
    auto input_wav = parser.get<std::string>("--entrada-wav");
    auto output_csv = parser.get<std::string>("--saida-csv");
    auto duration = parser.get<double>("--duracao");

    try
    {
        run_pipeline(input_wav, duration, extraction_cfg, snn_cfg, seed, output_csv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Erro: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
