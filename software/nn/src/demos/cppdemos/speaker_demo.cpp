/**
 * @file src/demos/cppdemos/speaker_demo.cpp
 * @brief Implementation for Speaker demo.
 *

 */
// cppcheck-suppress-file useStlAlgorithm
// Single, consistent CLI implementation using RAII to own subparsers.

// cppcheck-suppress useStlAlgorithm
#include <argparse/argparse.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "nn/logging/Logger.hpp"

using argparse::ArgumentParser;
using std::string;

template <typename T>
T arg_to(argparse::ArgumentParser& p, const std::string& name)
{
    try
    {
        return p.get<T>(name);
    }
    catch (const std::bad_any_cast&)
    {
        std::string s = p.get<std::string>(name);

        if constexpr (std::is_same_v<T, double>)
            return std::stod(s);
        else if constexpr (std::is_same_v<T, int>)
            return std::stoi(s);
        else if constexpr (std::is_same_v<T, bool>)
            return (s == "1" || s == "true" || s == "True");
        else if constexpr (std::is_same_v<T, std::string>)
            return s;
        else
            throw;
    }
}

// Forward declaration for the demo implementation in comandos.cpp
namespace demo
{
void cmd_demo(double duration,
    int sample_rate,
    int window_size,
    int hop_size,
    const std::string& wavelet,
    int num_bands,
    int steps_per_window,
    int depth,
    const std::string& plot_output,
    unsigned int random_seed = 0);
}

/**
 * Avoid dangling pointers by owning the root parser and all
 * subparsers.(heap-use-after-free error)
 */
struct ParserPointersOwner
{
    std::unique_ptr<ArgumentParser> parser;
    std::vector<std::unique_ptr<ArgumentParser>> subparsers;
};

/**
 * @brief Construct the CLI parser with all subcommands and options.
 *
 * @return std::unique_ptr<ParserPointersOwner>
 */
std::unique_ptr<ParserPointersOwner> build_cli()
{
    auto parser_owner = std::make_unique<ParserPointersOwner>();

    parser_owner->parser = std::make_unique<ArgumentParser>("biometria-voz");
    parser_owner->parser->add_description(
        "Biometria por voz (demo): WPT -> codificacao em spikes -> SNN -> "
        "classificacao por pessoa. Fluxo recomendado: capturar -> treinar -> "
        "identificar.");

    parser_owner->subparsers.emplace_back( //
        std::make_unique<ArgumentParser>(  //
            "demo",
            "Demonstracao do pipeline completo."));
    parser_owner->subparsers.emplace_back( //
        std::make_unique<ArgumentParser>(  //
            "capturar",
            "Captura amostras de voz para uma pessoa."));
    parser_owner->subparsers.emplace_back( //
        std::make_unique<ArgumentParser>(  //
            "treinar",
            "Treina um modelo SNN para reconhecimento de locutor."));
    parser_owner->subparsers.emplace_back( //
        std::make_unique<ArgumentParser>(  //
            "identificar",
            "Identifica um locutor a partir de uma amostra de voz."));
    parser_owner->subparsers.emplace_back( //
        std::make_unique<ArgumentParser>(  //
            "verificar",
            "Verifica se uma amostra de voz pertence a um locutor especifico."));
    parser_owner->subparsers.emplace_back( //
        std::make_unique<ArgumentParser>(  //
            "avaliar",
            "Avalia o desempenho do modelo em um conjunto de dados."));

    // demo
    auto* demo = parser_owner->subparsers[0].get();
    demo->add_argument("--duracao").default_value(1.0);
    demo->add_argument("--taxa-amostragem").default_value(44100);
    demo->add_argument("--tamanho-janela").default_value(512);
    demo->add_argument("--tamanho-passo").default_value(256);
    demo->add_argument("--wavelet").default_value(std::string("db4"));
    demo->add_argument("--num-bandas").default_value(100);
    demo->add_argument("--passos-por-janela").default_value(10);
    demo->add_argument("--profundidade").default_value(-1);
    demo->add_argument("--saida-plot").default_value(std::string("result_pipeline_wpt_snn.png"));

    // capturar
    auto* capture = parser_owner->subparsers[1].get();
    capture->add_argument("--pessoa").required();
    capture->add_argument("--diretorio-dados").default_value(std::string("dados/vozes"));
    capture->add_argument("--duracao").default_value(3.0);
    capture->add_argument("--taxa-amostragem").default_value(44100);

    // treinar
    auto* train = parser_owner->subparsers[2].get();
    train->add_argument("--diretorio-dados").default_value(std::string("dados/vozes"));
    train->add_argument("--taxa-amostragem").default_value(44100);
    train->add_argument("--tamanho-janela").default_value(512);
    train->add_argument("--tamanho-passo").default_value(256);
    train->add_argument("--wavelet").default_value(std::string("db4"));
    train->add_argument("--num-bandas").default_value(100);
    train->add_argument("--duracao-referencia").default_value(1.0);
    train->add_argument("--passos-por-janela").default_value(10);
    train->add_argument("--profundidade").default_value(-1);
    train->add_argument("--alvo-spikes-por-passo").default_value(0.10);
    train->add_argument("--epocas").default_value(5);
    train->add_argument("--learning-rate").default_value(1e-3);
    train->add_argument("--saida-modelo").default_value(std::string("modelo_snn_locutor.pt"));
    train->add_argument("--saida-rotulos").default_value(std::string("rotulos_locutor.json"));

    // identificar
    auto* identify = parser_owner->subparsers[3].get();
    identify->add_argument("--modelo").default_value(std::string("modelo_snn_locutor.pt"));
    identify->add_argument("--rotulos").default_value(std::string("rotulos_locutor.json"));
    identify->add_argument("--duracao").default_value(2.0);
    identify->add_argument("--taxa-amostragem").default_value(44100);
    identify->add_argument("--tamanho-janela").default_value(512);
    identify->add_argument("--tamanho-passo").default_value(256);
    identify->add_argument("--wavelet").default_value(std::string("db4"));
    identify->add_argument("--num-bandas").default_value(100);
    identify->add_argument("--duracao-referencia").default_value(1.0);
    identify->add_argument("--passos-por-janela").default_value(10);
    identify->add_argument("--profundidade").default_value(-1);
    identify->add_argument("--alvo-spikes-por-passo").default_value(0.10);

    // verificar
    auto* verify = parser_owner->subparsers[4].get();
    verify->add_argument("--modelo").default_value(std::string("modelo_snn_locutor.pt"));
    verify->add_argument("--rotulos").default_value(std::string("rotulos_locutor.json"));
    verify->add_argument("--duracao").default_value(2.0);
    verify->add_argument("--taxa-amostragem").default_value(44100);
    verify->add_argument("--tamanho-janela").default_value(512);
    verify->add_argument("--tamanho-passo").default_value(256);
    verify->add_argument("--wavelet").default_value(std::string("db4"));
    verify->add_argument("--num-bandas").default_value(100);
    verify->add_argument("--duracao-referencia").default_value(1.0);
    verify->add_argument("--passos-por-janela").default_value(10);
    verify->add_argument("--profundidade").default_value(-1);
    verify->add_argument("--alvo-spikes-por-passo").default_value(0.10);
    verify->add_argument("--limiar").default_value(0.55);

    // avaliar
    auto* evaluate = parser_owner->subparsers[5].get();
    evaluate->add_argument("--modelo").default_value(std::string("modelo_snn_locutor.pt"));
    evaluate->add_argument("--rotulos").default_value(std::string("rotulos_locutor.json"));
    evaluate->add_argument("--diretorio-dados").default_value(std::string("dados/vozes"));
    evaluate->add_argument("--taxa-amostragem").default_value(44100);
    evaluate->add_argument("--tamanho-janela").default_value(512);
    evaluate->add_argument("--tamanho-passo").default_value(256);
    evaluate->add_argument("--wavelet").default_value(std::string("db4"));
    evaluate->add_argument("--num-bandas").default_value(100);
    evaluate->add_argument("--duracao-referencia").default_value(1.0);
    evaluate->add_argument("--passos-por-janela").default_value(10);
    evaluate->add_argument("--profundidade").default_value(-1);
    evaluate->add_argument("--alvo-spikes-por-passo").default_value(0.10);
    evaluate->add_argument("--verbose").flag();

    for (auto& s : parser_owner->subparsers)
    {
        parser_owner->parser->add_subparser(*s);
    }

    return parser_owner;
}

int main(int argc, char** argv)
{
    try
    {
        auto parser_owner = build_cli();

        // parse_args uses references into the subparsers owned by `pointerOwner`,
        // so `pointerOwner` must remain alive until parse_args returns.
        try
        {
            // cppcheck-suppress useStlAlgorithm
            parser_owner->parser->parse_args(argc, argv);
        }
        catch (const std::exception& err)
        {
            NN_LOG_ERROR(std::string("Argument parse error: ") + err.what());
            return 1;
        }

        auto& program = *parser_owner->parser;

        if (program.is_subcommand_used("demo"))
        {
            auto& demo_parser = program.at<ArgumentParser>("demo");
            double duration = arg_to<double>(demo_parser, "--duracao");
            int sample_rate = arg_to<int>(demo_parser, "--taxa-amostragem");
            int window_size = arg_to<int>(demo_parser, "--tamanho-janela");
            int hop_size = arg_to<int>(demo_parser, "--tamanho-passo");
            std::string wavelet = arg_to<std::string>(demo_parser, "--wavelet");
            int num_bands = arg_to<int>(demo_parser, "--num-bandas");
            int steps_per_window = arg_to<int>(demo_parser, "--passos-por-janela");
            int depth = arg_to<int>(demo_parser, "--profundidade");
            std::string plot_output = arg_to<std::string>(demo_parser, "--saida-plot");
            try
            {
                demo::cmd_demo(duration,
                    sample_rate,
                    window_size,
                    hop_size,
                    wavelet,
                    num_bands,
                    steps_per_window,
                    depth,
                    plot_output,
                    0u);
                return 0;
            }
            catch (const std::exception& e)
            {
                NN_LOG_ERROR(std::string("Demo failed: ") + e.what());
                return 2;
            }
        }

        if (program.is_subcommand_used("capturar"))
        {
            std::string person = arg_to<string>(program.at<ArgumentParser>("capturar"), "--pessoa");
            std::string data_dir =
                arg_to<string>(program.at<ArgumentParser>("capturar"), "--diretorio-dados");
            double duration = arg_to<double>(program.at<ArgumentParser>("capturar"), "--duracao");
            NN_LOG_INFO("capturar: pessoa=" + person + " dir=" + data_dir +
                        " duracao=" + std::to_string(duration));
            return 0;
        }

        if (program.is_subcommand_used("treinar"))
        {
            auto& train_parser = program.at<ArgumentParser>("treinar");
            std::string data_dir = arg_to<std::string>(train_parser, "--diretorio-dados");
            int epochs = static_cast<int>(arg_to<int>(train_parser, "--epocas"));
            NN_LOG_INFO("treinar: dados=" + data_dir + " epocas=" + std::to_string(epochs));
            return 0;
        }

        if (program.is_subcommand_used("identificar"))
        {
            auto& identify_parser = program.at<ArgumentParser>("identificar");
            std::string model_path = arg_to<std::string>(identify_parser, "--modelo");
            NN_LOG_INFO("identificar: modelo=" + model_path);
            return 0;
        }

        if (program.is_subcommand_used("verificar"))
        {
            auto& verify_parser = program.at<ArgumentParser>("verificar");
            std::string model_path = arg_to<std::string>(verify_parser, "--modelo");
            double threshold = arg_to<double>(verify_parser, "--limiar");
            NN_LOG_INFO("verificar: modelo=" + model_path + " limiar=" + std::to_string(threshold));
            return 0;
        }

        if (program.is_subcommand_used("avaliar"))
        {
            auto& evaluate_parser = program.at<ArgumentParser>("avaliar");
            bool verbose = arg_to<bool>(evaluate_parser, "--verbose");
            NN_LOG_INFO("avaliar: verbose=" + std::string(verbose ? "true" : "false"));
            return 0;
        }

        std::cout << program;
        return 0;
    }
    catch (const std::exception& e)
    {
        NN_LOG_ERROR(std::string("Unhandled exception: ") + e.what());
        return 1;
    }
}
