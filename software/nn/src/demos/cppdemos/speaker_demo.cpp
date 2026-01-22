// Single, consistent CLI implementation using RAII to own subparsers.

#include <argparse/argparse.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

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

/**
 * Avoid dangling pointers by owning the root parser and all
 * subparsers.(heap-use-after-free error)
 */
struct ParserPointersOwner
{
    std::unique_ptr<ArgumentParser> parser;
    std::vector<std::unique_ptr<ArgumentParser>> subParsers;
};

/**
 * @brief Construct the CLI parser with all subcommands and options.
 *
 * @return std::unique_ptr<ParserPointersOwner>
 */
std::unique_ptr<ParserPointersOwner> construir_cli()
{
    auto pointerOwner = std::make_unique<ParserPointersOwner>();

    pointerOwner->parser = std::make_unique<ArgumentParser>("biometria-voz");
    pointerOwner->parser->add_description(
        "Biometria por voz (demo): WPT -> codificacao em spikes -> SNN -> "
        "classificacao por pessoa. Fluxo recomendado: capturar -> treinar -> "
        "identificar.");

    pointerOwner->subParsers.emplace_back( //
        std::make_unique<ArgumentParser>(  //
            "demo",
            "Demonstracao do pipeline completo."));
    pointerOwner->subParsers.emplace_back( //
        std::make_unique<ArgumentParser>(  //
            "capturar",
            "Captura amostras de voz para uma pessoa."));
    pointerOwner->subParsers.emplace_back( //
        std::make_unique<ArgumentParser>(  //
            "treinar",
            "Treina um modelo SNN para reconhecimento de locutor."));
    pointerOwner->subParsers.emplace_back( //
        std::make_unique<ArgumentParser>(  //
            "identificar",
            "Identifica um locutor a partir de uma amostra de voz."));
    pointerOwner->subParsers.emplace_back( //
        std::make_unique<ArgumentParser>(  //
            "verificar",
            "Verifica se uma amostra de voz pertence a um locutor especifico."));
    pointerOwner->subParsers.emplace_back( //
        std::make_unique<ArgumentParser>(  //
            "avaliar",
            "Avalia o desempenho do modelo em um conjunto de dados."));

    // demo
    auto* demo = pointerOwner->subParsers[0].get();
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
    auto* capturar = pointerOwner->subParsers[1].get();
    capturar->add_argument("--pessoa").required();
    capturar->add_argument("--diretorio-dados").default_value(std::string("dados/vozes"));
    capturar->add_argument("--duracao").default_value(3.0);
    capturar->add_argument("--taxa-amostragem").default_value(44100);

    // treinar
    auto* treinar = pointerOwner->subParsers[2].get();
    treinar->add_argument("--diretorio-dados").default_value(std::string("dados/vozes"));
    treinar->add_argument("--taxa-amostragem").default_value(44100);
    treinar->add_argument("--tamanho-janela").default_value(512);
    treinar->add_argument("--tamanho-passo").default_value(256);
    treinar->add_argument("--wavelet").default_value(std::string("db4"));
    treinar->add_argument("--num-bandas").default_value(100);
    treinar->add_argument("--duracao-referencia").default_value(1.0);
    treinar->add_argument("--passos-por-janela").default_value(10);
    treinar->add_argument("--profundidade").default_value(-1);
    treinar->add_argument("--alvo-spikes-por-passo").default_value(0.10);
    treinar->add_argument("--epocas").default_value(5);
    treinar->add_argument("--lr").default_value(1e-3);
    treinar->add_argument("--saida-modelo").default_value(std::string("modelo_snn_locutor.pt"));
    treinar->add_argument("--saida-rotulos").default_value(std::string("rotulos_locutor.json"));

    // identificar
    auto* identificar = pointerOwner->subParsers[3].get();
    identificar->add_argument("--modelo").default_value(std::string("modelo_snn_locutor.pt"));
    identificar->add_argument("--rotulos").default_value(std::string("rotulos_locutor.json"));
    identificar->add_argument("--duracao").default_value(2.0);
    identificar->add_argument("--taxa-amostragem").default_value(44100);
    identificar->add_argument("--tamanho-janela").default_value(512);
    identificar->add_argument("--tamanho-passo").default_value(256);
    identificar->add_argument("--wavelet").default_value(std::string("db4"));
    identificar->add_argument("--num-bandas").default_value(100);
    identificar->add_argument("--duracao-referencia").default_value(1.0);
    identificar->add_argument("--passos-por-janela").default_value(10);
    identificar->add_argument("--profundidade").default_value(-1);
    identificar->add_argument("--alvo-spikes-por-passo").default_value(0.10);

    // verificar
    auto* verificar = pointerOwner->subParsers[4].get();
    verificar->add_argument("--modelo").default_value(std::string("modelo_snn_locutor.pt"));
    verificar->add_argument("--rotulos").default_value(std::string("rotulos_locutor.json"));
    verificar->add_argument("--duracao").default_value(2.0);
    verificar->add_argument("--taxa-amostragem").default_value(44100);
    verificar->add_argument("--tamanho-janela").default_value(512);
    verificar->add_argument("--tamanho-passo").default_value(256);
    verificar->add_argument("--wavelet").default_value(std::string("db4"));
    verificar->add_argument("--num-bandas").default_value(100);
    verificar->add_argument("--duracao-referencia").default_value(1.0);
    verificar->add_argument("--passos-por-janela").default_value(10);
    verificar->add_argument("--profundidade").default_value(-1);
    verificar->add_argument("--alvo-spikes-por-passo").default_value(0.10);
    verificar->add_argument("--limiar").default_value(0.55);

    // avaliar
    auto* avaliar = pointerOwner->subParsers[5].get();
    avaliar->add_argument("--modelo").default_value(std::string("modelo_snn_locutor.pt"));
    avaliar->add_argument("--rotulos").default_value(std::string("rotulos_locutor.json"));
    avaliar->add_argument("--diretorio-dados").default_value(std::string("dados/vozes"));
    avaliar->add_argument("--taxa-amostragem").default_value(44100);
    avaliar->add_argument("--tamanho-janela").default_value(512);
    avaliar->add_argument("--tamanho-passo").default_value(256);
    avaliar->add_argument("--wavelet").default_value(std::string("db4"));
    avaliar->add_argument("--num-bandas").default_value(100);
    avaliar->add_argument("--duracao-referencia").default_value(1.0);
    avaliar->add_argument("--passos-por-janela").default_value(10);
    avaliar->add_argument("--profundidade").default_value(-1);
    avaliar->add_argument("--alvo-spikes-por-passo").default_value(0.10);
    avaliar->add_argument("--verbose").flag();

    for (auto& s : pointerOwner->subParsers)
    {
        pointerOwner->parser->add_subparser(*s);
    }

    return pointerOwner;
}

int main(int argc, char** argv)
{
    auto pointerOwner = construir_cli();

    try
    {
        // parse_args uses references into the subparsers owned by `pointerOwner`,
        // so `pointerOwner` must remain alive until parse_args returns.
        pointerOwner->parser->parse_args(argc, argv);
    }
    catch (const std::exception& err)
    {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    auto& program = *pointerOwner->parser;

    if (program.is_subcommand_used("demo"))
    {
        double dur = arg_to<double>(program.at<ArgumentParser>("demo"), "--duracao");
        std::cout << "demo: duracao=" << dur << std::endl;
        return 0;
    }

    if (program.is_subcommand_used("capturar"))
    {
        std::string pessoa = arg_to<string>(program.at<ArgumentParser>("capturar"), "--pessoa");
        std::string dir =
            arg_to<string>(program.at<ArgumentParser>("capturar"), "--diretorio-dados");
        double dur = arg_to<double>(program.at<ArgumentParser>("capturar"), "--duracao");
        std::cout << "capturar: pessoa=" << pessoa << " dir=" << dir << " duracao=" << dur
                  << std::endl;
        return 0;
    }

    if (program.is_subcommand_used("treinar"))
    {
        auto& treinar_parser = program.at<ArgumentParser>("treinar");
        std::string dir = arg_to<std::string>(treinar_parser, "--diretorio-dados");
        int epocas = static_cast<int>(arg_to<int>(treinar_parser, "--epocas"));
        std::cout << "treinar: dados=" << dir << " epocas=" << epocas << std::endl;
        return 0;
    }

    if (program.is_subcommand_used("identificar"))
    {
        auto& identificar_parser = program.at<ArgumentParser>("identificar");
        std::string model = arg_to<std::string>(identificar_parser, "--modelo");
        std::cout << "identificar: modelo=" << model << std::endl;
        return 0;
    }

    if (program.is_subcommand_used("verificar"))
    {
        auto& verificar_parser = program.at<ArgumentParser>("verificar");
        std::string model = arg_to<std::string>(verificar_parser, "--modelo");
        double limiar = arg_to<double>(verificar_parser, "--limiar");
        std::cout << "verificar: modelo=" << model << " limiar=" << limiar << std::endl;
        return 0;
    }

    if (program.is_subcommand_used("avaliar"))
    {
        auto& avaliar_parser = program.at<ArgumentParser>("avaliar");
        bool verb = arg_to<bool>(avaliar_parser, "--verbose");
        std::cout << "avaliar: verbose=" << (verb ? "true" : "false") << std::endl;
        return 0;
    }

    std::cerr << "No command provided. Use --help." << std::endl;
    return 1;
}
