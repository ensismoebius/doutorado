"""
CLI de alto nível do demo de biometria por voz.
Este arquivo mantém apenas a interface de linha de comando e o
entrypoint, delegando a implementação dos comandos para o módulo
`comandos`.
"""

import argparse

from app.comandos import (
    cmd_demo,
    cmd_capturar,
    cmd_treinar,
    cmd_identificar,
    cmd_verificar,
    cmd_avaliar,
)


def construir_cli() -> argparse.ArgumentParser:
    # Define o parser raiz e as subcomandos.
    p = argparse.ArgumentParser(
        description=(
            "Biometria por voz (demo): WPT -> codificação em spikes -> SNN -> classificação por pessoa.\n"
            "Fluxo recomendado: capturar dados -> treinar -> identificar."
        )
    )

    # Cada subcomando chama uma função `cmd_*` definida em `comandos`.
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- Demo visual ---
    demo = sub.add_parser("demo", help="Roda a pipeline e gera plots didáticos")
    demo.add_argument("--duracao", type=float, default=1.0)
    demo.add_argument("--taxa-amostragem", type=int, default=44100)
    demo.add_argument("--tamanho-janela", type=int, default=512)
    demo.add_argument("--tamanho-passo", type=int, default=256)
    demo.add_argument("--wavelet", type=str, default="db4")
    demo.add_argument("--num-bandas", type=int, default=100)
    demo.add_argument("--passos-por-janela", type=int, default=10)
    demo.add_argument(
        "--profundidade",
        type=int,
        default=None,
        help="Número de blocos residuais na SNN (override)",
    )
    demo.add_argument(
        "--tamanho-camada-oculta",
        type=int,
        default=100,
        help="Tamanho da camada oculta (hidden size) da SNN",
    )
    demo.add_argument("--saida-plot", type=str, default="result_pipeline_wpt_snn.png")

    # --- Captura/cadastro ---
    cap = sub.add_parser(
        "capturar", help="Captura áudio e salva WAV para uma pessoa (cadastro)"
    )
    cap.add_argument(
        "--pessoa", type=str, required=True, help="ID da pessoa (ex.: alice)"
    )
    cap.add_argument("--diretorio-dados", type=str, default="dados/vozes")
    cap.add_argument("--duracao", type=float, default=3.0)
    cap.add_argument("--taxa-amostragem", type=int, default=44100)

    # --- Treino ---
    tr = sub.add_parser("treinar", help="Treina a SNN para classificar locutores")
    tr.add_argument("--diretorio-dados", type=str, default="dados/vozes")
    tr.add_argument("--taxa-amostragem", type=int, default=44100)
    tr.add_argument("--tamanho-janela", type=int, default=512)
    tr.add_argument("--tamanho-passo", type=int, default=256)
    tr.add_argument("--wavelet", type=str, default="db4")
    tr.add_argument("--num-bandas", type=int, default=100)
    tr.add_argument("--duracao-referencia", type=float, default=1.0)
    tr.add_argument("--passos-por-janela", type=int, default=10)
    tr.add_argument(
        "--profundidade",
        type=int,
        default=None,
        help="Número de blocos residuais na SNN",
    )
    tr.add_argument("--alvo-spikes-por-passo", type=float, default=0.10)
    tr.add_argument(
        "--tamanho-camada-oculta",
        type=int,
        default=100,
        help="Tamanho da camada oculta (hidden size) da SNN",
    )
    tr.add_argument("--epocas", type=int, default=5)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--saida-modelo", type=str, default="modelo_snn_locutor.pt")
    tr.add_argument("--saida-rotulos", type=str, default="rotulos_locutor.json")

    # --- Identificação ---
    inf = sub.add_parser("identificar", help="Identifica a pessoa por voz (microfone)")
    inf.add_argument("--modelo", type=str, default="modelo_snn_locutor.pt")
    inf.add_argument("--rotulos", type=str, default="rotulos_locutor.json")
    inf.add_argument("--duracao", type=float, default=2.0)
    inf.add_argument("--taxa-amostragem", type=int, default=44100)
    inf.add_argument("--tamanho-janela", type=int, default=512)
    inf.add_argument("--tamanho-passo", type=int, default=256)
    inf.add_argument("--wavelet", type=str, default="db4")
    inf.add_argument("--num-bandas", type=int, default=100)
    inf.add_argument("--duracao-referencia", type=float, default=1.0)
    inf.add_argument("--passos-por-janela", type=int, default=10)
    inf.add_argument(
        "--profundidade",
        type=int,
        default=None,
        help="Número de blocos residuais na SNN (override ao carregar modelo)",
    )
    inf.add_argument("--alvo-spikes-por-passo", type=float, default=0.10)
    inf.add_argument(
        "--tamanho-camada-oculta",
        type=int,
        default=100,
        help="Tamanho da camada oculta (hidden size) da SNN",
    )

    # --- Verificação (com desconhecido) ---
    ver = sub.add_parser(
        "verificar",
        help=(
            "Verifica a identidade e pode retornar 'desconhecido' se a confiança for baixa"
        ),
    )
    ver.add_argument("--modelo", type=str, default="modelo_snn_locutor.pt")
    ver.add_argument("--rotulos", type=str, default="rotulos_locutor.json")
    ver.add_argument("--duracao", type=float, default=2.0)
    ver.add_argument("--taxa-amostragem", type=int, default=44100)
    ver.add_argument("--tamanho-janela", type=int, default=512)
    ver.add_argument("--tamanho-passo", type=int, default=256)
    ver.add_argument("--wavelet", type=str, default="db4")
    ver.add_argument("--num-bandas", type=int, default=100)
    ver.add_argument("--duracao-referencia", type=float, default=1.0)
    ver.add_argument("--passos-por-janela", type=int, default=10)
    ver.add_argument(
        "--profundidade",
        type=int,
        default=None,
        help="Número de blocos residuais na SNN (override ao carregar modelo)",
    )
    ver.add_argument("--alvo-spikes-por-passo", type=float, default=0.10)
    ver.add_argument(
        "--tamanho-camada-oculta",
        type=int,
        default=100,
        help="Tamanho da camada oculta (hidden size) da SNN",
    )
    ver.add_argument(
        "--limiar",
        type=float,
        default=0.55,
        help="Abaixo deste valor, retorna 'desconhecido'",
    )

    # --- Avaliação offline ---
    av = sub.add_parser(
        "avaliar", help="Avalia o modelo em WAVs gravados e imprime matriz de confusão"
    )
    av.add_argument("--modelo", type=str, default="modelo_snn_locutor.pt")
    av.add_argument("--rotulos", type=str, default="rotulos_locutor.json")
    av.add_argument("--diretorio-dados", type=str, default="dados/vozes")
    av.add_argument("--taxa-amostragem", type=int, default=44100)
    av.add_argument("--tamanho-janela", type=int, default=512)
    av.add_argument("--tamanho-passo", type=int, default=256)
    av.add_argument("--wavelet", type=str, default="db4")
    av.add_argument("--num-bandas", type=int, default=100)
    av.add_argument("--duracao-referencia", type=float, default=1.0)
    av.add_argument("--passos-por-janela", type=int, default=10)
    av.add_argument(
        "--profundidade",
        type=int,
        default=None,
        help="Número de blocos residuais na SNN",
    )
    av.add_argument("--alvo-spikes-por-passo", type=float, default=0.10)
    av.add_argument(
        "--tamanho-camada-oculta",
        type=int,
        default=100,
        help="Tamanho da camada oculta (hidden size) da SNN",
    )
    av.add_argument("--verbose", action="store_true")

    demo.set_defaults(func=cmd_demo)
    tr.set_defaults(func=cmd_treinar)
    av.set_defaults(func=cmd_avaliar)
    cap.set_defaults(func=cmd_capturar)
    ver.set_defaults(func=cmd_verificar)
    inf.set_defaults(func=cmd_identificar)

    return p


if __name__ == "__main__":
    # Ponto de entrada: parseia argumentos e executa o comando selecionado.
    p = construir_cli()
    args = p.parse_args()
    args.func(args)
