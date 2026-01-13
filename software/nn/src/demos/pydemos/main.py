import argparse

import numpy as np
import torch

from captura import capturar_audio
from conjunto_dados import ConfigExtracao, extrair_janelas_caracteristicas, listar_amostras_por_pessoa
from codificacao import codificar_poisson
from cadastro import capturar_e_salvar_amostra
from rede_snn import criar_modelo_snn
from identificacao_locutor import (
    ConfigSNN,
    aplicar_limiar_desconhecido,
    carregar_modelo_e_rotulos,
    identificar_locutor_por_microfone,
    identificar_locutor_por_wav,
    salvar_modelo_e_rotulos,
    treinar_classificador_locutor,
)
from visualizacao import plotar_resultados


def cmd_demo(args: argparse.Namespace) -> None:
    """Demo visual: pipeline WPT -> preprocess -> codificação Poisson -> SNN -> plots."""

    cfg_extracao = ConfigExtracao(
        taxa_amostragem=args.taxa_amostragem,
        tamanho_janela=args.tamanho_janela,
        tamanho_passo=args.tamanho_passo,
        wavelet_base=args.wavelet,
        num_bandas=args.num_bandas,
        duracao_referencia=args.duracao,
    )
    cfg_snn = ConfigSNN(passos_por_janela=args.passos_por_janela)

    audio = capturar_audio(args.duracao, args.taxa_amostragem)
    caracs = extrair_janelas_caracteristicas(audio, cfg=cfg_extracao)
    if not caracs:
        raise RuntimeError("Nenhuma janela gerada.")

    # Modelo com saída do mesmo tamanho das entradas (para manter plots didáticos)
    model = criar_modelo_snn(num_inputs=cfg_extracao.num_bandas, num_outputs=cfg_extracao.num_bandas)
    model.eval()

    lista_spikes_saida = []
    with torch.no_grad():
        state = None
        for c in caracs:
            xb = torch.tensor(c, dtype=torch.float32).unsqueeze(0)
            spk_in = codificar_poisson(xb, passos=cfg_snn.passos_por_janela, adaptativo=True)
            spk_out_seq, state = model(spk_in, state)

            # Para plotar por janela, agregamos os spikes ao longo dos passos.
            spk_janela = spk_out_seq.sum(dim=0)  # [1, num_bandas]
            lista_spikes_saida.append(spk_janela)

    plotar_resultados(
        caracs,
        lista_spikes_saida,
        sample_rate=cfg_extracao.taxa_amostragem,
        window_size=cfg_extracao.tamanho_janela,
        hop_size=cfg_extracao.tamanho_passo,
        duration=args.duracao,
        wavelet=cfg_extracao.wavelet_base,
        wpt_level=None,
        num_bands=cfg_extracao.num_bandas,
        stateful=True,
        output_file=args.saida_plot,
    )


def cmd_capturar(args: argparse.Namespace) -> None:
    capturar_e_salvar_amostra(
        pessoa=args.pessoa,
        diretorio_base=args.diretorio_dados,
        duracao=args.duracao,
        taxa_amostragem=args.taxa_amostragem,
    )


def cmd_treinar(args: argparse.Namespace) -> None:
    cfg_extracao = ConfigExtracao(
        taxa_amostragem=args.taxa_amostragem,
        tamanho_janela=args.tamanho_janela,
        tamanho_passo=args.tamanho_passo,
        wavelet_base=args.wavelet,
        num_bandas=args.num_bandas,
        duracao_referencia=args.duracao_referencia,
    )
    cfg_snn = ConfigSNN(
        passos_por_janela=args.passos_por_janela,
        alvo_spikes_por_passo=args.alvo_spikes_por_passo,
    )

    model, rotulos = treinar_classificador_locutor(
        args.diretorio_dados,
        cfg_extracao=cfg_extracao,
        cfg_snn=cfg_snn,
        epocas=args.epocas,
        taxa_aprendizado=args.lr,
    )
    salvar_modelo_e_rotulos(
        model,
        rotulos,
        caminho_modelo=args.saida_modelo,
        caminho_rotulos=args.saida_rotulos,
    )
    print(f"[Treino] Modelo salvo em: {args.saida_modelo}")
    print(f"[Treino] Rótulos salvos em: {args.saida_rotulos}")


def cmd_identificar(args: argparse.Namespace) -> None:
    cfg_extracao = ConfigExtracao(
        taxa_amostragem=args.taxa_amostragem,
        tamanho_janela=args.tamanho_janela,
        tamanho_passo=args.tamanho_passo,
        wavelet_base=args.wavelet,
        num_bandas=args.num_bandas,
        duracao_referencia=args.duracao_referencia,
    )
    cfg_snn = ConfigSNN(
        passos_por_janela=args.passos_por_janela,
        alvo_spikes_por_passo=args.alvo_spikes_por_passo,
    )

    model, rotulos = carregar_modelo_e_rotulos(
        caminho_modelo=args.modelo,
        caminho_rotulos=args.rotulos,
        num_inputs=cfg_extracao.num_bandas,
    )

    pessoa, conf, _ = identificar_locutor_por_microfone(
        model,
        rotulos,
        cfg_extracao=cfg_extracao,
        cfg_snn=cfg_snn,
        duracao=args.duracao,
        taxa_amostragem=args.taxa_amostragem,
    )
    print(f"[Identificação] Predição: {pessoa} (confiança≈{conf:.3f})")


def cmd_verificar(args: argparse.Namespace) -> None:
    """Identifica, mas permite retornar 'desconhecido' com base em um limiar."""

    cfg_extracao = ConfigExtracao(
        taxa_amostragem=args.taxa_amostragem,
        tamanho_janela=args.tamanho_janela,
        tamanho_passo=args.tamanho_passo,
        wavelet_base=args.wavelet,
        num_bandas=args.num_bandas,
        duracao_referencia=args.duracao_referencia,
    )
    cfg_snn = ConfigSNN(
        passos_por_janela=args.passos_por_janela,
        alvo_spikes_por_passo=args.alvo_spikes_por_passo,
    )

    model, rotulos = carregar_modelo_e_rotulos(
        caminho_modelo=args.modelo,
        caminho_rotulos=args.rotulos,
        num_inputs=cfg_extracao.num_bandas,
    )

    pessoa, conf, _ = identificar_locutor_por_microfone(
        model,
        rotulos,
        cfg_extracao=cfg_extracao,
        cfg_snn=cfg_snn,
        duracao=args.duracao,
        taxa_amostragem=args.taxa_amostragem,
    )

    final = aplicar_limiar_desconhecido(pessoa, conf, limiar=args.limiar)
    print(f"[Verificação] Predição: {final} (confiança≈{conf:.3f}, limiar={args.limiar:.2f})")


def cmd_avaliar(args: argparse.Namespace) -> None:
    """Avalia o modelo em WAVs gravados (por pessoa) e imprime uma matriz de confusão simples."""

    cfg_extracao = ConfigExtracao(
        taxa_amostragem=args.taxa_amostragem,
        tamanho_janela=args.tamanho_janela,
        tamanho_passo=args.tamanho_passo,
        wavelet_base=args.wavelet,
        num_bandas=args.num_bandas,
        duracao_referencia=args.duracao_referencia,
    )
    cfg_snn = ConfigSNN(
        passos_por_janela=args.passos_por_janela,
        alvo_spikes_por_passo=args.alvo_spikes_por_passo,
    )

    model, rotulos = carregar_modelo_e_rotulos(
        caminho_modelo=args.modelo,
        caminho_rotulos=args.rotulos,
        num_inputs=cfg_extracao.num_bandas,
    )

    pessoas = listar_amostras_por_pessoa(args.diretorio_dados)
    if not pessoas:
        raise FileNotFoundError("Nenhum WAV encontrado para avaliação. Esperado: <base>/<pessoa>/*.wav")

    idx_por_pessoa = {p: i for i, p in enumerate(rotulos)}
    conf = np.zeros((len(rotulos), len(rotulos)), dtype=np.int64)
    total = 0
    corretos = 0

    for pessoa_real, wavs in pessoas.items():
        if pessoa_real not in idx_por_pessoa:
            continue
        y = idx_por_pessoa[pessoa_real]
        for wav in wavs:
            pred, conf_pred, _ = identificar_locutor_por_wav(
                model,
                rotulos,
                caminho_wav=wav,
                cfg_extracao=cfg_extracao,
                cfg_snn=cfg_snn,
            )
            yhat = idx_por_pessoa.get(pred)
            if yhat is None:
                continue
            conf[y, yhat] += 1
            total += 1
            corretos += int(pred == pessoa_real)
            if args.verbose:
                print(f"[Avaliar] {pessoa_real} -> {pred} (conf≈{conf_pred:.3f}) | {wav}")

    acc = (corretos / total) if total else 0.0
    print(f"[Avaliar] Arquivos avaliados: {total} | acurácia={acc:.3f}")
    print("[Avaliar] Matriz de confusão (linhas=real, colunas=pred):")
    header = "        " + " ".join([f"{r:>8}" for r in rotulos])
    print(header)
    for i, r in enumerate(rotulos):
        row = " ".join([f"{int(v):8d}" for v in conf[i]])
        print(f"{r:>8} {row}")


def construir_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Biometria por voz (demo): WPT -> codificação em spikes -> SNN -> classificação por pessoa.\n"
            "Fluxo recomendado: capturar dados -> treinar -> identificar."
        )
    )

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
    demo.add_argument("--saida-plot", type=str, default="result_pipeline_wpt_snn.png")
    demo.set_defaults(func=cmd_demo)

    # --- Captura/cadastro ---
    cap = sub.add_parser("capturar", help="Captura áudio e salva WAV para uma pessoa (cadastro)")
    cap.add_argument("--pessoa", type=str, required=True, help="ID da pessoa (ex.: alice)")
    cap.add_argument("--diretorio-dados", type=str, default="dados/vozes")
    cap.add_argument("--duracao", type=float, default=3.0)
    cap.add_argument("--taxa-amostragem", type=int, default=44100)
    cap.set_defaults(func=cmd_capturar)

    # Alias mais biométrico ("enrolar" = cadastrar)
    enr = sub.add_parser("enrolar", help="Alias de capturar (cadastrar amostras por pessoa)")
    enr.add_argument("--pessoa", type=str, required=True, help="ID da pessoa (ex.: alice)")
    enr.add_argument("--diretorio-dados", type=str, default="dados/vozes")
    enr.add_argument("--duracao", type=float, default=3.0)
    enr.add_argument("--taxa-amostragem", type=int, default=44100)
    enr.set_defaults(func=cmd_capturar)

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
    tr.add_argument("--alvo-spikes-por-passo", type=float, default=0.10)
    tr.add_argument("--epocas", type=int, default=5)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--saida-modelo", type=str, default="modelo_snn_locutor.pt")
    tr.add_argument("--saida-rotulos", type=str, default="rotulos_locutor.json")
    tr.set_defaults(func=cmd_treinar)

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
    inf.add_argument("--alvo-spikes-por-passo", type=float, default=0.10)
    inf.set_defaults(func=cmd_identificar)

    # --- Verificação (com desconhecido) ---
    ver = sub.add_parser(
        "verificar",
        help=("Verifica a identidade e pode retornar 'desconhecido' se a confiança for baixa"),
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
    ver.add_argument("--alvo-spikes-por-passo", type=float, default=0.10)
    ver.add_argument(
        "--limiar",
        type=float,
        default=0.55,
        help="Abaixo deste valor, retorna 'desconhecido'",
    )
    ver.set_defaults(func=cmd_verificar)

    # --- Avaliação offline ---
    av = sub.add_parser("avaliar", help="Avalia o modelo em WAVs gravados e imprime matriz de confusão")
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
    av.add_argument("--alvo-spikes-por-passo", type=float, default=0.10)
    av.add_argument("--verbose", action="store_true")
    av.set_defaults(func=cmd_avaliar)

    return p


def main() -> None:
    p = construir_cli()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
