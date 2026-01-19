"""Implementação dos comandos da CLI do demo de biometria por voz."""

import argparse

import numpy as np
import torch

from core.configs import ConfigExtracao, ConfigSNN
from infra.captura import capturar_audio
from services.cadastro import capturar_e_salvar_amostra
from services.conjunto_dados import (
    extrair_janelas_caracteristicas,
    listar_amostras_por_pessoa,
)
from services.identificacao_locutor import (
    aplicar_limiar_desconhecido,
    carregar_modelo_e_rotulos,
    identificar_locutor_por_microfone,
    identificar_locutor_por_wav,
    salvar_modelo_e_rotulos,
    treinar_classificador_locutor,
)
from services.modelos.rede_snn import criar_modelo_snn
from utils.codificacao import codificar_poisson
from infra.visualizacao import plotar_resultados


def cmd_demo(args: argparse.Namespace) -> None:
    """Demo visual: pipeline WPT -> preprocess -> codificação Poisson -> SNN -> plots."""

    # Configurações da extração e da SNN para este demo visual.
    cfg_extracao = ConfigExtracao(
        taxa_amostragem=args.taxa_amostragem,
        tamanho_janela=args.tamanho_janela,
        tamanho_passo=args.tamanho_passo,
        wavelet_base=args.wavelet,
        num_bandas=args.num_bandas,
        duracao_referencia=args.duracao,
    )
    cfg_snn = ConfigSNN(
        passos_por_janela=args.passos_por_janela,
        profundidade=getattr(args, "profundidade", None),
    )

    # Captura áudio ao vivo e extrai características por janela.
    audio = capturar_audio(args.duracao, args.taxa_amostragem)
    caracs = extrair_janelas_caracteristicas(audio, cfg=cfg_extracao)
    if not caracs:
        raise RuntimeError("Nenhuma janela gerada.")

    # Modelo com saída do mesmo tamanho das entradas (para manter plots didáticos).
    model = criar_modelo_snn(
        num_inputs=cfg_extracao.num_bandas,
        num_outputs=cfg_extracao.num_bandas,
        profundidade=getattr(args, "profundidade", None),
    )
    model.eval()

    lista_spikes_saida = []
    with torch.no_grad():
        state = None
        # Processa cada janela; soma spikes ao longo do tempo para plot.
        for c in caracs:
            xb = torch.tensor(c, dtype=torch.float32).unsqueeze(0)
            spk_in = codificar_poisson(
                xb, passos=cfg_snn.passos_por_janela, adaptativo=True
            )
            spk_out_seq, state = model(spk_in, state)

            # Para plotar por janela, agregamos os spikes ao longo dos passos.
            spk_janela = spk_out_seq.sum(dim=0)  # [1, num_bandas]
            lista_spikes_saida.append(spk_janela)

    # Gera os gráficos de características e spikes.
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
    # Atalho para cadastrar uma nova amostra de voz.
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
        profundidade=getattr(args, "profundidade", None),
    )

    model, rotulos = treinar_classificador_locutor(
        # Configura extração e SNN para treino.
        args.diretorio_dados,
        cfg_extracao=cfg_extracao,
        cfg_snn=cfg_snn,
        epocas=args.epocas,
        taxa_aprendizado=args.lr,
        num_blocos_residuais=getattr(args, "profundidade", None),
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
        profundidade=getattr(args, "profundidade", None),
    )

    model, rotulos = carregar_modelo_e_rotulos(
        # Configura extração e SNN para inferência.
        caminho_modelo=args.modelo,
        caminho_rotulos=args.rotulos,
        num_inputs=cfg_extracao.num_bandas,
        num_blocos_residuais=getattr(args, "profundidade", None),
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
        profundidade=getattr(args, "profundidade", None),
    )

    model, rotulos = carregar_modelo_e_rotulos(
        caminho_modelo=args.modelo,
        caminho_rotulos=args.rotulos,
        num_inputs=cfg_extracao.num_bandas,
        num_blocos_residuais=getattr(args, "profundidade", None),
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
    print(
        f"[Verificação] Predição: {final} (confiança≈{conf:.3f}, limiar={args.limiar:.2f})"
    )


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
        num_blocos_residuais=getattr(args, "profundidade", None),
    )

    pessoas = listar_amostras_por_pessoa(args.diretorio_dados)
    if not pessoas:
        raise FileNotFoundError(
            "Nenhum WAV encontrado para avaliação. Esperado: <base>/<pessoa>/*.wav"
        )

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
                print(
                    f"[Avaliar] {pessoa_real} -> {pred} (conf≈{conf_pred:.3f}) | {wav}"
                )

    acc = (corretos / total) if total else 0.0
    print(f"[Avaliar] Arquivos avaliados: {total} | acurácia={acc:.3f}")
    print("[Avaliar] Matriz de confusão (linhas=real, colunas=pred):")
    header = "        " + " ".join([f"{r:>8}" for r in rotulos])
    print(header)
    for i, r in enumerate(rotulos):
        row = " ".join([f"{int(v):8d}" for v in conf[i]])
        print(f"{r:>8} {row}")
