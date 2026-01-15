"""Carga e preparação de datasets a partir de WAVs.

Este módulo centraliza:
- a estrutura de diretórios por pessoa;
- o janelamento do áudio;
- a extração de características WPT;
- a normalização para entrada da SNN.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from arquivo_audio import carregar_wav_pcm16
from captura import reamostrar_audio
from caracteristicas import calcular_energia_wpt
from ondaletas import calcular_nivel_wpt
from preprocessamento import preprocessar_energia_wpt_para_snn
from janelamento import aplicar_janelamento


@dataclass(frozen=True)
class ConfigExtracao:
    taxa_amostragem: int = 44100
    tamanho_janela: int = 512
    tamanho_passo: int = 256
    wavelet_base: str = "db4"
    num_bandas: int = 100
    duracao_referencia: float = 1.0


def listar_amostras_por_pessoa(diretorio_base: str) -> dict[str, list[str]]:
    """Espera a estrutura: <diretorio_base>/<id_pessoa>/*.wav"""

    # Mapeia cada pessoa para sua lista de WAVs (ordenada).
    pessoas: dict[str, list[str]] = {}
    for pessoa_dir in sorted(glob.glob(os.path.join(diretorio_base, "*"))):
        if not os.path.isdir(pessoa_dir):
            continue
        pessoa = os.path.basename(pessoa_dir)
        wavs = sorted(glob.glob(os.path.join(pessoa_dir, "*.wav")))
        if wavs:
            pessoas[pessoa] = wavs
    return pessoas


def extrair_janelas_caracteristicas(
    audio: np.ndarray,
    *,
    cfg: ConfigExtracao,
) -> list[np.ndarray]:
    """Extrai uma lista de vetores (um por janela) já preprocessados em [0,1]."""

    # Define o nível de WPT coerente com o tamanho de janela e número de bandas.
    nivel_wpt = calcular_nivel_wpt(
        duracao=cfg.duracao_referencia,
        taxa_amostragem=cfg.taxa_amostragem,
        tamanho_janela=cfg.tamanho_janela,
        num_bandas=cfg.num_bandas,
        wavelet_base=cfg.wavelet_base,
    )

    # Segmenta o áudio em janelas sobrepostas.
    janelas = aplicar_janelamento(
        audio,
        cfg.tamanho_janela,
        cfg.tamanho_passo,
        funcao_janela=np.hanning,
    )

    # Constrói o vetor de características por janela.
    caracteristicas: list[np.ndarray] = []
    for janela in janelas:
        # Energia por banda + normalização para [0, 1].
        energia = calcular_energia_wpt(
            janela,
            wavelet_base=cfg.wavelet_base,
            nivel_maximo=nivel_wpt,
            num_bandas=cfg.num_bandas,
        )
        caracteristicas.append(preprocessar_energia_wpt_para_snn(energia))

    return caracteristicas


def carregar_dataset_janelas(
    diretorio_base: str,
    *,
    cfg: ConfigExtracao,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carrega WAVs por pessoa e retorna um dataset de janelas.

    Retorna:
    - X: [N, num_bandas] (float32 em [0,1])
    - y: [N] (int64) -> índice da pessoa
    - rotulos: lista de ids de pessoas (ordem = índice)
    """

    # Lê a estrutura <base>/<pessoa>/*.wav
    pessoas = listar_amostras_por_pessoa(diretorio_base)
    if not pessoas:
        raise FileNotFoundError(
            "Nenhum WAV encontrado. Esperado: <base>/<pessoa>/*.wav. "
            "Ex.: dados/vozes/alice/*.wav"
        )

    # Rotula pessoas com índices estáveis (ordem alfabética).
    rotulos = sorted(pessoas.keys())
    idx_por_pessoa = {p: i for i, p in enumerate(rotulos)}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    # Percorre cada pessoa e empilha as janelas em um único dataset.
    for pessoa, arquivos in pessoas.items():
        for wav_path in arquivos:
            sinal, info = carregar_wav_pcm16(wav_path)
            if info.taxa_amostragem != cfg.taxa_amostragem:
                sinal = reamostrar_audio(
                    sinal,
                    taxa_origem=info.taxa_amostragem,
                    taxa_destino=cfg.taxa_amostragem,
                )

            caracs = extrair_janelas_caracteristicas(sinal, cfg=cfg)
            for c in caracs:
                X_list.append(c)
                y_list.append(idx_por_pessoa[pessoa])

    # Empilha tudo em arrays NumPy padronizados.
    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, rotulos
