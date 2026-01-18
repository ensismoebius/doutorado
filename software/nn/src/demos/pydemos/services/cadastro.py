"""Fluxo de cadastro (enrolamento) de amostras de voz.

Responsabilidade:
- capturar áudio do microfone;
- salvar em um diretório por pessoa;
- gerar nomes de arquivos com timestamp.
"""

from __future__ import annotations

import os
from datetime import datetime

from infra.arquivo_audio import salvar_wav_pcm16
from infra.captura import capturar_audio


def capturar_e_salvar_amostra(
    *,
    pessoa: str,
    diretorio_base: str,
    duracao: float,
    taxa_amostragem: int,
) -> str:
    """Captura áudio do microfone e salva como WAV para usar no treinamento.

    Estrutura gerada:
    - <diretorio_base>/<pessoa>/amostra_<timestamp>.wav

    Isso viabiliza um fluxo simples:
    1) Cadastrar 2-5 amostras por pessoa.
    2) Treinar o classificador SNN.
    3) Identificar uma pessoa em tempo de execução.
    """

    # Timestamp para evitar colisão de nomes (várias amostras por pessoa).
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta = os.path.join(diretorio_base, pessoa)
    os.makedirs(pasta, exist_ok=True)

    # Captura e grava em PCM 16-bit mono.
    caminho = os.path.join(pasta, f"amostra_{ts}.wav")
    audio = capturar_audio(duracao, taxa_amostragem)
    salvar_wav_pcm16(caminho, audio, taxa_amostragem)

    print(f"[Cadastro] Salvo: {caminho}")
    return caminho
