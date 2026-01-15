"""Utilitários de E/S de áudio em WAV PCM.

Este módulo é propositalmente simples e auto-contido:
- usa apenas a stdlib (`wave`) para leitura/escrita;
- trabalha com `float32` normalizado em [-1, 1];
- converte para PCM 16-bit na gravação.
"""

from __future__ import annotations

import os
import wave
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WavInfo:
    taxa_amostragem: int
    num_canais: int


def salvar_wav_pcm16(caminho: str, sinal: np.ndarray, taxa_amostragem: int) -> None:
    """Salva um WAV PCM 16-bit mono.

    Entrada:
    - sinal: float32 em [-1, 1] (ou próximo disso)

    Observação:
    - Usamos o módulo padrão `wave` para evitar dependências extras.
    """

    # Garante que o diretório existe antes de escrever o arquivo.
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)

    # Normaliza o sinal (float32) para o intervalo seguro de PCM 16-bit.
    x = np.asarray(sinal, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)

    with wave.open(caminho, "wb") as wf:
        # Header WAV: mono, 16-bit, taxa de amostragem definida pelo usuário.
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(taxa_amostragem))
        wf.writeframes(pcm.tobytes())


def carregar_wav_pcm16(caminho: str) -> tuple[np.ndarray, WavInfo]:
    """Carrega WAV PCM (8/16/32-bit int) usando stdlib.

    Retorna:
    - sinal_mono: float32 em [-1, 1]
    - info

    Limitações:
    - Arquivos WAV comprimidos (ex.: ADPCM) podem falhar.
    """

    # Lê metadados e frames brutos do arquivo WAV.
    with wave.open(caminho, "rb") as wf:
        num_canais = wf.getnchannels()
        taxa = int(wf.getframerate())
        largura = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    # Converte o PCM inteiro para float32 em [-1, 1].
    if largura == 1:
        # unsigned 8-bit
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif largura == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif largura == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Largura de amostra WAV não suportada: {largura} bytes")

    # Converte para mono se necessário (média simples por canal).
    if num_canais > 1:
        data = data.reshape(-1, num_canais).mean(axis=1)

    return data.astype(np.float32), WavInfo(taxa_amostragem=taxa, num_canais=num_canais)
