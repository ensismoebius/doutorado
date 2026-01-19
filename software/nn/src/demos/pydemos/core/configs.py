"""Estruturas de dados (configurações e tipos) sem dependências de I/O."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ConfigExtracao:
    taxa_amostragem: int = 44100
    tamanho_janela: int = 512
    tamanho_passo: int = 256
    wavelet_base: str = "db4"
    num_bandas: int = 100
    duracao_referencia: float = 1.0


@dataclass(frozen=True)
class ConfigSNN:
    passos_por_janela: int = 10
    alvo_spikes_por_passo: float = 0.10
    profundidade: Optional[int] = None


@dataclass(frozen=True)
class WavInfo:
    taxa_amostragem: int
    num_canais: int
