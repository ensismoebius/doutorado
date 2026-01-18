"""Pré-processamento das energias WPT para entrada em SNN."""

import numpy as np


def preprocessar_energia_wpt_para_snn(
    energia: np.ndarray,
    *,
    eps: float = 1e-8,
    usar_log: bool = True,
) -> np.ndarray:
    """Prepara o vetor de energia WPT para ser usado como entrada na SNN.

    Objetivo:
    - Reduzir faixa dinâmica (energia WPT pode variar bastante).
    - Padronizar escala para alimentar um codificador de spikes (tipicamente em [0, 1]).

    Estratégia (didática e robusta para demo):
    1) Remove negativos por segurança numérica.
    2) (Opcional) Comprime com log(1+E).
    3) Normaliza por máximo para obter valores em [0, 1].

    Observação:
    - Para treino sério em biometria de voz, você pode preferir normalização global (média/desvio)
      por pessoa ou por corpus. Aqui escolhemos uma normalização por janela para ser simples e estável.
    """

    # Converte para float32 e elimina valores negativos por segurança.
    x = np.asarray(energia, dtype=np.float32)
    x = np.maximum(x, 0.0)

    # Compressão logarítmica opcional para reduzir faixa dinâmica.
    if usar_log:
        x = np.log1p(x)

    # Normaliza em [0,1] usando o máximo da janela.
    maxv = float(np.max(x))
    if maxv < eps:
        return np.zeros_like(x, dtype=np.float32)

    x = x / (maxv + eps)
    x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32)
