"""Pré-processamento das energias WPT para entrada em SNN."""

import numpy as np


def preprocessar_energia_wpt_para_snn(
    energia: np.ndarray, *, limiar_maximo: float = 1e-8
) -> np.ndarray:
    """
    Prepara o vetor de energia WPT para ser usado como entrada na SNN.
    * Objetivo:
        - Reduzir faixa dinâmica (energia WPT pode variar bastante).
        - Padronizar escala para alimentar um codificador de spikes.
    * Estratégia:
        1) Remove negativos por segurança numérica.
        2) Comprime com log(1+E).
        3) Normaliza por máximo para obter valores no intervalo [0, 1].
    """

    # Converte para float32 e elimina valores negativos por segurança.
    energia_proc = np.asarray(energia, dtype=np.float32)
    energia_proc = np.maximum(energia_proc, 0.0)

    # Compressão logarítmica para reduzir faixa dinâmica.
    energia_proc = np.log1p(energia_proc)

    # Normaliza em [0,1] usando o máximo da janela.
    valor_maximo = float(np.max(energia_proc))

    # Evita divisão por zero.
    if valor_maximo < limiar_maximo:
        return np.zeros_like(energia_proc, dtype=np.float32)

    # Normalização final.
    energia_proc = energia_proc / (valor_maximo + limiar_maximo)
    energia_proc = np.clip(energia_proc, 0.0, 1.0)

    # Retorna o vetor processado.
    return energia_proc.astype(np.float32)
