import numpy as np

def aplicar_janelamento(
    sinal: np.ndarray,
    tamanho_janela: int,
    tamanho_passo: int,
    funcao_janela=None,
    **kwargs,
) -> list[np.ndarray]:
    """
    Segmenta o sinal em janelas sobrepostas com função de janelamento explícita.
    
    Args:
        sinal (np.ndarray): Sinal de entrada 1D.
        tamanho_janela (int): Tamanho da janela em amostras.
        tamanho_passo (int): Passo entre janelas em amostras.
        funcao_janela (callable): Função geradora da janela (ex.: np.hanning, np.hamming).
        
    Returns:
        list[np.ndarray]: Lista de janelas processadas.
    """
    # Compatibilidade: permitir chamada antiga com window_fn=...
    if funcao_janela is None:
        funcao_janela = kwargs.pop("window_fn", np.hanning)
    if kwargs:
        raise TypeError(f"Argumentos inesperados em aplicar_janelamento: {list(kwargs.keys())}")

    num_amostras = len(sinal)
    janelas = []
    
    if num_amostras < tamanho_janela:
        print("[Janelamento] Aviso: sinal menor que o tamanho da janela.")
        return []
        
    # Gerar a janela uma única vez
    janela = funcao_janela(tamanho_janela)
    
    # Iterar sobre o sinal com passo fixo (hop_size)
    for inicio in range(0, num_amostras - tamanho_janela + 1, tamanho_passo):
        fim = inicio + tamanho_janela
        segmento = sinal[inicio:fim]
        
        # Aplicar a função de janela (multiplicação de elementos)
        segmento_janelado = segmento * janela
        janelas.append(segmento_janelado)
        
    print(
        f"[Janelamento] Geradas {len(janelas)} janelas (tamanho={tamanho_janela}, passo={tamanho_passo})"
    )
    return janelas
