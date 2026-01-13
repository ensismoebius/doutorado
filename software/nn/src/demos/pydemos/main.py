from capture import capturar_audio
from windowing import aplicar_janelamento
from features import calcular_energia_wpt
from wavelet import calcular_nivel_wpt
from snn import criar_modelo_snn
from visualization import plotar_resultados
import torch
import numpy as np

# --- Configurações Explícitas ---
TAXA_AMOSTRAGEM = 44100  # 1) Captura de áudio: taxa fixa

TAMANHO_JANELA = 512  # 2) Janelamento: tamanho fixo
TAMANHO_PASSO = 256  # 2) Janelamento: passo fixo (50% overlap)

WAVELET_BASE = "db4"  # 3) WPT: base fixa

# 3. WPT: Max level (calculado dinamicamente)
# A profundidade máxima da Wavelet Packet depende do tamanho do sinal de entrada.
# Como aqui extraímos WPT *por janela*, o limite prático vem do tamanho efetivo da janela.
# Para manter compatibilidade com NUM_BANDS, escolhemos um nível tal que 2^level >= NUM_BANDS,
# respeitando o máximo permitido pelo tamanho do sinal disponível (que pode variar com DURATION).

NUM_BANDAS = 100  # 4) Extração de energia: 100 bandas
DURACAO = 1.0  # duração em segundos

def pipeline_exec():
    # 1) Captura de áudio
    audio_raw = capturar_audio(DURACAO, TAXA_AMOSTRAGEM)

    nivel_wpt = calcular_nivel_wpt(
        duracao=DURACAO,
        taxa_amostragem=TAXA_AMOSTRAGEM,
        tamanho_janela=TAMANHO_JANELA,
        num_bandas=NUM_BANDAS,
        wavelet_base=WAVELET_BASE,
    )
    print(f"[Características] Nível WPT calculado: {nivel_wpt}")

    # 2) Janelamento
    janelas = aplicar_janelamento(
        audio_raw, TAMANHO_JANELA, TAMANHO_PASSO, funcao_janela=np.hanning
    )

    if not janelas:
        print("Nenhuma janela gerada.")
        return

    # 3) WPT + 4) Energia por banda
    print(f"[Características] Extraindo energia WPT ({NUM_BANDAS} bandas)...")
    caracteristicas_janelas = []
    for janela in janelas:
        energia = calcular_energia_wpt(
            janela,
            wavelet_base=WAVELET_BASE,
            nivel_maximo=nivel_wpt,
            num_bandas=NUM_BANDAS,
        )
        caracteristicas_janelas.append(energia)

    # 6) Inicialização da SNN
    model = criar_modelo_snn()
    print("[SNN] Modelo inicializado (pesos determinísticos).")

    # 7) Inferência
    print("[SNN] Processando inferência...")
    lista_spikes_saida = []

    # --- Simulação temporal (explicação didática) ---
    # Conceito 1 — Janela de áudio:
    #   O sinal contínuo de áudio é dividido em pequenos trechos (“janelas”) de tamanho fixo.
    #   Cada janela vira um vetor de características (aqui: energia WPT com NUM_BANDAS dimensões).
    #
    # Conceito 2 — Passo de tempo (passo temporal) para uma SNN:
    #   Em uma Spiking Neural Network, a dinâmica temporal normalmente é modelada por estados internos
    #   (ex.: potencial de membrana). Em um cenário “online”, a rede recebe uma sequência ao longo do tempo:
    #   x[0], x[1], x[2], ... e atualiza seu estado a cada passo.
    #
    # Ideia usada aqui:
    #   Tratamos *cada janela* como um passo de tempo: para cada vetor de características, chamamos o modelo uma vez.
    #   Isso cria uma sequência de saídas (spikes) alinhada às janelas, permitindo visualizar “atividade” ao longo
    #   da gravação (eixo x = índice da janela).
    #
    # Importante — Estado interno (com estado vs sem estado):
    #   Se o neurônio spiking mantiver estado entre passos, a saída em t pode depender de entradas anteriores
    #   (memória temporal). Porém, em snnTorch, alguns módulos podem reinicializar o estado automaticamente
    #   dependendo de como o modelo foi implementado.
    #
    # No nosso caso (com estado):
    #   Aqui nós *propagamos explicitamente* o estado entre janelas: o modelo retorna (spk, state) e nós
    #   alimentamos o mesmo state de volta na próxima iteração. Esse state contém as memórias (ex.: mem1/mem2/mem3)
    #   dos neurônios LIF, então a saída em uma janela pode depender do histórico recente.
    #
    # O que isso muda na prática:
    #   - A rede passa a ter “memória” entre janelas (dinâmica temporal de verdade).
    #   - Dois sinais com as mesmas características instantâneas podem produzir spikes diferentes dependendo do contexto.
    #
    # Observação:
    #   Se você quiser reiniciar a dinâmica (por exemplo, em outra gravação), basta resetar `state = None`.

    with torch.no_grad():
        state = None  # estado interno (memórias) propagado entre janelas
        for carac in caracteristicas_janelas:
            # 5) Codificação: acontece dentro do modelo (injeção direta de corrente via escala)
            entrada = torch.tensor(carac, dtype=torch.float32).unsqueeze(0)

            # Forward pass
            spk, state = model(entrada, state)

            lista_spikes_saida.append(spk)

    # 8) Coleta e visualização
    plotar_resultados(
        caracteristicas_janelas,
        lista_spikes_saida,
        sample_rate=TAXA_AMOSTRAGEM,
        window_size=TAMANHO_JANELA,
        hop_size=TAMANHO_PASSO,
        duration=DURACAO,
        wavelet=WAVELET_BASE,
        wpt_level=nivel_wpt,
        num_bands=NUM_BANDAS,
        stateful=True,
    )
    print("Pipeline concluído.")


if __name__ == "__main__":
    try:
        pipeline_exec()
    except Exception as e:
        print(f"Erro fatal: {e}")
        import traceback

        traceback.print_exc()
