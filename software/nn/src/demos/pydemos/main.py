from capture import capturar_audio
from windowing import aplicar_janelamento
from features import compute_wpt_energy
from wavelet import calcular_wpt_level
from snn import create_snn_model
from visualization import plotar_resultados
import torch
import numpy as np

# --- Configurações Explícitas (Requirements) ---
SAMPLE_RATE = 44100  # 1. Audio Acquisition: Fixed fs

WINDOW_SIZE = 512  # 2. Windowing: Fixed length
HOP_SIZE = 256  # 2. Windowing: Fixed hop (50% overlap)

WAVELET_TYPE = "db4"  # 3. WPT: Fixed basis

# 3. WPT: Max level (calculado dinamicamente)
# A profundidade máxima da Wavelet Packet depende do tamanho do sinal de entrada.
# Como aqui extraímos WPT *por janela*, o limite prático vem do tamanho efetivo da janela.
# Para manter compatibilidade com NUM_BANDS, escolhemos um nível tal que 2^level >= NUM_BANDS,
# respeitando o máximo permitido pelo tamanho do sinal disponível (que pode variar com DURATION).

NUM_BANDS = 100  # 4. Energy Extraction: 100 bands
DURATION = 1.0  # Duration in seconds

def pipeline_exec():
    # 1. Audio Acquisition
    audio_raw = capturar_audio(DURATION, SAMPLE_RATE)

    wpt_level = calcular_wpt_level(DURATION, SAMPLE_RATE, WINDOW_SIZE, NUM_BANDS, WAVELET_TYPE)
    print(f"[Features] WPT_LEVEL calculado: {wpt_level}")

    # 2. Windowing
    janelas = aplicar_janelamento(
        audio_raw, WINDOW_SIZE, HOP_SIZE, window_fn=np.hanning
    )

    if not janelas:
        print("Nenhuma janela gerada.")
        return

    # 3 & 4. WPT & Energy Extraction
    print(f"[Features] Extraindo energia WPT ({NUM_BANDS} bandas)...")
    features_janelas = []
    for janela in janelas:
        energy = compute_wpt_energy(
            janela, wavelet=WAVELET_TYPE, max_level=wpt_level, num_bands=NUM_BANDS
        )
        features_janelas.append(energy)

    # 6. SNN Initialization
    model = create_snn_model()
    print(f"[SNN] Modelo inicializado (Weights Deterministic).")

    # 7. Inference
    print("[SNN] Processando inferência...")
    spikes_saida = []

    # --- Simulação temporal (explicação didática) ---
    # Conceito 1 — Janela de áudio:
    #   O sinal contínuo de áudio é dividido em pequenos trechos (“janelas”) de tamanho fixo (WINDOW_SIZE).
    #   Cada janela vira um vetor de features (aqui: energia WPT com NUM_BANDS dimensões).
    #
    # Conceito 2 — Passo de tempo (timestep) para uma SNN:
    #   Em uma Spiking Neural Network, a dinâmica temporal normalmente é modelada por estados internos
    #   (ex.: potencial de membrana). Em um cenário “online”, a rede recebe uma sequência ao longo do tempo:
    #   x[0], x[1], x[2], ... e atualiza seu estado a cada passo.
    #
    # Ideia usada aqui:
    #   Tratamos *cada janela* como um timestep: para cada vetor de features, chamamos o modelo uma vez.
    #   Isso cria uma sequência de saídas (spikes) alinhada às janelas, permitindo visualizar “atividade” ao longo
    #   da gravação (eixo x = índice da janela).
    #
    # Importante — Estado interno (stateful vs stateless):
    #   Se o neurônio spiking mantiver estado entre passos, a saída em t pode depender de entradas anteriores
    #   (memória temporal). Porém, em snnTorch, muitos módulos (ex.: snntorch.Leaky) podem reinicializar o estado
    #   automaticamente dependendo de como o modelo foi implementado (ex.: init_hidden=True e sem passar mem).
    #
    # No nosso caso (stateful):
    #   Aqui nós *propagamos explicitamente* o estado entre janelas: o modelo retorna (spk, state) e nós
    #   alimentamos o mesmo state de volta na próxima iteração. Esse state contém as memórias (ex.: mem1/mem2/mem3)
    #   dos neurônios LIF, então a saída em uma janela pode depender do histórico recente.
    #
    # O que isso muda na prática:
    #   - A rede passa a ter “memória” entre janelas (dinâmica temporal de verdade).
    #   - Dois sinais com as mesmas features instantâneas podem produzir spikes diferentes dependendo do contexto.
    #
    # Observação:
    #   Se você quiser reiniciar a dinâmica (por exemplo, em outra gravação), basta resetar `state = None`.

    with torch.no_grad():
        state = None  # estado interno (memórias) propagado entre janelas
        for feat in features_janelas:
            # 5. Encoding (Input Transformation happens inside forward via direct injection scaling)
            # Input shape: [1, 100]
            input_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

            # Forward pass
            spk, state = model(input_t, state)

            spikes_saida.append(spk)

    # 8. Data Collection & Visualization
    plotar_resultados(
        features_janelas,
        spikes_saida,
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        duration=DURATION,
        wavelet=WAVELET_TYPE,
        wpt_level=wpt_level,
        num_bands=NUM_BANDS,
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
