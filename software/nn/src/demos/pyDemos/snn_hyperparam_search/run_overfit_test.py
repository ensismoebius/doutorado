"""Overfit sanity check: train on a tiny subset to see if loss can go below 0.01."""

from __future__ import annotations
import os
import pathlib
import sys
import time

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "voice_biometrics_snn_py"
sys.path.insert(0, str(PKG_ROOT))

from core.configs import ConfigExtracao, ConfigSNN
from services.identificacao_locutor import treinar_classificador_locutor
from services.conjunto_dados import carregar_dataset_janelas

DATA_DIR = "dados/vozes"
EPOCHS = 300
MAX_SAMPLES = 8

cfg_extr = ConfigExtracao(
    taxa_amostragem=44100,
    tamanho_janela=512,
    tamanho_passo=256,
    wavelet_base="db4",
    num_bandas=100,
)
cfg_snn = ConfigSNN(passos_por_janela=5, alvo_spikes_por_passo=0.1)

print("Loading dataset (tiny subset)")
X, y, rotulos = carregar_dataset_janelas(DATA_DIR, cfg=cfg_extr)
print("Dataset loaded, total samples:", X.shape[0])
print("Using first", MAX_SAMPLES, "samples to overfit")

start = time.time()
model, labels, stats = treinar_classificador_locutor(
    DATA_DIR,
    cfg_extracao=cfg_extr,
    cfg_snn=cfg_snn,
    epocas=EPOCHS,
    taxa_aprendizado=1e-3,
    num_blocos_residuais=1,
    tamanho_camada_oculta=512,
    loss_mode="mse_vector",
    num_passes=10,
    dataset_override=(X[:MAX_SAMPLES], y[:MAX_SAMPLES], rotulos),
    max_samples=MAX_SAMPLES,
    return_stats=True,
)
elapsed = time.time() - start
print("Overfit finished in", round(elapsed, 2), "s; final loss=", stats.get("loss"))
if stats.get("loss") is not None and stats.get("loss") < 0.01:
    print("SUCCESS: reached loss < 0.01")
else:
    print("Did not reach loss < 0.01")
