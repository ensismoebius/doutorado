"""Targeted hyperparameter search focusing on larger capacity and passes.
Stops early if any configuration reaches final loss < 0.01.
Saves results to `experiments/targeted_search.csv` and prints progress.
"""

from __future__ import annotations
import csv
import itertools
import os
import pathlib
import sys
import time

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from core.configs import ConfigExtracao, ConfigSNN
from services.identificacao_locutor import treinar_classificador_locutor
from services.conjunto_dados import carregar_dataset_janelas

DATA_DIR = "dados/vozes"
EPOCHS = 80

RESULTS_DIR = os.path.dirname(__file__) 
RESULTS_CSV = os.path.join(RESULTS_DIR, "targeted_search.csv")

# Targeted grid (kept moderate to finish in reasonable time)
grid = {
    "lr": [1e-3, 5e-4, 1e-4],
    "hidden": [256, 512, 1024],
    "loss_mode": ["mse_vector", "van_rossum"],
    "num_passes": [5, 10],
}

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Preloading dataset (full) — this may take a moment")
cfg_extr = ConfigExtracao(
    taxa_amostragem=44100,
    tamanho_janela=512,
    tamanho_passo=256,
    wavelet_base="db4",
    num_bandas=100,
)
cfg_snn = ConfigSNN(passos_por_janela=5, alvo_spikes_por_passo=0.1)
X, y, rotulos = carregar_dataset_janelas(DATA_DIR, cfg=cfg_extr)
print(f"Loaded {X.shape[0]} samples; running targeted grid")

combinations = list(
    itertools.product(grid["lr"], grid["hidden"], grid["loss_mode"], grid["num_passes"])
)
print(f"Running {len(combinations)} configurations")

header = [
    "lr",
    "hidden",
    "loss_mode",
    "num_passes",
    "time_s",
    "final_loss",
    "final_acc",
]
found = None
with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    for lr, hidden, loss_mode, num_passes in combinations:
        cfg = {
            "lr": lr,
            "hidden": hidden,
            "loss_mode": loss_mode,
            "num_passes": num_passes,
        }
        print("Running:", cfg)
        start = time.time()
        model, labels, stats = treinar_classificador_locutor(
            DATA_DIR,
            cfg_extracao=cfg_extr,
            cfg_snn=cfg_snn,
            epocas=EPOCHS,
            taxa_aprendizado=lr,
            num_blocos_residuais=1,
            tamanho_camada_oculta=hidden,
            loss_mode=loss_mode,
            num_passes=num_passes,
            dataset_override=(X, y, rotulos),
            max_samples=None,
            return_stats=True,
        )
        elapsed = time.time() - start
        row = {
            "lr": lr,
            "hidden": hidden,
            "loss_mode": loss_mode,
            "num_passes": num_passes,
            "time_s": round(elapsed, 2),
            "final_loss": stats.get("loss"),
            "final_acc": stats.get("acc"),
        }
        writer.writerow(row)
        f.flush()
        print("Done:", row)
        if row["final_loss"] is not None and row["final_loss"] < 0.01:
            found = cfg
            print("Found config with loss < 0.01:", cfg)
            break

if found:
    print("SUCCESS: found config:", found)
else:
    print("No config reached loss < 0.01 in targeted search")

print("Results written to:", RESULTS_CSV)
