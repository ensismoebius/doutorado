"""Small hyperparameter grid search (quick, uses dataset preloading).

This script preloads the dataset once, subsamples it to `max_samples`, then
runs a short grid of parameter combinations calling
`treinar_classificador_locutor` for a small number of epochs.

It writes results to `experiments/hp_results.csv` in the same folder.
"""

from __future__ import annotations

import csv
import itertools
import os
import sys
import time
from typing import Any

import pathlib
import torch

# Ensure local package root (src/demos/pydemos) is on sys.path so imports like
# `core.configs` resolve when running this script directly.
PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from core.configs import ConfigExtracao, ConfigSNN
from services.identificacao_locutor import (
    treinar_classificador_locutor,
    carregar_modelo_e_rotulos,
)
from services.conjunto_dados import carregar_dataset_janelas

# Quick search configuration (keep small for a fast run)
DATA_DIR = "dados/vozes"
EPOCHS = 3
MAX_SAMPLES = 200  # subsample to speed up runs
RESULTS_CSV = "experiments/hp_results.csv"

# Grid to explore
grid = {
    "lr": [1e-3, 5e-4],
    "hidden": [32, 64],
    "loss_mode": ["rate", "membrane", "van_rossum"],
    "num_passes": [1, 2],
}

# Create experiments folder
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

# Preload dataset once
cfg_extr = ConfigExtracao(
    taxa_amostragem=44100,
    tamanho_janela=512,
    tamanho_passo=256,
    wavelet_base="db4",
    num_bandas=100,
)
cfg_snn = ConfigSNN(passos_por_janela=5, alvo_spikes_por_passo=0.1)
print("Preloading dataset (this may take a moment)")
X, y, rotulos = carregar_dataset_janelas(DATA_DIR, cfg=cfg_extr)
print(f"Loaded dataset: {X.shape[0]} samples; using up to {MAX_SAMPLES}")


def run_one(config: dict[str, Any]):
    # run short training and return final accuracy and loss
    start = time.time()
    model, labels, stats = treinar_classificador_locutor(
        DATA_DIR,
        cfg_extracao=cfg_extr,
        cfg_snn=cfg_snn,
        epocas=EPOCHS,
        taxa_aprendizado=config["lr"],
        num_blocos_residuais=1,
        tamanho_camada_oculta=config["hidden"],
        loss_mode=config["loss_mode"],
        num_passes=config["num_passes"],
        dataset_override=(X, y, rotulos),
        max_samples=MAX_SAMPLES,
        return_stats=True,
    )
    elapsed = time.time() - start
    # As treinar_classificador_locutor não retorna loss/acc históricos, re-run a
    # quick identify on a small slice to measure accuracy
    return {
        "lr": config["lr"],
        "hidden": config["hidden"],
        "loss_mode": config["loss_mode"],
        "num_passes": config["num_passes"],
        "time_s": round(elapsed, 2),
        "rotulos_len": len(labels),
        "final_loss": stats.get("loss"),
        "final_acc": stats.get("acc"),
    }


# Prepare CSV header
header = [
    "lr",
    "hidden",
    "loss_mode",
    "num_passes",
    "time_s",
    "rotulos_len",
    "final_loss",
    "final_acc",
]

# Iterate grid
combinations = list(
    itertools.product(grid["lr"], grid["hidden"], grid["loss_mode"], grid["num_passes"])
)
print(f"Running {len(combinations)} experiments (short)")
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
        res = run_one(cfg)
        writer.writerow(res)
        f.flush()
        print("Done:", res)

print("Grid search finished. Results in:", RESULTS_CSV)
