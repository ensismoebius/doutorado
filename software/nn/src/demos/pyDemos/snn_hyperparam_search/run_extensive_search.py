"""Staged hyperparameter search.

Stage 1: run a reasonably sized grid for few epochs on a subsample.
Stage 2: pick top-k candidates and run longer training; stop early if loss<0.01.

Writes results to:
 - experiments/ext_hp_stage1.csv
 - experiments/ext_hp_stage2.csv

Be mindful: this can take time depending on grid and epochs.
"""

from __future__ import annotations
import csv
import itertools
import os
import sys
import time
import pathlib
from typing import Any

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "voice_biometrics_snn_py"
sys.path.insert(0, str(PKG_ROOT))

from core.configs import ConfigExtracao, ConfigSNN
from services.identificacao_locutor import treinar_classificador_locutor
from services.conjunto_dados import carregar_dataset_janelas

# Params
DATA_DIR = "dados/vozes"
STAGE1_EPOCHS = 8
STAGE2_EPOCHS = 80
MAX_SAMPLES_STAGE1 = 800
TOP_K = 5

RESULTS_DIR = os.path.join(os.path.dirname(__file__))
STAGE1_CSV = os.path.join(RESULTS_DIR, "ext_hp_stage1.csv")
STAGE2_CSV = os.path.join(RESULTS_DIR, "ext_hp_stage2.csv")

# Grid (reasonable breadth)
grid = {
    "lr": [1e-3, 5e-4, 1e-4],
    "hidden": [64, 128, 256],
    "loss_mode": ["van_rossum", "membrane", "mse_vector"],
    "num_passes": [1, 2, 5],
}

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Preloading dataset... (stage1 will subsample)")
cfg_extr = ConfigExtracao(
    taxa_amostragem=44100,
    tamanho_janela=512,
    tamanho_passo=256,
    wavelet_base="db4",
    num_bandas=100,
)
cfg_snn = ConfigSNN(passos_por_janela=5, alvo_spikes_por_passo=0.1)
X, y, rotulos = carregar_dataset_janelas(DATA_DIR, cfg=cfg_extr)
print(f"Loaded {X.shape[0]} samples; using up to {MAX_SAMPLES_STAGE1} for stage1")

# Stage 1
combinations = list(
    itertools.product(grid["lr"], grid["hidden"], grid["loss_mode"], grid["num_passes"])
)
print(f"Stage1: {len(combinations)} experiments, {STAGE1_EPOCHS} epochs each")

header = [
    "lr",
    "hidden",
    "loss_mode",
    "num_passes",
    "time_s",
    "final_loss",
    "final_acc",
]
with open(STAGE1_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    for lr, hidden, loss_mode, num_passes in combinations:
        cfg = {
            "lr": lr,
            "hidden": hidden,
            "loss_mode": loss_mode,
            "num_passes": num_passes,
        }
        print("Running stage1:", cfg)
        start = time.time()
        model, labels, stats = treinar_classificador_locutor(
            DATA_DIR,
            cfg_extracao=cfg_extr,
            cfg_snn=cfg_snn,
            epocas=STAGE1_EPOCHS,
            taxa_aprendizado=lr,
            num_blocos_residuais=1,
            tamanho_camada_oculta=hidden,
            loss_mode=loss_mode,
            num_passes=num_passes,
            dataset_override=(X, y, rotulos),
            max_samples=MAX_SAMPLES_STAGE1,
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
        print("Stage1 done:", row)

# Read stage1 results and pick top-K by final_loss
import pandas as pd

stage1 = pd.read_csv(STAGE1_CSV)
stage1 = stage1.dropna(subset=["final_loss"]) if not stage1.empty else stage1
stage1_sorted = stage1.sort_values(by=["final_loss"]) if not stage1.empty else stage1
candidates = []
for _, r in stage1_sorted.head(TOP_K).iterrows():
    candidates.append(
        {
            "lr": float(r["lr"]),
            "hidden": int(r["hidden"]),
            "loss_mode": r["loss_mode"],
            "num_passes": int(r["num_passes"]),
        }
    )

print("Top candidates:")
for c in candidates:
    print(c)

# Stage 2: longer runs for top candidates
found = None
with open(STAGE2_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "lr",
            "hidden",
            "loss_mode",
            "num_passes",
            "time_s",
            "final_loss",
            "final_acc",
        ],
    )
    writer.writeheader()
    for cfg in candidates:
        print("Running stage2 for", cfg)
        start = time.time()
        model, labels, stats = treinar_classificador_locutor(
            DATA_DIR,
            cfg_extracao=cfg_extr,
            cfg_snn=cfg_snn,
            epocas=STAGE2_EPOCHS,
            taxa_aprendizado=cfg["lr"],
            num_blocos_residuais=1,
            tamanho_camada_oculta=cfg["hidden"],
            loss_mode=cfg["loss_mode"],
            num_passes=cfg["num_passes"],
            dataset_override=(X, y, rotulos),
            max_samples=None,
            return_stats=True,
        )
        elapsed = time.time() - start
        row = {
            "lr": cfg["lr"],
            "hidden": cfg["hidden"],
            "loss_mode": cfg["loss_mode"],
            "num_passes": cfg["num_passes"],
            "time_s": round(elapsed, 2),
            "final_loss": stats.get("loss"),
            "final_acc": stats.get("acc"),
        }
        writer.writerow(row)
        f.flush()
        print("Stage2 done:", row)
        if row["final_loss"] is not None and row["final_loss"] < 0.01:
            found = cfg
            print("Found config with loss < 0.01:", cfg)
            break

if found:
    print("SUCCESS: found config:", found)
else:
    print("No config reached loss < 0.01")

print("Extensive search finished. Stage1:", STAGE1_CSV, "Stage2:", STAGE2_CSV)
