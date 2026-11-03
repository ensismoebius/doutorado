#!/usr/bin/env python3
"""Train the PyTorch twin of the C++ LSTM-AE on the SAME windows the C++ run trained on.

`lstm_ae_pytorch_parity_gtest` proves forward/loss/gradients agree for one step. It cannot say
why the C++ LSTM-AE sits at loss ~1.0 during training. This script trains the independent
PyTorch implementation, same data, same shape (direct encoding: the window repeated T times,
re-cut into S = T*M/F frames of F samples; masked MSE; Adam; batch 1), and prints the loss
curve per epoch for a few variants. If PyTorch also stays at ~1.0, the cause is the setup
(init, budget, lr), not the C++ code.

Input: the *_target_{train,val}_windows.npy + *_meta.csv a meeting01 run writes next to its
results (valid_length gives the mask).

Usage:
    software/nn/.venv/bin/python software/nn/scripts/analysis/compare_lstm_ae_torch.py \
        --results DIR --prefix meeting01_loso_fsdd_fold0 --hidden 64 --layers 1 --epochs 15
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testing"))
from lstm_ae_torch import LstmAe, cpp_style_init, masked_mse  # noqa: E402


def load(results: str, prefix: str, split: str):
    w = np.load(os.path.join(results, f"{prefix}_target_{split}_windows.npy")).astype(np.float32)
    with open(os.path.join(results, f"{prefix}_target_{split}_windows_meta.csv")) as f:
        valid = [int(r["valid_length"]) for r in csv.DictReader(f)]
    return w, np.array(valid)


def frames(w: np.ndarray, valid: int, t: int, f: int):
    """Direct encoding: window repeated T times, flattened, cut into frames of F samples."""
    m = w.shape[0]
    mask = np.zeros(m, dtype=np.float32)
    mask[:valid] = 1.0
    x = np.tile(w, t).reshape(-1, f)
    k = np.tile(mask, t).reshape(-1, f)
    return torch.from_numpy(x), torch.from_numpy(k)


def init_model(model, init, rng):
    """cpp = N(0, 0.05) like the C++ constructor; std<x> = the same with another std;
    ortho = orthogonal recurrent U and W, Glorot-uniform Linear weights, forget bias 1 (Saxe et
    al. 2014; Jozefowicz et al. 2015); default = PyTorch's own init."""
    if init == "default":
        return
    if init == "cpp" or init.startswith("std"):
        import lstm_ae_torch as t
        saved = t.INIT_STD
        if init.startswith("std"):
            t.INIT_STD = float(init[3:])
        try:
            cpp_style_init(model, rng)
        finally:
            t.INIT_STD = saved
        return
    if init == "ortho":
        with torch.no_grad():
            for lstm in (model.enc, model.dec):
                for l in range(model.layers):
                    for g in range(4):
                        h = model.h
                        for nm in ("weight_ih_l", "weight_hh_l"):
                            w = getattr(lstm, f"{nm}{l}")
                            blk = w[g * h:(g + 1) * h]
                            torch.nn.init.orthogonal_(blk)
                    getattr(lstm, f"bias_ih_l{l}").zero_()
                    getattr(lstm, f"bias_hh_l{l}").zero_()
                    getattr(lstm, f"bias_ih_l{l}")[model.h:2 * model.h] = 1.0
            for lin in (model.enc_proj, model.dec_expand, model.out_proj):
                torch.nn.init.xavier_uniform_(lin.weight)
                lin.bias.zero_()
        return
    raise ValueError(f"unknown init '{init}'")


def run(name, args, data, init, lr, clip, seed):
    torch.manual_seed(seed)
    d = args.frame
    model = LstmAe(d, args.hidden, args.latent, args.layers)
    rng = np.random.default_rng(seed)
    init_model(model, init, rng)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    (trw, trv), (vaw, vav) = data
    tr = [frames(w, v, args.steps, d) for w, v in zip(trw, trv)]
    va = [frames(w, v, args.steps, d) for w, v in zip(vaw, vav)]
    order = np.random.default_rng(seed)
    left, best, final = None, 9.0, None
    for ep in range(1, args.epochs + 1):
        tl = 0.0
        for i in order.permutation(len(tr)):
            x, k = tr[i]
            opt.zero_grad()
            loss = masked_mse(model(x), x, k)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tl += loss.item()
        tl /= len(tr)
        with torch.no_grad():
            vl = float(np.mean([masked_mse(model(x), x, k).item() for x, k in va]))
        best = min(best, vl)
        if left is None and tl < 0.98:
            left = ep
        final = (tl, vl)
    print(f"{name:28s} seed={seed}  left_plateau_epoch={left}  final_train={final[0]:.4f}  "
          f"final_val={final[1]:.4f}  best_val={best:.4f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--frame", type=int, default=8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--variants", default="cpp:1e-3,default:1e-3,cpp:1e-2",
                   help="comma list of init:lr[:clip], init in {cpp, std<x>, ortho, default}")
    p.add_argument("--seeds", default="43", help="comma list of seeds")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    data = (load(args.results, args.prefix, "train"), load(args.results, args.prefix, "val"))
    for v in args.variants.split(","):
        init, lr, *rest = v.split(":")
        clip = float(rest[0]) if rest else 0.0
        for seed in [int(x) for x in args.seeds.split(",")]:
            run(f"{init} lr={lr} clip={clip}", args, data, init, float(lr), clip, seed)


if __name__ == "__main__":
    main()
