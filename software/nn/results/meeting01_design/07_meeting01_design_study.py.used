#!/usr/bin/env python3
"""07_meeting01_design_study.py -- the design study: which inherited choices move the SNN-AE's
error, measured on a logged PyTorch twin of the C++ network, crossed and with dispersion.

Why it exists. The paper's design study (Section "Design Study on the Development Folds") asks
three questions: does scoring every simulation step (the loss) hurt, does a spiking decoder hurt,
and does the initializer decide the comparison? Its earlier table ran on windows with the four
defects results_format 10 removed, varied one factor at a time with `--` cells, 3 seeds and no
dispersion, so the reviewers sustained M18 (one factor at a time, confounded, underpowered) and
M19 (loss and decoder changed together), and asked for a logged twin with at least four seeds,
crossed with the decoder and the pooling, a Poisson-coded winner, and the share of saturated
hidden codes printed (review/ruling_visual.md item 2, review_visual.md C294). This script is that
study. It replaces scripts/analysis/snn_gap_ablation_torch.py for the paper (that script keeps
the last weights and plain patience, unlike the harness).

The twin, and where each part comes from (results_format 11 sources):

    window x (256 z-scored samples) -> code: direct = x at every step, poisson = T Bernoulli
        draws of p = (x - min x) / (max x - min x), fixed per window   (Meeting01Encoding.cpp)
    encoder  Linear(256 -> H) -> hidden neuron   (T steps)   ┐ one hidden layer per encoder
             Linear(H -> Z), no neuron (analog latent)        ┘ width but the last (render_encoder_spec)
    readout  lat_mean: time mean of the latent, decoder once  (snn_time_pool "latent", adopted)
    decoder  Linear(Z -> H) -> decoder neuron -> Linear(H -> 256)  (render_decoder_spec)
    LIF      v = beta v + I, s = [v > v_th], hard reset to 0; beta = exp(-1 / (R C)), R = 1,
             C = -1 / ln(alpha), so beta = alpha at the start; R, C, v_th trained, one scalar
             each per layer, at learning_rate_biophysical (LifBPTT.hpp, make_snn_cfg). Gradient:
             ArcTan surrogate alpha/(2(1 + (pi/2 alpha (v - v_th))^2)), alpha = 2; the reset is
             detached on the membrane path (dv_post/dv_pre = 1 - s) and keeps the surrogate on
             the V_th path (dv_post/dV_th = surr * v_pre) -- LifBPTT::compute_grad_step.
    init     Xavier uniform sqrt(6 / (in + out)) or Kaiming uniform sqrt(6 / in), bias 0
             (xavier.hpp, kaiming_snn.hpp); leaky ReLU slope 0.01 (LeakyReLU.hpp)
    training Adam (0.9, 0.999, 1e-8), one window per step, every training window each epoch
             in a fresh order, at most 30 epochs; the profile's rates (learning_rate 1e-3,
             learning_rate_biophysical 1e-4)
    stopping the harness's rule (Meeting01Stopping.hpp, EarlyStoppingCallback.hpp,
             BestWeightsCallback.hpp): L0 = the monitor loss of the elementwise masked mean of
             the training targets; patience (5) counts only once the monitor loss has reached
             (1 - 0.10) L0; a new best needs a drop of more than 1e-8; the network kept is the
             best monitor epoch's.

Design, fixed before the first run (2026-11-09):

    winners  the four SNN-AE winners of the results_format 11 pre-validation (GridUnesp job
             5317827), read from its selection manifests: two per development fold, FSDD and
             EEGMMIDB fold 0. Three are Poisson-coded, one direct (FSDD run 1).
    seeds    42, 43, 44, 45. A seed sets the initial weights, the window order and the Poisson
             draws, so the cells of one winner and seed share all three (paired by seed).
    A  readout x decoder (3 x 2), LIF encoder at the winner's code, T, v_th and alpha, Xavier:
         readout  step      the loss scores every step's output; the decoder runs every step
                  out_mean  the loss scores the time mean of the output; decoder every step
                  lat_mean  the latent's time mean; the decoder runs once (a LIF decoder gets
                            that mean as a constant current for T steps) -- adopted with lrelu
         decoder  lif | lrelu
    B  encoder hidden neuron x initializer (5 x 2), lat_mean readout, lrelu decoder:
         neuron   lif (the winner's code) | lrelu (the ann-ae control) | sigmoid |
                  clamp to [0, 1] | clamp to [0, 1] with an identity gradient (clamp_ste)
         init     xavier (adopted) | kaiming
       A's (lat_mean, lrelu) cell is B's (lif, xavier) cell: trained once.
       An analog encoder reads the analog window once (T = 1), as ann-ae does. The two EEGMMIDB
       winners have the same widths, so their analog cells are the same networks: trained once.
    score    the masked MSE of the network's ONE reconstruction (the time mean of the output
             when the decoder runs every step) on the fold's test windows: what the adopted
             pipeline reports, so every cell is scored alike. A `step` cell also logs the mean
             over T of its per-step errors (what the harness reported when it trained so).
    logged   per epoch (train loss, monitor loss, armed, best) in logs/<name>.jsonl; per
             training the best epoch, the stop reason, test / validation / training error,
             the encoder's and a LIF decoder's code shares (below), trained v_th and beta.
    shares   per (window, hidden neuron) on the test windows: a LIF neuron's firing rate over
             T, a bounded analog code's value; "low" <= 0.01, "high" >= 0.99 (the harness's
             hidden_silent_share / hidden_always_share). Not comparable across T: at T = 19,
             "high" means 19 spikes out of 19 (review/ruling_visual_round4.md R3).
    contrasts  per winner, paired by seed, mean difference and a Student-t 95% interval over
             the 4 seeds (df 3): the training-to-training spread on fixed windows, not a
             generalisation over speakers or subjects.

Reading limits stated before the run: two development folds, fold 0 only; the test block of
fold 0 is already out of the headline (DEVELOPMENT_FOLDS) and its validation block is fold 1's
test block, already out too (DESIGN_INFORMED_FOLDS), so this study adds no new design-informed
fold. A twin is checked against the harness at the two points both ran (adopted SNN-AE, ANN-AE
control); the table prints both.

Usage (from software/nn; 07_meeting01_design_study.sh calls --run and --report):
    07_meeting01_design_study.py --plan [--prevalidation D]
    07_meeting01_design_study.py --run --out-root O [--prevalidation D] [--jobs N]
        O/rows/<name>.json, O/logs/<name>.jsonl (a row whose source revision is not this one is
        refused, not skipped)
    07_meeting01_design_study.py --report --out-root O --data-dir X [--prevalidation D]
        O/design_trainings.csv, O/design_cells.csv, O/design_contrasts.csv,
        X/paper_design.tex, X/paper_design_shares.tex, X/paper_design_stats.tex
    07_meeting01_design_study.py --placeholders --data-dir X
    07_meeting01_design_study.py --self-test

Refuses (no fallbacks): a pre-validation of another results_format, a winner manifest that is
missing, ambiguous or not an SNN-AE's, a widths list it cannot render, a row from another source
revision, a missing training at report time.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys
import tempfile
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from meeting01_results import (  # noqa: E402
    DATASET_SHORT,
    DEVELOPMENT_FOLDS,
    RESULTS_FORMAT,
    ResultsError,
    split_manifest,
)

RUN_TAG = "meeting01_prevalidation"
DEFAULT_PREVALIDATION = "results/meeting01_prevalidation"
DATASETS = ("fsdd", "eegmmidb")
RUNS = (1, 2)
SEEDS = (42, 43, 44, 45)
EPOCHS = 30
PATIENCE = 5
MIN_SKILL = 0.10
MIN_DELTA = 1e-8
LR = 1e-3
LR_BIO = 1e-4
SURROGATE_ALPHA = 2.0
LEAKY_SLOPE = 0.01
SHARE_LOW, SHARE_HIGH = 0.01, 0.99
EVAL_CHUNK = 250

READOUTS = ("step", "out_mean", "lat_mean")
DECODERS = ("lif", "lrelu")
NEURONS = ("lif", "lrelu", "sigmoid", "clamp", "clamp_ste")
INITS = ("xavier", "kaiming")
ANALOG = ("lrelu", "sigmoid", "clamp", "clamp_ste")
BOUNDED = ("lif", "sigmoid", "clamp", "clamp_ste")
ADOPTED = ("lif", "lat_mean", "lrelu", "xavier")

LABEL = {
    "step": "loss on every step", "out_mean": "loss on time mean", "lat_mean": "latent mean",
    "lif": "LIF", "lrelu": "leaky ReLU", "sigmoid": "sigmoid", "clamp": "clamp $[0,1]$",
    "clamp_ste": "clamp, id.\\ grad.", "xavier": "Xavier", "kaiming": "Kaiming",
}

ROW_FIELDS = [
    "name", "dataset", "run", "widths", "encoding", "time_steps", "neuron", "readout", "decoder",
    "init", "seed", "source_revision", "epochs_run", "best_epoch", "armed_epoch", "stop_reason",
    "trivial_loss", "best_monitor", "test_mse", "test_mse_per_step", "val_mse", "train_mse",
    "enc_low_share", "enc_high_share", "enc_mean_code", "dec_low_share", "dec_high_share",
    "lif_v_th_mean", "lif_beta_mean", "wall_s",
]


# ------------------------------------------------------------------------- winners and tasks
def _prevalidation_family_dir(prevalidation: pathlib.Path) -> pathlib.Path:
    return prevalidation / "snn-ae"


def read_winners(prevalidation: pathlib.Path) -> list[dict]:
    """The SNN-AE winners of the pre-validation: one per (development dataset, run)."""
    d = _prevalidation_family_dir(prevalidation)
    out = []
    for ds in DATASETS:
        folds = sorted(DEVELOPMENT_FOLDS.get(ds, ()))
        if len(folds) != 1:
            raise ResultsError(f"{ds}: the design study runs on the one development fold, found {folds}")
        fold = folds[0]
        split_manifest(d, RUN_TAG, ds, fold)  # checks results_format
        with (d / f"{RUN_TAG}_{ds}_fold{fold}_comparative_metrics.csv").open(encoding="utf-8") as f:
            metrics = [r for r in csv.DictReader(f) if r["split"] == "test"]
        for run in RUNS:
            found = [p for p in d.glob(f"{RUN_TAG}_{ds}_fold{fold}_*_run{run}_model_selection_manifest.json")
                     if "_ann-ae_" not in p.name]
            if len(found) != 1:
                raise ResultsError(f"{ds} fold {fold} run {run}: {len(found)} SNN-AE selection manifests "
                                   f"in {d}, expected 1: {[p.name for p in found]}")
            man = json.loads(found[0].read_text(encoding="utf-8"))
            if man.get("results_format") != RESULTS_FORMAT:
                raise ResultsError(f"{found[0].name}: results_format {man.get('results_format')}, "
                                   f"this script reads {RESULTS_FORMAT}")
            sel = man.get("selected") or {}
            need = ("encoder_widths", "encoding", "time_steps", "v_th", "alpha")
            if any(k not in sel for k in need) or sel.get("architecture") != "dense":
                raise ResultsError(f"{found[0].name}: not a dense SNN-AE winner (selected = {sel})")
            widths = [int(w) for w in sel["encoder_widths"]]
            if len(widths) < 2:
                raise ResultsError(f"{found[0].name}: widths {widths} have no hidden layer; "
                                   "render_encoder_spec always keeps one")
            if sel["encoding"] not in ("direct", "poisson"):
                raise ResultsError(f"{found[0].name}: encoding {sel['encoding']!r} is not one this "
                                   "twin implements (direct, poisson)")
            harness = {}
            for model in ("snn-ae", "ann-ae"):
                rows = [r for r in metrics if r["model"] == model and int(r["run"]) == run]
                if len(rows) != 1:
                    raise ResultsError(f"{ds} run {run}: {len(rows)} {model} test rows, expected 1")
                harness[model] = rows[0]
            out.append({
                "dataset": ds, "fold": fold, "run": run, "widths": widths,
                "encoding": sel["encoding"], "time_steps": int(sel["time_steps"]),
                "v_th": float(sel["v_th"]), "alpha": float(sel["alpha"]),
                "seed_of_search": int(man["seed"]),
                "harness_snn_test": float(harness["snn-ae"]["mse"]),
                "harness_ann_test": float(harness["ann-ae"]["mse"]),
                "harness_snn_low": float(harness["snn-ae"]["hidden_silent_share"]),
                "harness_snn_high": float(harness["snn-ae"]["hidden_always_share"]),
            })
    return out


def winner_key(w: dict) -> str:
    return f"{w['dataset']}_r{w['run']}"


def widths_key(widths) -> str:
    return "x".join(str(int(v)) for v in widths)


def cells() -> list[tuple[str, str, str, str]]:
    """(neuron, readout, decoder, init) of one winner, A then B, the shared cell once."""
    out = [("lif", r, d, "xavier") for r in READOUTS for d in DECODERS]
    for n in NEURONS:
        for i in INITS:
            c = (n, "lat_mean", "lrelu", i)
            if c not in out:
                out.append(c)
    return out


def task_name(w: dict, cell, seed: int) -> str:
    n, r, d, i = cell
    if n in ANALOG:  # the same network for every winner with these widths
        return f"{w['dataset']}_w{widths_key(w['widths'])}_{n}_{i}_s{seed}"
    return f"{winner_key(w)}_{n}_{r}_{d}_{i}_s{seed}"


def build_tasks(winners: list[dict]) -> list[dict]:
    seen, out = set(), []
    for w in winners:
        for cell in cells():
            for seed in SEEDS:
                name = task_name(w, cell, seed)
                if name in seen:
                    continue
                seen.add(name)
                n, r, d, i = cell
                spiking = n == "lif" or d == "lif"
                out.append({"name": name, "winner": w, "neuron": n, "readout": r, "decoder": d,
                            "init": i, "seed": seed,
                            "steps": w["time_steps"] if spiking else 1,
                            "encoding": w["encoding"] if n == "lif" else "direct"})
    # longest first, so the pool does not end on one long training
    out.sort(key=lambda t: -(t["steps"] * (2 if t["decoder"] == "lif" else 1)))
    return out


# ------------------------------------------------------------------------- the twin (torch)
def _torch():
    import torch  # imported lazily: --plan, --report and --placeholders do not need it
    return torch


def make_twin_classes():
    torch = _torch()
    nn = torch.nn

    class ArcTanSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, diff):
            ctx.save_for_backward(diff)
            return (diff > 0).to(diff.dtype)

        @staticmethod
        def backward(ctx, g):
            (diff,) = ctx.saved_tensors
            a = SURROGATE_ALPHA
            return g * a / (2.0 * (1.0 + (math.pi / 2.0 * a * diff) ** 2))

    class Lif(nn.Module):
        """LifBPTT over (T, B, F): v = beta v + I, s = [v > v_th], v <- 0 after a spike."""

        def __init__(self, v_th: float, alpha: float):
            super().__init__()
            self.R = nn.Parameter(torch.tensor(1.0))
            self.C = nn.Parameter(torch.tensor(-1.0 / math.log(alpha)))
            self.v_th = nn.Parameter(torch.tensor(float(v_th)))

        def beta(self):
            return torch.exp(-1.0 / (self.R.clamp_min(1e-6) * self.C.clamp_min(1e-6)))

        def forward(self, cur):
            return LifUnroll.apply(cur, self.beta(), self.v_th)

        def forward_reference(self, cur):
            """The same neuron through autograd, one step at a time: what the self-test checks
            LifUnroll's hand-written backward against (it is ~10x slower)."""
            beta = self.beta()
            v = torch.zeros_like(cur[0])
            out = []
            for t in range(cur.shape[0]):
                v = beta * v + cur[t]
                s = ArcTanSpike.apply(v - self.v_th)
                # The reset: forward v * (1 - s); backward 1 - s on the membrane path (detached
                # spike) and surr * v_pre on the V_th path, as LifBPTT::compute_grad_step.
                s_th = ArcTanSpike.apply(v.detach() - self.v_th)
                v = v * (1.0 - s.detach()) - v.detach() * (s_th - s_th.detach())
                out.append(s)
            return torch.stack(out)

    def _surr(diff):
        a = SURROGATE_ALPHA
        return a / (2.0 * (1.0 + (math.pi / 2.0 * a * diff) ** 2))

    class LifUnroll(torch.autograd.Function):
        """The T-step LIF in one autograd node, its backward LifBPTT::compute_grad_step's:

            g_post[t] = g_pre[t+1] * beta                  (v_pre[t+1] = beta v_post[t] + I[t+1])
            g_pre[t]  = g_out[t] * surr[t] + g_post[t] * (1 - s[t])
            dL/dI[t]  = g_pre[t]
            dL/dbeta  = sum_t g_pre[t+1] * v_post[t]
            dL/dV_th  = sum_t ( -g_out[t] * surr[t] + g_post[t] * surr[t] * v_pre[t] )
        """

        @staticmethod
        def forward(ctx, cur, beta, v_th):
            steps = cur.shape[0]
            # cur may be an expanded view (direct input: one current for every step)
            v_pre = torch.empty(cur.shape, dtype=cur.dtype)
            spikes = torch.empty(cur.shape, dtype=cur.dtype)
            v = torch.zeros(cur.shape[1:], dtype=cur.dtype)
            for t in range(steps):
                v = beta * v + cur[t]
                v_pre[t] = v
                s = (v > v_th).to(v.dtype)
                spikes[t] = s
                v = v * (1.0 - s)
            ctx.save_for_backward(v_pre, spikes, beta, v_th)
            return spikes

        @staticmethod
        def backward(ctx, g_out):
            v_pre, spikes, beta, v_th = ctx.saved_tensors
            steps = v_pre.shape[0]
            g_cur = torch.empty_like(v_pre)
            g_next = torch.zeros_like(v_pre[0])  # dL/dv_pre[t+1]
            d_beta = torch.zeros((), dtype=v_pre.dtype)
            d_vth = torch.zeros((), dtype=v_pre.dtype)
            for t in range(steps - 1, -1, -1):
                surr = _surr(v_pre[t] - v_th)
                g_post = g_next * beta
                if t + 1 < steps:
                    d_beta = d_beta + (g_next * v_pre[t] * (1.0 - spikes[t])).sum()
                g_pre = g_out[t] * surr + g_post * (1.0 - spikes[t])
                d_vth = d_vth + (-g_out[t] * surr + g_post * surr * v_pre[t]).sum()
                g_cur[t] = g_pre
                g_next = g_pre
            return g_cur, d_beta, d_vth

    class Leaky(nn.Module):
        def forward(self, x):
            return torch.nn.functional.leaky_relu(x, LEAKY_SLOPE)

    class Clamp(nn.Module):
        def forward(self, x):
            return x.clamp(0.0, 1.0)

    class ClampSte(nn.Module):
        def forward(self, x):
            return x + (x.clamp(0.0, 1.0) - x).detach()

    def make_neuron(kind, v_th, alpha):
        return {"lif": lambda: Lif(v_th, alpha), "lrelu": Leaky, "sigmoid": nn.Sigmoid,
                "clamp": Clamp, "clamp_ste": ClampSte}[kind]()

    def linear(i, o, init, gen):
        lin = nn.Linear(i, o)
        lim = math.sqrt(6.0 / (i + o)) if init == "xavier" else math.sqrt(6.0 / i)
        with torch.no_grad():
            lin.weight.copy_((torch.rand(o, i, generator=gen) * 2.0 - 1.0) * lim)
            lin.bias.zero_()
        return lin

    class Twin(nn.Module):
        def __init__(self, widths, d, neuron, readout, decoder, init, v_th, alpha, steps, gen):
            super().__init__()
            hidden, z = list(widths[:-1]), widths[-1]
            self.readout, self.steps, self.decoder_kind = readout, steps, decoder
            self.enc_lin, self.enc_act = nn.ModuleList(), nn.ModuleList()
            prev = d
            for h in hidden:
                self.enc_lin.append(linear(prev, h, init, gen))
                self.enc_act.append(make_neuron(neuron, v_th, alpha))
                prev = h
            self.lat = linear(prev, z, init, gen)
            self.dec_lin, self.dec_act = nn.ModuleList(), nn.ModuleList()
            prev = z
            for h in reversed(hidden):
                self.dec_lin.append(linear(prev, h, init, gen))
                self.dec_act.append(make_neuron(decoder, v_th, alpha))
                prev = h
            self.out = linear(prev, d, init, gen)
            self.capture = None

        def forward(self, x):
            """x: (T_in, B, D) with T_in = 1 (direct; expanded to T after the first Linear,
            the same current at every step) or T (poisson). Returns (T_out, B, D)."""
            h = x
            for k, (lin, act) in enumerate(zip(self.enc_lin, self.enc_act)):
                cur = lin(h)
                if cur.shape[0] == 1 and self.steps > 1:
                    cur = cur.expand(self.steps, -1, -1)
                h = act(cur)
                if self.capture is not None:
                    self.capture["enc"].append(h.detach())
            zt = self.lat(h)
            if self.readout == "lat_mean":
                zt = zt.mean(dim=0, keepdim=True)
            for lin, act in zip(self.dec_lin, self.dec_act):
                cur = lin(zt)
                if self.decoder_kind == "lif" and cur.shape[0] == 1 and self.steps > 1:
                    cur = cur.expand(self.steps, -1, -1)
                zt = act(cur)
                if self.capture is not None:
                    self.capture["dec"].append(zt.detach())
            return self.out(zt)

        def lifs(self):
            return [m for m in self.modules() if isinstance(m, Lif)]

        def biophysical(self):
            return [p for m in self.lifs() for p in (m.R, m.C, m.v_th)]

    return {"Twin": Twin, "Lif": Lif, "ArcTanSpike": ArcTanSpike, "LifUnroll": LifUnroll}


def masked_mse(recon, x, m):
    """recon (B, D), x, m (B, D) -> per-window sum(m * diff^2) / sum(m), shape (B,)."""
    return (m * (recon - x) ** 2).sum(dim=1) / m.sum(dim=1)


def monitor_loss(y, x, m, readout):
    """The loss a cell trains on: per-step (mean over T of each step's error) for `step`, the
    error of the time mean otherwise. Per window, shape (B,)."""
    if readout == "step" and y.shape[0] > 1:
        per_step = (m.unsqueeze(0) * (y - x.unsqueeze(0)) ** 2).sum(dim=2) / m.sum(dim=1).unsqueeze(0)
        return per_step.mean(dim=0)  # (T, B) -> (B,)
    return masked_mse(y.mean(dim=0), x, m)


def trivial_loss(fit_x, fit_m, mon_x, mon_m) -> float:
    """L0: the elementwise masked mean of the fit targets, scored on the monitor windows one
    window at a time (input_independent_monitor_loss with batch_size 1). A position no fit
    window covers is predicted as 0."""
    num = (fit_x * fit_m).sum(axis=0, dtype=np.float64)
    den = fit_m.sum(axis=0, dtype=np.float64)
    mean = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    d = (mean[None, :] - mon_x) ** 2 * mon_m
    per = d.sum(axis=1) / np.maximum(mon_m.sum(axis=1), 1e-30)
    per = np.where(mon_m.sum(axis=1) > 0, per, 0.0)
    return float(per.mean())


class Stopper:
    """EarlyStoppingCallback + BestWeightsCallback: arm at loss <= arm_below (before the
    improvement test), count bad epochs only while armed, keep the best epoch."""

    def __init__(self, arm_below: float, patience: int = PATIENCE, min_delta: float = MIN_DELTA):
        self.arm_below, self.patience, self.min_delta = arm_below, patience, min_delta
        self.best, self.best_epoch, self.bad, self.armed_epoch = math.inf, 0, 0, 0
        self.stop = False

    def update(self, epoch: int, loss: float) -> bool:
        """True when this epoch is a new best (the caller keeps its weights)."""
        if self.armed_epoch == 0 and loss <= self.arm_below:
            self.armed_epoch = epoch
        if loss < self.best - self.min_delta:
            self.best, self.best_epoch, self.bad = loss, epoch, 0
            return True
        if self.armed_epoch:
            self.bad += 1
            if self.bad >= self.patience:
                self.stop = True
        return False


def poisson_code(x: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
    """(N, D) windows -> (T, N, D) spikes: p = (x - min) / max(max - min, 1e-6) per window,
    clamped to [0, 1], one Bernoulli draw per step (encode_sample)."""
    lo = x.min(axis=1, keepdims=True)
    rng_ = np.maximum(x.max(axis=1, keepdims=True) - lo, 1e-6)
    p = np.clip((x - lo) / rng_, 0.0, 1.0).astype(np.float32)
    u = rng.random((steps,) + x.shape, dtype=np.float32)
    return (u < p[None]).astype(np.float32)


def load_split(prevalidation: pathlib.Path, ds: str, fold: int, split: str):
    d = _prevalidation_family_dir(prevalidation)
    base = d / f"{RUN_TAG}_{ds}_fold{fold}_target_{split}_windows"
    x = np.load(f"{base}.npy").astype(np.float32)
    with open(f"{base}_meta.csv", encoding="utf-8") as f:
        valid = np.array([int(r["valid_length"]) for r in csv.DictReader(f)])
    if len(valid) != len(x):
        raise ResultsError(f"{base}: {len(x)} windows, {len(valid)} meta rows")
    m = (np.arange(x.shape[1])[None, :] < valid[:, None]).astype(np.float32)
    return x, m


def _shares(codes, kind: str, steps: int):
    """(low, high, mean) over (window, neuron) of a hidden code, or NaN where not bounded."""
    if not codes or kind not in BOUNDED:
        return float("nan"), float("nan"), float("nan")
    torch = _torch()
    # LIF: firing rate over T; analog bounded: the value itself (T = 1)
    vals = torch.cat([c.mean(dim=0).reshape(-1) for c in codes])
    return (float((vals <= SHARE_LOW).float().mean()), float((vals >= SHARE_HIGH).float().mean()),
            float(vals.mean()))


def train_one(task: dict, prevalidation: str, out_root: str, source_revision: str) -> dict:
    torch = _torch()
    torch.set_num_threads(1)
    classes = make_twin_classes()
    w, seed = task["winner"], task["seed"]
    pre = pathlib.Path(prevalidation)
    data = {s: load_split(pre, w["dataset"], w["fold"], s) for s in ("train", "val", "test")}
    steps, enc = task["steps"], task["encoding"]

    def code(split, idx):
        x = data[split][0]
        if enc == "poisson":
            rng = np.random.default_rng([seed, 2, idx])
            return torch.from_numpy(poisson_code(x, steps, rng))
        return torch.from_numpy(x).unsqueeze(0)

    inputs = {s: code(s, k) for k, s in enumerate(("train", "val", "test"))}
    xs = {s: torch.from_numpy(data[s][0]) for s in data}
    ms = {s: torch.from_numpy(data[s][1]) for s in data}

    gen = torch.Generator().manual_seed(seed)
    model = classes["Twin"](w["widths"], xs["train"].shape[1], task["neuron"], task["readout"],
                            task["decoder"], task["init"], w["v_th"], w["alpha"], steps, gen)
    bio = model.biophysical()
    bio_ids = {id(p) for p in bio}
    groups = [{"params": [p for p in model.parameters() if id(p) not in bio_ids], "lr": LR}]
    if bio:
        groups.append({"params": bio, "lr": LR_BIO})
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)

    def evaluate(split, score=False):
        """monitor loss (the cell's), the one-reconstruction score, the per-step score."""
        model.eval()
        mon, one, per_step = [], [], []
        with torch.no_grad():
            for i in range(0, xs[split].shape[0], EVAL_CHUNK):
                y = model(inputs[split][:, i:i + EVAL_CHUNK])
                x, m = xs[split][i:i + EVAL_CHUNK], ms[split][i:i + EVAL_CHUNK]
                mon.append(monitor_loss(y, x, m, task["readout"]))
                if score:
                    one.append(masked_mse(y.mean(dim=0), x, m))
                    per_step.append(monitor_loss(y, x, m, "step"))
        model.train()
        f = lambda v: float(torch.cat(v).mean()) if v else float("nan")  # noqa: E731
        return f(mon), f(one), f(per_step)

    l0 = trivial_loss(data["train"][0], data["train"][1], data["val"][0], data["val"][1])
    stopper = Stopper((1.0 - MIN_SKILL) * l0)
    best_state = None
    order_rng = np.random.default_rng([seed, 1])
    log_path = pathlib.Path(out_root) / "logs" / f"{task['name']}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    epochs_run, stop_reason = 0, "max_epochs"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps({"type": "start", "name": task["name"], "trivial_loss": l0,
                              "arm_below": stopper.arm_below, "steps": steps, "encoding": enc,
                              "source_revision": source_revision}) + "\n")
        for epoch in range(1, EPOCHS + 1):
            total = 0.0
            for i in order_rng.permutation(xs["train"].shape[0]):
                y = model(inputs["train"][:, i:i + 1])
                loss = monitor_loss(y, xs["train"][i:i + 1], ms["train"][i:i + 1],
                                    task["readout"]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item()
            mon = evaluate("val")[0]
            if not math.isfinite(mon):
                stop_reason, epochs_run = "non_finite", epoch
                log.write(json.dumps({"type": "epoch", "epoch": epoch, "monitor": mon}) + "\n")
                break
            if stopper.update(epoch, mon):
                best_state = copy.deepcopy(model.state_dict())
            epochs_run = epoch
            log.write(json.dumps({"type": "epoch", "epoch": epoch,
                                  "train_loss": total / xs["train"].shape[0], "monitor": mon,
                                  "armed": stopper.armed_epoch > 0, "best_epoch": stopper.best_epoch,
                                  "bad": stopper.bad}) + "\n")
            log.flush()
            if stopper.stop:
                stop_reason = "patience"
                break
        if best_state is None:
            raise ResultsError(f"{task['name']}: no finite monitor epoch, no best network to keep")
        model.load_state_dict(best_state)
        model.capture = {"enc": [], "dec": []}
        test_mon, test_one, test_step = evaluate("test", score=True)
        captured = model.capture
        model.capture = None
        val_one = evaluate("val", score=True)[1]
        train_one_ = evaluate("train", score=True)[1]
        enc_low, enc_high, enc_mean = _shares(captured["enc"], task["neuron"], steps)
        dec_low, dec_high, _ = _shares(captured["dec"], task["decoder"] if task["decoder"] == "lif"
                                       else "none", steps)
        lifs = model.lifs()
        row = {
            "name": task["name"], "dataset": w["dataset"], "run": w["run"],
            "widths": widths_key(w["widths"]), "encoding": enc, "time_steps": steps,
            "neuron": task["neuron"], "readout": task["readout"], "decoder": task["decoder"],
            "init": task["init"], "seed": seed, "source_revision": source_revision,
            "epochs_run": epochs_run, "best_epoch": stopper.best_epoch,
            "armed_epoch": stopper.armed_epoch, "stop_reason": stop_reason, "trivial_loss": l0,
            "best_monitor": stopper.best, "test_mse": test_one,
            "test_mse_per_step": test_step if task["readout"] == "step" and steps > 1 else float("nan"),
            "val_mse": val_one, "train_mse": train_one_,
            "enc_low_share": enc_low, "enc_high_share": enc_high, "enc_mean_code": enc_mean,
            "dec_low_share": dec_low, "dec_high_share": dec_high,
            "lif_v_th_mean": float(np.mean([float(m.v_th.detach()) for m in lifs])) if lifs else float("nan"),
            "lif_beta_mean": float(np.mean([float(m.beta().detach()) for m in lifs])) if lifs else float("nan"),
            "wall_s": time.time() - t0,
        }
        log.write(json.dumps({"type": "end", **row}) + "\n")
    rows_dir = pathlib.Path(out_root) / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    tmp = rows_dir / f"{task['name']}.json.tmp"
    tmp.write_text(json.dumps(row), encoding="utf-8")
    tmp.replace(rows_dir / f"{task['name']}.json")
    return row


def _worker(args):
    task, prevalidation, out_root, rev = args
    try:
        row = train_one(task, prevalidation, out_root, rev)
        return task["name"], row, None
    except Exception as exc:  # reported, and the run then fails: no row is invented
        return task["name"], None, f"{type(exc).__name__}: {exc}"


def run(prevalidation: pathlib.Path, out_root: pathlib.Path, jobs: int) -> int:
    rev = os.environ.get("MEETING01_GIT_COMMIT", "")
    if not rev:
        raise ResultsError("MEETING01_GIT_COMMIT is not set: run through 07_meeting01_design_study.sh, "
                           "which resolves the source revision")
    tasks = build_tasks(read_winners(prevalidation))
    todo = []
    for t in tasks:
        p = out_root / "rows" / f"{t['name']}.json"
        if p.is_file():
            old = json.loads(p.read_text(encoding="utf-8")).get("source_revision")
            if old != rev:
                raise ResultsError(f"{p}: made by {old}, this run is {rev}; move {out_root} aside")
            continue
        todo.append(t)
    print(f"[design] {len(tasks)} trainings, {len(tasks) - len(todo)} already done, "
          f"{len(todo)} to run on {jobs} processes", flush=True)
    failed = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(jobs, maxtasksperchild=1) as pool:
        args = [(t, str(prevalidation), str(out_root), rev) for t in todo]
        for k, (name, row, err) in enumerate(pool.imap_unordered(_worker, args), 1):
            if err:
                failed.append(name)
                print(f"[design] {time.strftime('%F %T')} FAILED {name}: {err}", flush=True)
            else:
                print(f"[design] {time.strftime('%F %T')} {k}/{len(todo)} {name}: test "
                      f"{row['test_mse']:.4f} epochs {row['epochs_run']} armed@{row['armed_epoch']} "
                      f"{row['wall_s']:.0f}s", flush=True)
    if failed:
        print(f"[design] {len(failed)} trainings failed: {failed}", file=sys.stderr)
        return 1
    return 0


# ------------------------------------------------------------------------- report
def _t_quantile_975(df: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
    if df not in table:
        raise ResultsError(f"no t quantile tabulated for df={df}")
    return table[df]


def paired_contrast(a: dict, b: dict) -> dict:
    """a, b: seed -> value. Mean of a - b over the common seeds, Student-t 95% interval."""
    seeds = sorted(set(a) & set(b))
    if set(a) != set(b) or len(seeds) < 2:
        raise ResultsError(f"paired contrast needs the same seeds on both sides: {sorted(a)} vs {sorted(b)}")
    d = np.array([a[s] - b[s] for s in seeds])
    half = _t_quantile_975(len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return {"n": len(d), "mean": float(d.mean()), "lo": float(d.mean() - half),
            "hi": float(d.mean() + half)}


def load_rows(out_root: pathlib.Path) -> list[dict]:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out_root / "rows").glob("*.json"))]
    revs = {r["source_revision"] for r in rows}
    if len(revs) > 1:
        raise ResultsError(f"rows from {len(revs)} source revisions in {out_root}: {sorted(revs)}")
    return rows


def cell_values(rows: list[dict], winners: list[dict]) -> dict:
    """(winner_key, cell) -> {seed: row}; analog rows are shared by winners of equal widths."""
    by_name = {r["name"]: r for r in rows}
    out = {}
    for w in winners:
        for cell in cells():
            got = {}
            for seed in SEEDS:
                name = task_name(w, cell, seed)
                if name not in by_name:
                    raise ResultsError(f"training {name} is missing: rerun 07_meeting01_design_study.sh "
                                       "with RESUME=1")
                got[seed] = by_name[name]
            out[(winner_key(w), cell)] = got
    return out


def _fmt(vals) -> str:
    """mean with sd in units of the last digit shown: 0.773(19) = 0.773 +- 0.019."""
    v = np.array(vals, dtype=float)
    if v.size == 0 or not np.all(np.isfinite(v)):
        return "$-$"
    mean, sd = float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0
    sd_m = int(round(sd * 1000))
    return f"{mean:.3f}({sd_m})" if sd_m < 1000 else f"{mean:.3f}({sd:.2f})"


def _share_fmt(lo, hi) -> str:
    lo, hi = np.array(lo, float), np.array(hi, float)
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return "$-$"
    return f"{lo.mean():.2f}/{hi.mean():.2f}"


def _winner_header(winners: list[dict]) -> list[str]:
    by_ds = {}
    for w in winners:
        by_ds.setdefault(w["dataset"], []).append(w)
    head1 = " & " + " & ".join(f"\\multicolumn{{{len(v)}}}{{c}}{{{DATASET_SHORT.get(ds, ds)}}}"
                               for ds, v in by_ds.items()) + " \\\\"
    head2 = "Cell & " + " & ".join(
        f"{'P' if w['encoding'] == 'poisson' else 'D'}{w['time_steps']}" for w in winners) + " \\\\"
    return [head1, head2]


def design_table(winners, cv) -> str:
    cols = "l" + "c" * len(winners)
    lines = ["% auto-generated by 07_meeting01_design_study.py -- do not edit",
             f"\\begin{{tabular}}{{{cols}}}", "\\toprule", *_winner_header(winners), "\\midrule",
             f"\\multicolumn{{{len(winners) + 1}}}{{l}}{{\\emph{{A. Readout $\\times$ decoder (LIF "
             "encoder, Xavier)}}}} \\\\"]

    def row(label, cell):
        vals = [_fmt([r["test_mse"] for r in cv[(winner_key(w), cell)].values()]) for w in winners]
        return f"{label} & " + " & ".join(vals) + " \\\\"

    for r in READOUTS:
        for d in DECODERS:
            mark = "$^\\dagger$" if ("lif", r, d, "xavier") == ADOPTED else ""
            lines.append(row(f"{LABEL[r]}, {LABEL[d]} dec.{mark}", ("lif", r, d, "xavier")))
    lines += ["\\midrule",
              f"\\multicolumn{{{len(winners) + 1}}}{{l}}{{\\emph{{B. Encoder neuron $\\times$ "
              "initializer (latent mean, leaky ReLU dec.)}}}} \\\\"]
    for n in NEURONS:
        for i in INITS:
            mark = "$^\\dagger$" if (n, "lat_mean", "lrelu", i) == ADOPTED else ""
            lines.append(row(f"{LABEL[n]}, {LABEL[i]}{mark}", (n, "lat_mean", "lrelu", i)))
    lines += ["\\midrule",
              f"\\multicolumn{{{len(winners) + 1}}}{{l}}{{\\emph{{C++ harness, one seed}}}} \\\\",
              "SNN-AE (adopted) & " + " & ".join(f"{w['harness_snn_test']:.3f}" for w in winners) + " \\\\",
              "ANN-AE (control) & " + " & ".join(f"{w['harness_ann_test']:.3f}" for w in winners) + " \\\\",
              "\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def shares_table(winners, cv) -> str:
    cols = "l" + "c" * len(winners)
    lines = ["% auto-generated by 07_meeting01_design_study.py -- do not edit",
             f"\\begin{{tabular}}{{{cols}}}", "\\toprule", *_winner_header(winners), "\\midrule"]
    for n in BOUNDED:
        for i in INITS:
            cell = (n, "lat_mean", "lrelu", i)
            vals = [_share_fmt([r["enc_low_share"] for r in cv[(winner_key(w), cell)].values()],
                               [r["enc_high_share"] for r in cv[(winner_key(w), cell)].values()])
                    for w in winners]
            lines.append(f"{LABEL[n]}, {LABEL[i]} & " + " & ".join(vals) + " \\\\")
    lines += ["\\midrule", "C++ harness SNN-AE & " + " & ".join(
        f"{w['harness_snn_low']:.2f}/{w['harness_snn_high']:.2f}" for w in winners) + " \\\\",
              "\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def contrasts(winners, cv) -> list[dict]:
    out = []

    def seedmap(cell, w, field="test_mse"):
        return {s: r[field] for s, r in cv[(winner_key(w), cell)].items()}

    for w in winners:
        k = winner_key(w)

        def add(name, a, b):
            c = paired_contrast(a, b)
            out.append({"winner": k, "contrast": name, **c})

        L = lambda r, d: seedmap(("lif", r, d, "xavier"), w)  # noqa: E731
        for d in DECODERS:
            add(f"loss step - out_mean | {d} dec", L("step", d), L("out_mean", d))
            add(f"pool out_mean - lat_mean | {d} dec", L("out_mean", d), L("lat_mean", d))
        for r in READOUTS:
            add(f"decoder lif - lrelu | {r}", L(r, "lif"), L(r, "lrelu"))
        inter = {s: (L("step", "lif")[s] - L("step", "lrelu")[s]) - (L("out_mean", "lif")[s] - L("out_mean", "lrelu")[s])
                 for s in SEEDS}
        add("interaction loss x decoder", inter, {s: 0.0 for s in SEEDS})
        for i in INITS:
            ctrl = seedmap(("lrelu", "lat_mean", "lrelu", i), w)
            for n in NEURONS:
                if n != "lrelu":
                    add(f"{n} - lrelu | {i}", seedmap((n, "lat_mean", "lrelu", i), w), ctrl)
        for n in NEURONS:
            add(f"kaiming - xavier | {n}", seedmap((n, "lat_mean", "lrelu", "kaiming"), w),
                seedmap((n, "lat_mean", "lrelu", "xavier"), w))
        twin = seedmap(ADOPTED, w)
        add("twin adopted - harness SNN-AE", twin, {s: w["harness_snn_test"] for s in SEEDS})
        add("twin lrelu xavier - harness ANN-AE",
            seedmap(("lrelu", "lat_mean", "lrelu", "xavier"), w), {s: w["harness_ann_test"] for s in SEEDS})
    return out


def write_tex(data_dir: pathlib.Path, design: str, shares: str, status: str, n_trainings: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "paper_design.tex").write_text(design, encoding="utf-8")
    (data_dir / "paper_design_shares.tex").write_text(shares, encoding="utf-8")
    macros = {"designDataStatus": status, "designSeeds": str(len(SEEDS)),
              "designTrainings": n_trainings, "designEpochs": str(EPOCHS),
              "designMinSkill": f"{MIN_SKILL:.2f}"}
    lines = ["% auto-generated by 07_meeting01_design_study.py -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (data_dir / "paper_design_stats.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def placeholder_winners() -> list[dict]:
    out = []
    for ds, enc, t in (("fsdd", "direct", 0), ("fsdd", "poisson", 0), ("eegmmidb", "poisson", 0),
                       ("eegmmidb", "poisson", 0)):
        out.append({"dataset": ds, "encoding": enc, "time_steps": "", "harness_snn_test": float("nan"),
                    "harness_ann_test": float("nan"), "harness_snn_low": float("nan"),
                    "harness_snn_high": float("nan"), "run": len([o for o in out if o["dataset"] == ds]) + 1})
    return out


def write_placeholders(data_dir: pathlib.Path) -> None:
    ws = placeholder_winners()
    empty = {}
    for w in ws:
        for cell in cells():
            empty[(winner_key(w), cell)] = {s: {"test_mse": float("nan"), "enc_low_share": float("nan"),
                                                "enc_high_share": float("nan")} for s in SEEDS}
    design = design_table(ws, empty).replace("nan", "$-$")
    shares = shares_table(ws, empty).replace("nan/nan", "$-$")
    write_tex(data_dir, design, shares, "placeholder", "$-$")


def report(prevalidation: pathlib.Path, out_root: pathlib.Path, data_dir: pathlib.Path) -> None:
    winners = read_winners(prevalidation)
    rows = load_rows(out_root)
    cv = cell_values(rows, winners)
    expected = len(build_tasks(winners))
    if len(rows) != expected:
        raise ResultsError(f"{len(rows)} rows in {out_root}/rows, the design has {expected}")
    with (out_root / "design_trainings.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=ROW_FIELDS, lineterminator="\n")
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: r["name"]))
    cell_fields = ["winner", "neuron", "readout", "decoder", "init", "n", "test_mean", "test_sd",
                   "train_mean", "val_mean", "test_per_step_mean", "enc_low", "enc_high",
                   "dec_low", "dec_high", "armed", "epochs_mean"]
    with (out_root / "design_cells.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cell_fields, lineterminator="\n")
        wr.writeheader()
        for (k, (n, r, d, i)), got in sorted(cv.items()):
            g = list(got.values())
            mean = lambda fld: float(np.mean([x[fld] for x in g]))  # noqa: E731
            wr.writerow({"winner": k, "neuron": n, "readout": r, "decoder": d, "init": i, "n": len(g),
                         "test_mean": mean("test_mse"),
                         "test_sd": float(np.std([x["test_mse"] for x in g], ddof=1)),
                         "train_mean": mean("train_mse"), "val_mean": mean("val_mse"),
                         "test_per_step_mean": mean("test_mse_per_step"),
                         "enc_low": mean("enc_low_share"), "enc_high": mean("enc_high_share"),
                         "dec_low": mean("dec_low_share"), "dec_high": mean("dec_high_share"),
                         "armed": sum(1 for x in g if x["armed_epoch"] > 0),
                         "epochs_mean": mean("epochs_run")})
    con = contrasts(winners, cv)
    with (out_root / "design_contrasts.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["winner", "contrast", "n", "mean", "lo", "hi"],
                           lineterminator="\n")
        wr.writeheader()
        wr.writerows(con)
    write_tex(data_dir, design_table(winners, cv), shares_table(winners, cv), "results", str(len(rows)))
    for c in con:
        flag = "" if c["lo"] <= 0 <= c["hi"] else "  *"
        print(f"{c['winner']:12s} {c['contrast']:40s} {c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]{flag}")


# ------------------------------------------------------------------------- self-test
def self_test() -> int:
    failures = []
    torch = _torch()
    cl = make_twin_classes()

    # 1. LIF staircase: I = 0.6, v_th = 1, beta = 0.5 -> v 0.6, 0.9, 1.05 (spike, reset) ... period 3
    lif = cl["Lif"](1.0, 0.5)
    s = lif(torch.full((9, 1, 1), 0.6))
    if s.reshape(-1).tolist() != [0, 0, 1, 0, 0, 1, 0, 0, 1]:
        failures.append(f"LIF staircase: {s.reshape(-1).tolist()}")
    if abs(float(lif.beta().detach()) - 0.5) > 1e-6:
        failures.append(f"beta(alpha=0.5) = {float(lif.beta().detach())}")

    # 2. surrogate at threshold = alpha / 2; reset path gradients (membrane 1 - s, V_th surr * v_pre)
    d = torch.zeros(1, requires_grad=True)
    cl["ArcTanSpike"].apply(d).sum().backward()
    if abs(float(d.grad) - SURROGATE_ALPHA / 2) > 1e-6:
        failures.append(f"ArcTan surrogate at 0: {float(d.grad)}")
    lif2 = cl["Lif"](1.0, 0.5)
    cur = torch.tensor([[[1.2]], [[0.0]]], requires_grad=True)  # spike at t0, then v1 = beta*0
    out = lif2(cur)
    # output at t1 = spike(beta * v_post0 - v_th); v_post0 = 0, so d out1 / d cur0 = surr(-1) * beta * (1 - 1) = 0
    out[1].sum().backward()
    if abs(float(cur.grad[0])) > 1e-9:
        failures.append(f"hard reset leaks a membrane gradient: {float(cur.grad[0])}")
    surr_m1 = SURROGATE_ALPHA / (2 * (1 + (math.pi / 2 * SURROGATE_ALPHA * -1.0) ** 2))
    # d out1 / d v_th = -surr(v1 - v_th) [direct] + surr(v1 - v_th) * beta * surr(v0 - v_th) * v0 [reset path]
    surr_02 = SURROGATE_ALPHA / (2 * (1 + (math.pi / 2 * SURROGATE_ALPHA * 0.2) ** 2))
    want = -surr_m1 + surr_m1 * 0.5 * surr_02 * 1.2
    if abs(float(lif2.v_th.grad) - want) > 1e-6:
        failures.append(f"V_th gradient {float(lif2.v_th.grad)}, want {want} (LifBPTT reset path)")

    # 2b. LifUnroll's hand-written backward = autograd through the step-by-step reference,
    #     for the input current, R, C and V_th (random currents, some steps spiking)
    for trial in range(3):
        g = torch.Generator().manual_seed(trial)
        cur0 = torch.rand(7, 3, 5, generator=g) * 1.6 - 0.2
        wts = torch.randn(7, 3, 5, generator=g)
        grads = []
        for ref in (False, True):
            cell = cl["Lif"](0.8, 0.85)
            c = cur0.clone().requires_grad_(True)
            o = cell.forward_reference(c) if ref else cell(c)
            (o * wts).sum().backward()
            grads.append((o.detach(), c.grad, cell.R.grad, cell.C.grad, cell.v_th.grad))
        names = ("spikes", "dL/dI", "dL/dR", "dL/dC", "dL/dV_th")
        for nm, a_, b_ in zip(names, *grads):
            if not torch.allclose(a_, b_, rtol=1e-4, atol=1e-6):
                failures.append(f"LifUnroll {nm} differs from the step-by-step reference (trial {trial})")
    # an expanded (direct-input) current gives the same gradient as a materialised one
    cell = cl["Lif"](0.8, 0.85)
    base = torch.rand(1, 2, 4, generator=torch.Generator().manual_seed(9)).requires_grad_(True)
    cell(base.expand(6, -1, -1)).sum().backward()
    g_exp = base.grad.clone()
    base.grad = None
    cell.zero_grad()
    cell(base.expand(6, -1, -1).contiguous()).sum().backward()
    if not torch.allclose(g_exp, base.grad):
        failures.append("LifUnroll on an expanded current differs from a contiguous one")

    # 3. masked MSE and the monitor losses
    x = torch.tensor([[1.0, 2.0, 3.0]])
    m = torch.tensor([[1.0, 1.0, 0.0]])
    y = torch.tensor([[[0.0, 2.0, 9.0]], [[2.0, 2.0, 9.0]]])  # (T=2, B=1, D=3)
    if abs(float(monitor_loss(y, x, m, "step")) - 0.5) > 1e-6:
        failures.append(f"per-step loss {float(monitor_loss(y, x, m, 'step'))}, want 0.5")
    if abs(float(monitor_loss(y, x, m, "out_mean"))) > 1e-6:
        failures.append("time-mean loss of a symmetric pair must be 0")

    # 4. L0: fit mean [1, 2], monitor [[0, 2]] -> (1 + 0) / 2 = 0.5; uncovered position predicted 0
    fx = np.array([[0.0, 2.0, 5.0], [2.0, 2.0, 7.0]])
    fm = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    l0 = trivial_loss(fx, fm, np.array([[0.0, 2.0, 3.0]]), np.array([[1.0, 1.0, 1.0]]))
    if abs(l0 - (1.0 + 0.0 + 9.0) / 3) > 1e-9:
        failures.append(f"L0 {l0}, want {(1 + 0 + 9) / 3}")

    # 5. stopping: arms at 0.89 (epoch 5), stops 5 bad epochs later, keeps epoch 5
    st = Stopper(0.9)
    seq = [1.0, 0.99, 0.98, 0.95, 0.89, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1]
    stop_at = None
    for e, v in enumerate(seq, 1):
        st.update(e, v)
        if st.stop:
            stop_at = e
            break
    if (st.armed_epoch, st.best_epoch, stop_at) != (5, 5, 10):
        failures.append(f"stopping: armed {st.armed_epoch} best {st.best_epoch} stop {stop_at}, want 5 5 10")
    st2 = Stopper(0.5)
    for e in range(1, 31):
        st2.update(e, 1.0)
    if st2.stop or st2.armed_epoch:
        failures.append("a plateau run must train every epoch")

    # 6. Poisson code: min never fires, max always fires, a middle value at its probability
    xw = np.array([[0.0, 1.0, 4.0]], dtype=np.float32)
    sp = poisson_code(xw, 4000, np.random.default_rng(0))
    r = sp.mean(axis=0)[0]
    if r[0] != 0.0 or r[2] != 1.0 or abs(r[1] - 0.25) > 0.03:
        failures.append(f"poisson rates {r}, want 0, ~0.25, 1")

    # 7. init limits, bias 0, the decoder mirrors the encoder
    gen = torch.Generator().manual_seed(1)
    tw = cl["Twin"]([59, 16], 256, "lif", "lat_mean", "lrelu", "xavier", 0.5, 0.9, 4, gen)
    if float(tw.enc_lin[0].weight.abs().max()) > math.sqrt(6 / (256 + 59)) + 1e-6 or \
            float(tw.enc_lin[0].bias.abs().max()) != 0.0:
        failures.append("xavier init out of its limit or bias not 0")
    if [l.out_features for l in tw.dec_lin] != [59] or tw.out.out_features != 256 or tw.lat.out_features != 16:
        failures.append("decoder does not mirror [59, 16]")
    tk = cl["Twin"]([59, 16], 256, "lif", "lat_mean", "lrelu", "kaiming", 0.5, 0.9, 4, gen)
    if float(tk.enc_lin[0].weight.abs().max()) <= math.sqrt(6 / (256 + 59)):
        failures.append("kaiming init no wider than xavier on 256 -> 59")
    y = tw(torch.zeros(1, 2, 256))
    if tuple(y.shape) != (1, 2, 256):
        failures.append(f"lat_mean twin output {tuple(y.shape)}, want (1, 2, 256)")
    tl = cl["Twin"]([59, 16], 256, "lif", "lat_mean", "lif", "xavier", 0.5, 0.9, 4, gen)
    if tuple(tl(torch.zeros(1, 2, 256)).shape) != (4, 2, 256):
        failures.append("lat_mean with a LIF decoder must run the decoder T steps")

    # 8. the design: 15 cells per winner, analog cells shared by equal widths
    if len(cells()) != 15 or ADOPTED not in cells():
        failures.append(f"{len(cells())} cells per winner, want 15 with the adopted one")
    fake = [{"dataset": "fsdd", "run": 1, "widths": [59, 16], "encoding": "direct", "time_steps": 28},
            {"dataset": "fsdd", "run": 2, "widths": [35, 16], "encoding": "poisson", "time_steps": 19},
            {"dataset": "eegmmidb", "run": 1, "widths": [65, 64], "encoding": "poisson", "time_steps": 31},
            {"dataset": "eegmmidb", "run": 2, "widths": [65, 64], "encoding": "poisson", "time_steps": 19}]
    n_tasks = len(build_tasks(fake))
    want_tasks = len(SEEDS) * (4 * 7 + 3 * 8)
    if n_tasks != want_tasks:
        failures.append(f"{n_tasks} trainings, want {want_tasks} (4 winners x 7 spiking cells + "
                        "3 widths x 8 analog cells, x 4 seeds)")
    t_an = [t for t in build_tasks(fake) if t["neuron"] == "sigmoid"][0]
    if t_an["steps"] != 1 or t_an["encoding"] != "direct":
        failures.append("an analog encoder must read the window once, directly")

    # 9. formatting and contrasts
    if _fmt([0.760, 0.786]) != "0.773(18)":
        failures.append(f"_fmt: {_fmt([0.760, 0.786])}")
    c = paired_contrast({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}, {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0})
    if (c["mean"], c["lo"], c["hi"]) != (1.0, 1.0, 1.0):
        failures.append(f"constant paired difference: {c}")
    try:
        paired_contrast({1: 1.0, 2: 2.0}, {1: 1.0, 3: 2.0})
        failures.append("a contrast over different seeds was accepted")
    except ResultsError:
        pass

    # 10. report on synthetic rows: table strings, harness rows, refusal of a missing training
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for w in fake:
            w.update(harness_snn_test=0.786, harness_ann_test=0.716, harness_snn_low=0.46,
                     harness_snn_high=0.40, fold=0)
        rows = []
        for t in build_tasks(fake):
            rows.append({"name": t["name"], "test_mse": 0.7 + 0.01 * (t["seed"] - 42),
                         "enc_low_share": 0.3, "enc_high_share": 0.02})
        cv = cell_values(rows, fake)
        tex = design_table(fake, cv)
        for needle in ("D28 & P19 & P31 & P19", "latent mean, leaky ReLU dec.$^\\dagger$",
                       "LIF, Xavier$^\\dagger$", "0.715(13)", "SNN-AE (adopted) & 0.786",
                       "\\multicolumn{2}{c}{FSDD}"):
            if needle not in tex:
                failures.append(f"design table lacks {needle!r}")
        sh = shares_table(fake, cv)
        if "0.30/0.02" not in sh or "C++ harness SNN-AE & 0.46/0.40" not in sh:
            failures.append("shares table content")
        try:
            cell_values(rows[1:], fake)
            failures.append("a missing training was accepted")
        except ResultsError:
            pass
        write_placeholders(tmp)
        ptxt = (tmp / "paper_design.tex").read_text(encoding="utf-8")
        if "nan" in ptxt or "$-$" not in ptxt:
            failures.append("placeholder table shows nan or no dashes")
        if "\\newcommand{\\designDataStatus}{placeholder}" not in (tmp / "paper_design_stats.tex").read_text():
            failures.append("placeholder status macro")

    for f in failures:
        print("[self-test] FAIL:", f)
    if not failures:
        print("[self-test] all design-study checks passed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--placeholders", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    ap.add_argument("--prevalidation", default=DEFAULT_PREVALIDATION)
    ap.add_argument("--out-root")
    ap.add_argument("--data-dir")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    try:
        if a.self_test:
            return self_test()
        if a.placeholders:
            if not a.data_dir:
                ap.error("--placeholders needs --data-dir")
            write_placeholders(pathlib.Path(a.data_dir))
            return 0
        pre = pathlib.Path(a.prevalidation)
        if a.plan:
            winners = read_winners(pre)
            for w in winners:
                print(f"{winner_key(w)}: widths {w['widths']} {w['encoding']} T={w['time_steps']} "
                      f"v_th={w['v_th']:.3f} alpha={w['alpha']:.3f}  harness SNN {w['harness_snn_test']:.3f} "
                      f"ANN {w['harness_ann_test']:.3f}")
            tasks = build_tasks(winners)
            print(f"{len(tasks)} trainings ({sum(1 for t in tasks if t['steps'] > 1)} spiking)")
            return 0
        if not a.out_root:
            ap.error("--run and --report need --out-root")
        if a.run:
            if a.jobs < 1:
                ap.error("--jobs must be >= 1")
            return run(pre, pathlib.Path(a.out_root), a.jobs)
        if not a.data_dir:
            ap.error("--report needs --data-dir")
        report(pre, pathlib.Path(a.out_root), pathlib.Path(a.data_dir))
        return 0
    except ResultsError as exc:
        print(f"[design] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
