#!/usr/bin/env python3
"""Why does the meeting01 SNN-AE reconstruct worse than the same network without spikes?

The matched control (`ann-ae`) is the SNN-AE with a leaky ReLU where the SNN has a LIF neuron, and
on FSDD fold 0 it is ~0.15 MSE better. This script isolates the candidate causes, one at a time,
on a PyTorch twin of the C++ SNN-AE, trained on the windows a meeting01 run saved.

What the C++ SNN-AE actually is (AutoencoderBuilders.hpp, spec path; `make_snn_cfg` always uses it):

    window (T rows, `direct`: the same 256 values at every step)
      -> Linear 256->64 -> LIF        ("linear:64:leaky" in SNN mode = LifBPTT, spiking)
      -> Linear 64->16                ("linear:16:identity" = NO neuron: the latent is ANALOG)
      -> Linear 16->64 -> LIF
      -> Linear 64->256               ("linear:output:identity": analog, no readout integrator)
    loss: masked MSE over all T rows against the window repeated T times (per-step loss)

LIF (LifBPTT): v = beta*v + I, s = [v > v_th], hard reset v = 0 after a spike (reset path detached
from the gradient, as LifBPTT's dvpost/dvpre = 1 - s), beta = exp(-1/(R*C)), R = 1,
C = -1/ln(alpha), and R, C, v_th trainable at 0.1x the weight lr. ArcTan surrogate, alpha = 2.
Linear init: Kaiming uniform sqrt(6/fan_in), zero bias (kaimingSNNInitializer).

Variants (each changes ONE thing from `snn`):
    snn            the twin of the C++ SNN-AE
    snn_rate       loss on the MEAN over T of the output (a rate readout), target = the window once
    enc_lif_only   decoder hidden neuron is a leaky ReLU
    dec_lif_only   encoder hidden neuron is a leaky ReLU
    snn_integr     output passes a non-spiking leaky integrator (LifBPTT readout_mode)
    snn_soft       soft reset (v -= v_th) instead of v = 0
    ann            both hidden neurons leaky ReLU (the matched control)
    ann_relu / ann_clip / ann_sigmoid
                   the control with ReLU / clamp(x, 0, 1) / sigmoid: is it the BOUNDED output of a
                   spike rate (at most one spike per step) rather than the spike itself?
    ann_clipste    clamp(x, 0, 1) forward, identity backward: bounded range, no dead gradient
    enc_lif_pool   encoder LIF, then the MEAN over T of the latent, then the analog decoder ONCE
                   (`enc_lif_only_rate` runs the decoder at every step and averages its output)
A `_rate` suffix adds the rate readout to any variant (`snn_rate` = `snn` + rate readout,
`enc_lif_only_rate`, ...).
Every variant is scored the same way: masked MSE of its reconstruction against the analog window.
For per-step variants this is the mean over the T rows (what the C++ pipeline reports); the
"time-mean" column also scores the mean over T of their output, without training for it.

Usage:
    software/nn/.venv/bin/python software/nn/scripts/analysis/snn_gap_ablation_torch.py \
        --results DIR --prefix meeting01_loso_fsdd_fold0 --variants snn,snn_rate,ann
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np
import torch
import torch.nn as nn


def load(results: str, prefix: str, split: str):
    w = np.load(os.path.join(results, f"{prefix}_target_{split}_windows.npy")).astype(np.float32)
    with open(os.path.join(results, f"{prefix}_target_{split}_windows_meta.csv")) as f:
        valid = np.array([int(r["valid_length"]) for r in csv.DictReader(f)])
    mask = (np.arange(w.shape[1])[None, :] < valid[:, None]).astype(np.float32)
    return torch.from_numpy(w), torch.from_numpy(mask)


class ArcTanSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        return g * a / (2.0 * (1.0 + (math.pi / 2.0 * a * x) ** 2)), None


class Lif(nn.Module):
    """LifBPTT over a (T, B, F) input. readout=True: emit v (no spike, no reset)."""

    surrogate_alpha = 2.0  # ArcTanSurrogate's default in the C++; --surrogate_alpha overrides it

    def __init__(self, v_th: float, alpha: float, readout: bool = False, soft_reset: bool = False):
        super().__init__()
        self.R = nn.Parameter(torch.tensor(1.0))
        self.C = nn.Parameter(torch.tensor(-1.0 / math.log(alpha)))
        self.v_th = nn.Parameter(torch.tensor(1.0 if readout else v_th))
        self.readout, self.soft = readout, soft_reset
        self.rates = []

    def forward(self, x):
        beta = torch.exp(-1.0 / (self.R.clamp_min(1e-6) * self.C.clamp_min(1e-6)))
        v = torch.zeros_like(x[0])
        out = []
        for t in range(x.shape[0]):
            v = beta * v + x[t]
            if self.readout:
                out.append(v)
                continue
            s = ArcTanSpike.apply(v - self.v_th, self.surrogate_alpha)
            out.append(s)
            v = v - s.detach() * self.v_th if self.soft else v * (1.0 - s.detach())
        y = torch.stack(out)
        if not self.readout and not torch.is_grad_enabled():  # evaluation only
            self.rates.append(float(y.mean()))
        return y


class StepwiseLeaky(nn.Module):
    def forward(self, x):
        return torch.nn.functional.leaky_relu(x, 0.01)


class Clip01(nn.Module):
    """Bounded like a spike rate: clamp(x, 0, 1)."""

    def forward(self, x):
        return x.clamp(0.0, 1.0)


class Clip01Ste(nn.Module):
    """clamp(x, 0, 1) forward, identity gradient: the bounded range without the dead gradient."""

    def forward(self, x):
        return x + (x.clamp(0.0, 1.0) - x).detach()


ANALOG_ACTS = {"ann": StepwiseLeaky, "ann_relu": nn.ReLU, "ann_clip": Clip01,
               "ann_clipste": Clip01Ste, "ann_sigmoid": nn.Sigmoid}


INIT = "kaiming"  # --init: the C++ SNN builder's kaimingSNNInitializer; "xavier" = its ANN builder's


def kaiming_linear(i: int, o: int) -> nn.Linear:
    """Uniform init, zero bias: limit sqrt(6 / fan_in) (kaimingSNNInitializer, every layer the C++
    SNN builder makes) or, with --init xavier, sqrt(6 / (fan_in + fan_out)) (xavierInitializer,
    every layer the C++ ANN builder makes -- so ann-ae is initialised differently from SNN-AE)."""
    lin = nn.Linear(i, o)
    lim = math.sqrt(6.0 / i) if INIT == "kaiming" else math.sqrt(6.0 / (i + o))
    with torch.no_grad():
        lin.weight.uniform_(-lim, lim)
        lin.bias.zero_()
    return lin


class Ae(nn.Module):
    def __init__(self, variant: str, d: int, h: int, z: int, v_th: float, alpha: float):
        super().__init__()
        base = variant.removesuffix("_rate")
        soft = base == "snn_soft"
        enc_lif = base in ("snn", "enc_lif_only", "snn_integr", "snn_soft", "enc_lif_pool")
        dec_lif = base in ("snn", "dec_lif_only", "snn_integr", "snn_soft")
        self.l1, self.l2 = kaiming_linear(d, h), kaiming_linear(h, z)
        self.l3, self.l4 = kaiming_linear(z, h), kaiming_linear(h, d)
        analog = ANALOG_ACTS.get(base, StepwiseLeaky)
        self.a1 = Lif(v_th, alpha, soft_reset=soft) if enc_lif else analog()
        self.a2 = Lif(v_th, alpha, soft_reset=soft) if dec_lif else analog()
        self.integr = Lif(1.0, alpha, readout=True) if base == "snn_integr" else None
        self.rate_readout = variant.endswith("_rate")
        self.pool_latent = base == "enc_lif_pool"

    def forward(self, x):  # x: (T, B, D)
        z = self.l2(self.a1(self.l1(x)))
        if self.pool_latent:
            z = z.mean(dim=0, keepdim=True)  # (1, B, latent): the decoder runs once
        y = self.l4(self.a2(self.l3(z)))
        return self.integr(y) if self.integr is not None else y

    def biophysical(self):
        return [p for n, p in self.named_parameters() if n.split(".")[-1] in ("R", "C", "v_th")]


def masked_mse(recon, x, m):
    """recon (T, B, D) or (B, D); x, m (B, D). Mean over windows of sum(m*diff^2)/sum(m),
    averaged over the T rows when recon has a time axis (the C++ pipeline's number)."""
    if recon.dim() == 3:
        per = (m * (recon - x) ** 2).sum(dim=2) / m.sum(dim=1)  # (T, B)
        return per.mean(dim=0)
    return (m * (recon - x) ** 2).sum(dim=1) / m.sum(dim=1)


def evaluate(model, x, m, steps, chunk=500):
    out_step, out_mean, rates = [], [], []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            xb, mb = x[i:i + chunk], m[i:i + chunk]
            for mod in model.modules():
                if isinstance(mod, Lif):
                    mod.rates = []
            y = model(xb.unsqueeze(0).expand(steps, -1, -1))
            out_mean.append(masked_mse(y.mean(dim=0), xb, mb))
            out_step.append(masked_mse(y, xb, mb) if not model.rate_readout
                            else masked_mse(y.mean(dim=0), xb, mb))
            rates += [r for mod in model.modules() if isinstance(mod, Lif) for r in mod.rates]
    rate = float(np.mean(rates)) if rates else float("nan")
    return float(torch.cat(out_step).mean()), float(torch.cat(out_mean).mean()), rate


def train(variant, args, tr, va, seed):
    torch.manual_seed(seed)
    steps = 1 if variant in ANALOG_ACTS else args.steps
    model = Ae(variant, tr[0].shape[1], args.hidden, args.latent, args.v_th, args.alpha)
    bio = model.biophysical()
    bio_ids = {id(p) for p in bio}
    groups = [{"params": [p for p in model.parameters() if id(p) not in bio_ids], "lr": args.lr}]
    if bio:
        groups.append({"params": bio, "lr": args.lr * 0.1})
    opt = torch.optim.Adam(groups)
    rng = np.random.default_rng(seed)
    best, bad, ep_stop = 9.0, 0, args.epochs
    for ep in range(1, args.epochs + 1):
        for i in rng.permutation(len(tr[0])):
            x, m = tr[0][i:i + 1], tr[1][i:i + 1]
            y = model(x.unsqueeze(0).expand(steps, -1, -1))
            loss = masked_mse(y.mean(dim=0) if model.rate_readout else y, x, m).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = evaluate(model, *va, steps)[0]
        if v < best - 1e-8:
            best, bad = v, 0
        else:
            bad += 1
            if bad >= args.patience:  # the C++ EarlyStoppingCallback: stop, keep the last weights
                ep_stop = ep
                break
    return model, steps, ep_stop, best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--variants", default="snn,snn_rate,enc_lif_only,dec_lif_only,snn_integr,snn_soft,ann")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--v_th", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--init", choices=("kaiming", "xavier"), default="kaiming",
                   help="weight init of every Linear (kaiming = the C++ SNN builder's)")
    p.add_argument("--surrogate_alpha", type=float, default=2.0,
                   help="ArcTan surrogate sharpness; smaller = wider gradient window")
    args = p.parse_args()
    Lif.surrogate_alpha = args.surrogate_alpha
    global INIT
    INIT = args.init
    torch.set_num_threads(args.threads)
    tr, va, te = (load(args.results, args.prefix, s) for s in ("train", "val", "test"))
    print(f"windows train={len(tr[0])} val={len(va[0])} test={len(te[0])}  T={args.steps}  "
          f"v_th={args.v_th} alpha={args.alpha} surrogate_alpha={args.surrogate_alpha} "
          f"init={args.init}", flush=True)
    for variant in args.variants.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            model, steps, ep_stop, best = train(variant, args, tr, va, seed)
            test, test_tmean, rate = evaluate(model, *te, steps)
            print(f"{variant:13s} T={steps:2d} seed={seed}  stopped@{ep_stop:2d}  best_val={best:.4f}  "
                  f"test={test:.4f}  test_time_mean={test_tmean:.4f}  spike_rate={rate:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
