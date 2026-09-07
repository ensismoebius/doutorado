#!/usr/bin/env python3
"""Generate INDEPENDENT cross-check fixtures for SpikeTimeLoss and PoissonLatentLayer.

IMPORTANT — read before trusting these numbers the way you would pytorch_refs.npz or
optimizer_refs.npz: this is NOT a diff against an external library. No external package
implements either operation (confirmed during the investigation that produced this file:
snnTorch's `ce_temporal_loss` is classification cross-entropy against a class label, not
MSE between two spike trains' first-spike-times; `spikegen.rate` is Bernoulli image
encoding, not this project's Poisson-VAE reparameterization). What follows is a SECOND,
INDEPENDENT re-derivation of the same documented formula in NumPy, written directly from
the prose/formula comments in the C++ headers rather than transliterated line-by-line from
the C++ source. It catches "the C++ implementation doesn't match its own documented
formula" bugs; it does NOT catch "the documented formula itself is wrong" bugs the way an
external-library parity test would. Do not describe results from this fixture as "ground
truth" or "parity" — call it a cross-check / independent re-derivation.

Covered:
  SpikeTimeLoss        (include/layers/losses/SpikeTimeLoss.hpp)
  PoissonLatentLayer    (include/layers/spiking/PoissonLatentLayer.hpp) -- deterministic
                        parts only (rate=softplus, KL term, backward). The stochastic
                        training-mode sample s~Poisson(rate*T) is NOT covered here:
                        std::poisson_distribution and np.random.poisson are different
                        algorithms with different RNG streams and will not agree
                        bit-for-bit even from "the same seed" -- see the file docstring
                        on PoissonLatentLayer.hpp's forward() for the sampling formula
                        this deliberately skips.

Run (developer step; requires only numpy, already in this project's venv):
    software/nn/.venv/bin/python software/nn/scripts/testing/gen_independent_refs.py

Output: software/nn/src/core/layers/tests/fixtures/independent_refs.npz
"""
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "src", "core", "layers", "tests", "fixtures",
                   "independent_refs.npz")
OUT = os.path.normpath(OUT)

A = {}  # name -> float32 ndarray


def put(name, arr):
    arr = np.asarray(arr)
    # Preserve integer dtype (e.g. np.int64 case counts / dims) -- only float arrays get
    # cast to float32. Casting everything to float32 unconditionally corrupts int64 arrays
    # (their raw bytes get reinterpreted as int64 on the C++ side and read as garbage).
    if not np.issubdtype(arr.dtype, np.floating):
        A[name] = np.ascontiguousarray(arr)
    else:
        A[name] = np.ascontiguousarray(arr.astype(np.float32))


# ── SpikeTimeLoss: independent re-derivation ──────────────────────────────────
# From SpikeTimeLoss.hpp's documented formula:
#   first_spike_time(b,f) = min{ t : spikes[t,b,f] > 0.5 }, or T if no such t.
#   loss = mean_{b,f} (pred_time - tgt_time)^2
#   dL/d_input[t,b,f] = 2*(pred_t - tgt_t)/(B*F)  if t == pred_time, else 0
# Implemented here with vectorized NumPy boolean-mask argmax, not a transliteration of
# the C++ triple-nested loop.
def independent_first_spike_times(spikes_tbf, T):
    """spikes_tbf: (T, B, F) binary array. Returns (B, F) float array of first-spike
    times in [0, T] (T = no spike, per the documented penalty convention)."""
    fired = spikes_tbf > 0.5
    ever_fired = fired.any(axis=0)                 # (B, F)
    first_idx = fired.argmax(axis=0)                # first True along T; 0 if none fired
    times = np.where(ever_fired, first_idx, T)       # penalize no-spike entries with T
    return times.astype(np.float32)


def independent_spike_time_loss(pred_tbf, tgt_tbf, T):
    B, F = pred_tbf.shape[1], pred_tbf.shape[2]
    pred_t = independent_first_spike_times(pred_tbf, T)
    tgt_t = independent_first_spike_times(tgt_tbf, T)
    diff = pred_t - tgt_t
    loss = float(np.mean(diff ** 2))

    grad = np.zeros_like(pred_tbf, dtype=np.float32)
    scale = 2.0 / (B * F)
    g_per_bf = scale * diff
    has_spike = pred_t < T
    bs, fs = np.where(has_spike)
    ts = pred_t[has_spike].astype(np.int64)
    grad[ts, bs, fs] = g_per_bf[has_spike]
    return pred_t, tgt_t, loss, grad


SPIKETIME_CASES = [(5, 2, 3), (6, 3, 2)]  # (T, B, F)
put("spiketime_num", np.array([len(SPIKETIME_CASES)], np.int64))
for idx, (T, B, F) in enumerate(SPIKETIME_CASES):
    rng = np.random.default_rng(1000 + idx)
    pred = (rng.random((T, B, F)) > 0.6).astype(np.float32)   # sparse-ish random spikes
    tgt = (rng.random((T, B, F)) > 0.6).astype(np.float32)

    pred_t, tgt_t, loss, grad = independent_spike_time_loss(pred, tgt, T)

    p = f"spiketime_{idx}_"
    put(p + "dims", np.array([T, B, F], np.int64))
    put(p + "input", pred.reshape(T * B, F))     # time-major (T*B, F)
    put(p + "target", tgt.reshape(T * B, F))
    put(p + "loss", np.array([[loss]], np.float32))
    put(p + "grad_input", grad.reshape(T * B, F))


# ── PoissonLatentLayer: independent re-derivation (deterministic parts only) ──
# From PoissonLatentLayer.hpp's documented formulas:
#   rate = softplus(z)                          (numerically stable form)
#   kl   = mean_{b,f}( prior - rate + rate*log(rate/prior + 1e-8) ) * beta_kl
#   dL/dz = grad_output * (1/T) * sigmoid(z)
#           + [beta_kl>0] * (1 - prior/(rate+1e-8)) * sigmoid(z) * beta_kl/(B*F)
#   inference (requires_grad=false): output = rate directly.
# Implemented from scratch here (independent stable-softplus branch selection, not a
# transliteration of the C++ per-element loop).
def independent_softplus(z):
    # Stable: log(1+exp(z)) for z<=20, else z (matches the documented "numerically
    # stable form", re-derived independently rather than copied from the C++ branch).
    return np.where(z > 20.0, z, np.log1p(np.exp(z)))


def independent_sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def independent_poisson_latent(z, prior_rate, beta_kl, T, grad_output):
    rate = independent_softplus(z)
    B, F = z.shape

    kl = 0.0
    if beta_kl > 0.0:
        kl_terms = prior_rate - rate + rate * np.log(rate / prior_rate + 1e-8)
        kl = beta_kl * float(np.mean(kl_terms))

    sig = independent_sigmoid(z)
    grad = grad_output * (1.0 / T) * sig
    if beta_kl > 0.0:
        kl_grad = (1.0 - prior_rate / (rate + 1e-8)) * sig * (beta_kl / (B * F))
        grad = grad + kl_grad

    return rate, kl, grad


POISSON_CASES = [(3, 4, 1, 0.1, 1.0), (2, 5, 3, 0.1, 0.0)]  # (B, F, T, prior_rate, beta_kl)
put("poisson_num", np.array([len(POISSON_CASES)], np.int64))
for idx, (B, F, T, prior_rate, beta_kl) in enumerate(POISSON_CASES):
    rng = np.random.default_rng(2000 + idx)
    z = (rng.standard_normal((B, F)) * 1.5).astype(np.float32)
    grad_output = rng.standard_normal((B, F)).astype(np.float32)

    rate, kl, grad = independent_poisson_latent(z, prior_rate, beta_kl, T, grad_output)

    p = f"poisson_{idx}_"
    put(p + "params", np.array([B, F, T, prior_rate, beta_kl], np.float32))
    put(p + "input", z)                         # (B, F)
    put(p + "grad_output", grad_output)          # (B, F)
    put(p + "rate", rate)                        # (B, F) == inference-mode output
    put(p + "kl_loss", np.array([[kl]], np.float32))
    put(p + "grad_input", grad)                  # (B, F)


os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **A)
print(f"wrote {len(A)} arrays -> {OUT}")
