#!/usr/bin/env python3
"""Generate PyWavelets ground-truth fixtures for the C++ wavelet ops.

Compares SUBBAND ROOT-ENERGIES (L2 norm per band) — the quantity the thesis
feature pipeline actually consumes (extract_subband_energies) — rather than
raw coefficients: band L2 norms are invariant to the circular phase shift
between this implementation's correlation indexing and pywt's convolution,
while still catching wrong filters, tree structure, band ordering, boundary
handling, or normalization.

Conventions mapped from src/core/wavelet/waveletOperations.cpp:
- Circular indexing         -> pywt mode='periodization'
- Packet high-pass swap     -> pywt order='freq' leaves
- C++ DaubN tag = N filter taps -> pywt 'db{N//2}' (Daub4=db2, Daub10=db5, ...)
- extract_subband_energies order:
    regular: [cA_L, cD_1, cD_2, ..., cD_L]   (finest detail FIRST)
    packet : 2^L bands, frequency order
- Energies are L2 norms: sqrt(sum(x^2)).

Output: src/core/wavelet/tests/fixtures/pywt_refs.npz (committed).
Deps (dev only): numpy, pywavelets. CI/C++ side needs only the .npz.
"""
import os

import numpy as np
import pywt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "src", "core", "wavelet", "tests",
                   "fixtures", "pywt_refs.npz")

N = 512
LEVEL = 3
rng = np.random.default_rng(42)

signals = {
    "sig0": rng.standard_normal(N),
    "sig1": (np.sin(2 * np.pi * 5 * np.arange(N) / N)
             + 0.5 * np.sin(2 * np.pi * 60 * np.arange(N) / N)
             + 0.1 * rng.standard_normal(N)),
}

# C++ tag name -> pywt wavelet name
WAVELETS = {"daub4": "db2", "daub10": "db5", "daub20": "db10"}

# ── phase alignment ──────────────────────────────────────────────────────────
# malat() correlates with the filter at even offsets (sum_f x[t+f] h[f]); pywt
# convolves with the reversed filter at its own offset. For circular
# (periodization) transforms the two are the same decomposition applied to a
# circularly shifted signal — but the shift depends on the filter (measured:
# db2 -> +1, db5 -> +4, db10 -> +1, modulo 2^level). Rather than hardcode a
# rule, mirror the C++ convention in numpy and SEARCH for the matching shift;
# failing to find one would itself flag a broken filter table.
def qmf(h):
    n = len(h)
    k = np.arange(n)
    return ((-1) ** k) * h[n - 1 - k]  # linearAlgebra::calc_orthogonal_vector

def cxx_step(x, h, g):
    n = len(x)
    lp = np.zeros(n // 2)
    hp = np.zeros(n // 2)
    for t in range(0, n, 2):
        idx = (t + np.arange(len(h))) % n
        lp[t // 2] = np.dot(x[idx], h)
        hp[t // 2] = np.dot(x[idx], g)
    return lp, hp

def find_alignment_shift(sig, pname, level):
    h = np.array(pywt.Wavelet(pname).filter_bank[2])
    g = qmf(h)
    x = sig.copy()
    dets = []
    for _ in range(level):
        x, d = cxx_step(x, h, g)
        dets.append(d)
    ours = np.array([np.linalg.norm(x)] + [np.linalg.norm(d) for d in reversed(dets)])
    for shift in range(2 ** level):
        c = pywt.wavedec(np.roll(sig, shift), pname, mode="periodization", level=level)
        ref = np.array([np.linalg.norm(b) for b in c])
        if np.abs(ref - ours).max() < 1e-9:
            return shift
    raise RuntimeError(f"no circular shift aligns pywt {pname} with the C++ "
                       "convention — filter table or convention regression?")

out = {}
for sname, sig in signals.items():
    out[sname] = sig.astype(np.float64)
    for cname, pname in WAVELETS.items():
        psig = np.roll(sig, find_alignment_shift(sig, pname, LEVEL))
        # Regular DWT: pywt returns [cA_L, cD_L, ..., cD_1]; C++ helper emits
        # [cA_L, cD_1, ..., cD_L] — store in the C++ order.
        coeffs = pywt.wavedec(psig, pname, mode="periodization", level=LEVEL)
        bands = [coeffs[0]] + list(reversed(coeffs[1:]))
        out[f"reg_{cname}_L{LEVEL}_{sname}"] = np.array(
            [np.sqrt(np.sum(b * b)) for b in bands], dtype=np.float64)

        # Packet: frequency-ordered leaves at LEVEL.
        wp = pywt.WaveletPacket(psig, pname, mode="periodization", maxlevel=LEVEL)
        nodes = wp.get_level(LEVEL, order="freq")
        out[f"pkt_{cname}_L{LEVEL}_{sname}"] = np.array(
            [np.sqrt(np.sum(n.data * n.data)) for n in nodes], dtype=np.float64)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **out)
print(f"wrote {OUT} ({len(out)} arrays)")
for k in sorted(out):
    print(f"  {k}: shape {out[k].shape}")
