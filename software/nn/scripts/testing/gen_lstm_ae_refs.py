#!/usr/bin/env python3
"""Generate PyTorch ground-truth fixtures for the WHOLE LSTM autoencoder.

`micro_network_parity_gtest` checks one LSTM layer + a Linear head, forward only. The network
the meeting01 experiment trains is `nn::models::lstm::LSTMAutoencoder`: stacked encoder, tanh
bottleneck, a decoder fed the same expanded latent at every step, stacked decoder, per-step
output projection, trained on a MASKED mean-squared error. This pins that composition -- the
reconstruction, the loss, and the gradient of every parameter after one backward pass -- against
an independent PyTorch implementation (`lstm_ae_torch.py`) from IDENTICAL weights.

Two cases, both with weights drawn much wider than the training init (std 0.4 / 0.3 instead of
0.05) so every gate works in its non-linear range and the gradients are not ~0:
    case 0   D=3 H=5 Z=2 L=2 S=7   a stacked (2-layer) autoencoder
    case 1   D=4 H=6 Z=3 L=1 S=9   a single layer, the shape the GA also draws
The mask has zero entries in the last frames (a zero-padded window tail), as in training.

Run (developer step; needs torch -- CI consumes the committed .npz):
    software/nn/.venv/bin/python software/nn/scripts/testing/gen_lstm_ae_refs.py

Output: software/nn/src/core/models/lstm/tests/fixtures/lstm_ae_refs.npz
"""
import os

import numpy as np
import torch

from lstm_ae_torch import LstmAe, grads_to_cpp, load_cpp_params, masked_mse, params_to_cpp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "src", "core", "models", "lstm", "tests", "fixtures",
                   "lstm_ae_refs.npz")

CASES = [  # (D, H, Z, L, S, weight std, seed)
    (3, 5, 2, 2, 7, 0.4, 1234),
    (4, 6, 3, 1, 9, 0.3, 4321),
]


def main() -> None:
    torch.set_default_dtype(torch.float32)
    refs: dict[str, np.ndarray] = {}
    refs["n_cases"] = np.array([len(CASES)], dtype=np.float32)

    for k, (d, h, z, layers, s, std, seed) in enumerate(CASES):
        rng = np.random.default_rng(seed)
        model = LstmAe(d, h, z, layers)

        # Wide random parameters in the C++ layout, then loaded into torch (so the C++ test
        # is handed exactly what torch ran on).
        cpp = {name: rng.normal(0.0, std, arr.shape).astype(np.float32)
               for name, arr in params_to_cpp(model).items()}
        load_cpp_params(model, cpp)

        x = rng.normal(0.0, 1.0, (s, d)).astype(np.float32)
        target = rng.normal(0.0, 1.0, (s, d)).astype(np.float32)
        mask = np.ones((s, d), dtype=np.float32)
        mask[s - 2:, :] = 0.0  # a zero-padded tail

        xt, tt, mt = (torch.from_numpy(a) for a in (x, target, mask))
        model.zero_grad()
        recon = model(xt)
        loss = masked_mse(recon, tt, mt)
        loss.backward()

        # Round trip: what torch holds must read back as the fixture the C++ side loads.
        back = params_to_cpp(model)
        for name, arr in cpp.items():
            assert np.allclose(back[name], arr, atol=1e-7), f"layout round trip failed: {name}"

        p = f"c{k}_"
        refs[p + "dims"] = np.array([d, h, z, layers, s], dtype=np.float32)
        refs[p + "x"] = x
        refs[p + "target"] = target
        refs[p + "mask"] = mask
        refs[p + "recon"] = recon.detach().numpy()
        refs[p + "loss"] = np.array([loss.item()], dtype=np.float32)
        for name, arr in cpp.items():
            refs[p + "p." + name] = arr
        for name, arr in grads_to_cpp(model).items():
            refs[p + "g." + name] = arr

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **refs)
    print(f"wrote {os.path.normpath(OUT)}: {len(refs)} arrays")


if __name__ == "__main__":
    main()
