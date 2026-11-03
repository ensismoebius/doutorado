"""PyTorch twin of nn::models::lstm::LSTMAutoencoder (include/models/lstm/LSTMAutoencoder.hpp).

Why this exists: `micro_network_parity_gtest` pins ONE LSTM layer + a Linear head against
torch.nn.LSTM, forward only. The network the meeting01 experiment trains is the whole
autoencoder -- stacked encoder, tanh bottleneck, a decoder fed the SAME expanded latent at every
step, stacked decoder, per-step output projection, masked MSE -- and nothing compared that
composition (nor its parameter gradients) with an independent implementation. This file is that
independent implementation, written from the C++ source's documented structure, not copied from
it. `gen_lstm_ae_refs.py` turns it into committed fixtures for `lstm_ae_pytorch_parity_gtest`;
`scripts/analysis/compare_lstm_ae_torch.py` uses it to train on real data.

C++ structure mirrored (LSTMAutoencoder.cpp):
    encode : x (S, D) -> L stacked LSTM layers -> last row h_S (1, H)
             -> Linear(H, Z) -> tanh                                  = z (1, Z)
    decode : Linear(Z, H) (NO activation) -> the same (1, H) row repeated S times
             -> L stacked LSTM layers -> Linear(H, D) per step        = recon (S, D)
    loss   : sum(mask * (recon - target)^2) / sum(mask)               (MSELossImpl, masked)

Layout differences between the two codebases, handled by the two *_gates helpers:
    gate order   C++ rows [i | f | o | g]      torch rows [i | f | g | o]
    bias         C++ one merged vector b       torch bias_ih + bias_hh (we put b in bias_ih, 0 in
                 bias_hh -- only their sum enters the pre-activation)
    Linear bias  C++ (out, 1)                  torch (out,)
Everything else (W is (4H, in), U is (4H, H), Linear weight is (out, in), y = x W^T + b) agrees.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

INIT_STD = 0.05  # LSTMLayerImpl / LSTMAutoencoder constructors: N(0, 0.05) for every weight


def cpp_gates_to_torch(m: np.ndarray) -> np.ndarray:
    """Rows [i|f|o|g] (C++) -> [i|f|g|o] (torch). Works for (4H, k) and (4H,) arrays."""
    h = m.shape[0] // 4
    return np.concatenate([m[:h], m[h:2 * h], m[3 * h:], m[2 * h:3 * h]], axis=0)


def torch_gates_to_cpp(m: np.ndarray) -> np.ndarray:
    """Inverse of `cpp_gates_to_torch`."""
    h = m.shape[0] // 4
    return np.concatenate([m[:h], m[h:2 * h], m[3 * h:], m[2 * h:3 * h]], axis=0)


class LstmAe(nn.Module):
    """Same topology as the C++ LSTMAutoencoder. Input/target are (S, D) single sequences."""

    def __init__(self, d: int, h: int, z: int, layers: int) -> None:
        super().__init__()
        self.d, self.h, self.z, self.layers = d, h, z, layers
        self.enc = nn.LSTM(d, h, num_layers=layers)
        self.enc_proj = nn.Linear(h, z)
        self.dec_expand = nn.Linear(z, h)
        self.dec = nn.LSTM(h, h, num_layers=layers)
        self.out_proj = nn.Linear(h, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape[0]
        enc_out, _ = self.enc(x.unsqueeze(1))                       # (S, 1, H)
        z = torch.tanh(self.enc_proj(enc_out[-1]))                  # (1, Z)
        he = self.dec_expand(z)                                     # (1, H), no activation
        dec_in = he.unsqueeze(0).expand(s, 1, self.h).contiguous()  # (S, 1, H): same row each step
        dec_out, _ = self.dec(dec_in)                               # (S, 1, H)
        return self.out_proj(dec_out[:, 0, :])                      # (S, D)


def masked_mse(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSELossImpl with a mask: sum(mask * diff^2) / sum(mask)."""
    return (mask * (recon - target) ** 2).sum() / mask.sum()


def cpp_style_init(model: LstmAe, rng: np.random.Generator) -> None:
    """The DISTRIBUTION of the C++ constructor, not its bits: every weight N(0, 0.05); every
    bias 0 except the LSTM forget-gate bias, which is 1. (The C++ code reuses ONE seed per
    weight kind across layers, so its layers start from identical draws; here each layer draws
    independently. Not a difference that matters for a loss curve, but it is one.)"""
    with torch.no_grad():
        for lstm in (model.enc, model.dec):
            for l in range(model.layers):
                w_ih = getattr(lstm, f"weight_ih_l{l}")
                w_hh = getattr(lstm, f"weight_hh_l{l}")
                b_ih = getattr(lstm, f"bias_ih_l{l}")
                b_hh = getattr(lstm, f"bias_hh_l{l}")
                w_ih.copy_(torch.from_numpy(rng.normal(0, INIT_STD, w_ih.shape).astype(np.float32)))
                w_hh.copy_(torch.from_numpy(rng.normal(0, INIT_STD, w_hh.shape).astype(np.float32)))
                b_ih.zero_()
                b_hh.zero_()
                hh = model.h
                b_ih[hh:2 * hh] = 1.0  # torch order [i|f|g|o]: rows h..2h are the forget gate
        for lin in (model.enc_proj, model.dec_expand, model.out_proj):
            lin.weight.copy_(torch.from_numpy(
                rng.normal(0, INIT_STD, lin.weight.shape).astype(np.float32)))
            lin.bias.zero_()


def params_to_cpp(model: LstmAe) -> dict[str, np.ndarray]:
    """Parameters in the C++ layout/naming of LSTMAutoencoder::state_dict()."""
    out: dict[str, np.ndarray] = {}
    for tag, lstm in (("enc_lstm", model.enc), ("dec_lstm", model.dec)):
        for l in range(model.layers):
            w_ih = getattr(lstm, f"weight_ih_l{l}").detach().numpy()
            w_hh = getattr(lstm, f"weight_hh_l{l}").detach().numpy()
            b = (getattr(lstm, f"bias_ih_l{l}") + getattr(lstm, f"bias_hh_l{l}")).detach().numpy()
            out[f"{tag}{l}.W"] = torch_gates_to_cpp(w_ih)
            out[f"{tag}{l}.U"] = torch_gates_to_cpp(w_hh)
            out[f"{tag}{l}.b"] = torch_gates_to_cpp(b).reshape(-1, 1)
    for tag, lin in (("enc_proj", model.enc_proj), ("dec_expand", model.dec_expand),
                     ("out_proj", model.out_proj)):
        out[f"{tag}.weight"] = lin.weight.detach().numpy()
        out[f"{tag}.bias"] = lin.bias.detach().numpy().reshape(-1, 1)
    return out


def grads_to_cpp(model: LstmAe) -> dict[str, np.ndarray]:
    """Parameter gradients in the same layout as `params_to_cpp`. For the merged C++ bias the
    gradient is that of bias_ih (bias_hh enters the pre-activation identically, so its
    gradient is the same vector; the C++ b is their sum)."""
    out: dict[str, np.ndarray] = {}
    for tag, lstm in (("enc_lstm", model.enc), ("dec_lstm", model.dec)):
        for l in range(model.layers):
            out[f"{tag}{l}.W"] = torch_gates_to_cpp(getattr(lstm, f"weight_ih_l{l}").grad.numpy())
            out[f"{tag}{l}.U"] = torch_gates_to_cpp(getattr(lstm, f"weight_hh_l{l}").grad.numpy())
            out[f"{tag}{l}.b"] = torch_gates_to_cpp(
                getattr(lstm, f"bias_ih_l{l}").grad.numpy()).reshape(-1, 1)
    for tag, lin in (("enc_proj", model.enc_proj), ("dec_expand", model.dec_expand),
                     ("out_proj", model.out_proj)):
        out[f"{tag}.weight"] = lin.weight.grad.numpy()
        out[f"{tag}.bias"] = lin.bias.grad.numpy().reshape(-1, 1)
    return out


def load_cpp_params(model: LstmAe, cpp: dict[str, np.ndarray]) -> None:
    """Inverse of `params_to_cpp`: put C++-layout parameters into the torch model."""
    with torch.no_grad():
        for tag, lstm in (("enc_lstm", model.enc), ("dec_lstm", model.dec)):
            for l in range(model.layers):
                getattr(lstm, f"weight_ih_l{l}").copy_(
                    torch.from_numpy(cpp_gates_to_torch(cpp[f"{tag}{l}.W"])))
                getattr(lstm, f"weight_hh_l{l}").copy_(
                    torch.from_numpy(cpp_gates_to_torch(cpp[f"{tag}{l}.U"])))
                getattr(lstm, f"bias_ih_l{l}").copy_(
                    torch.from_numpy(cpp_gates_to_torch(cpp[f"{tag}{l}.b"].reshape(-1))))
                getattr(lstm, f"bias_hh_l{l}").zero_()
        for tag, lin in (("enc_proj", model.enc_proj), ("dec_expand", model.dec_expand),
                         ("out_proj", model.out_proj)):
            lin.weight.copy_(torch.from_numpy(cpp[f"{tag}.weight"]))
            lin.bias.copy_(torch.from_numpy(cpp[f"{tag}.bias"].reshape(-1)))
