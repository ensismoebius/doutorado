#!/usr/bin/env python3
"""Generate PyTorch ground-truth fixtures for the C++ parity tests.

For each covered layer this builds a torch module with fixed, seeded weights,
runs forward (and backward where relevant), and saves inputs / weights /
outputs / gradients as float32 arrays into a single .npz. The C++ test
`pytorch_parity_gtest` loads that .npz, sets the same weights into our layers,
runs forward/backward, and asserts EXPECT_NEAR against these references.

Run (developer step; requires torch — CI consumes the committed .npz):
    software/nn/.venv/bin/python software/nn/scripts/testing/gen_pytorch_refs.py

Output: software/nn/src/core/layers/tests/fixtures/pytorch_refs.npz
"""
import math
import os

import numpy as np
import snntorch as snn
import torch

torch.manual_seed(0)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "src", "core", "layers", "tests", "fixtures",
                   "pytorch_refs.npz")
OUT = os.path.normpath(OUT)

A = {}  # name -> float32 ndarray


def put(name, t):
    if isinstance(t, np.ndarray):
        A[name] = np.ascontiguousarray(t)          # keep dtype (e.g. int64 counts)
    else:
        A[name] = np.ascontiguousarray(t.detach().cpu().numpy().astype(np.float32))


# ── Linear (forward + backward) ───────────────────────────────────────────────
# torch nn.Linear weight is (out,in), bias (out) — same layout as our Linear.
LINEAR_CASES = [(2, 4, 3), (3, 5, 2), (1, 6, 6)]
put("linear_num", np.array([len(LINEAR_CASES)], np.int64))
for idx, (B, IN, OUT_F) in enumerate(LINEAR_CASES):
    lin = torch.nn.Linear(IN, OUT_F)
    torch.nn.init.normal_(lin.weight, std=0.5)
    torch.nn.init.normal_(lin.bias, std=0.5)
    x = torch.randn(B, IN, requires_grad=True)
    y = lin(x)
    g = torch.randn(B, OUT_F)
    y.backward(g)
    p = f"linear_{idx}_"
    put(p + "weight", lin.weight)          # (OUT_F, IN)
    put(p + "bias", lin.bias)              # (OUT_F,)
    put(p + "input", x)                    # (B, IN)
    put(p + "output", y)                   # (B, OUT_F)
    put(p + "grad_output", g)              # (B, OUT_F)
    put(p + "grad_input", x.grad)          # (B, IN)
    put(p + "grad_weight", lin.weight.grad)  # (OUT_F, IN)
    put(p + "grad_bias", lin.bias.grad)    # (OUT_F,)


# ── Elementwise activations (forward + backward) ──────────────────────────────
ACTS = {
    "tanh": (torch.tanh, {}),
    "sigmoid": (torch.sigmoid, {}),
    "relu": (torch.relu, {}),
    "leaky": (lambda t: torch.nn.functional.leaky_relu(t, 0.01), {}),
}
for name, (fn, _) in ACTS.items():
    x = torch.randn(4, 5, requires_grad=True)
    y = fn(x)
    g = torch.randn(4, 5)
    y.backward(g)
    p = f"act_{name}_"
    put(p + "input", x)
    put(p + "output", y)
    put(p + "grad_output", g)
    put(p + "grad_input", x.grad)


# ── MSELoss (mean reduction) ──────────────────────────────────────────────────
pred = torch.randn(3, 4, requires_grad=True)
tgt = torch.randn(3, 4)
loss = torch.nn.functional.mse_loss(pred, tgt, reduction="mean")
loss.backward()
put("mse_pred", pred)
put("mse_target", tgt)
put("mse_loss", loss.reshape(1, 1))
put("mse_grad_pred", pred.grad)


# ── LSTM (forward) — weights stored in OUR gate order i,f,o,g, merged bias ─────
# torch nn.LSTM gate order is i,f,g,o with separate bias_ih + bias_hh; ours is
# i,f,o,g with a single bias. Permute the last two H-blocks and sum the biases.
def to_our_gate_order(w, H):
    # w: (4H, X) in torch order [i,f,g,o] -> our order [i,f,o,g]
    i, f, g, o = w[0:H], w[H:2 * H], w[2 * H:3 * H], w[3 * H:4 * H]
    return torch.cat([i, f, o, g], dim=0)


# Our LSTM uses rational-approximation activations (FastActivations.hpp) instead
# of exact sigmoid/tanh. Reproduce them here so the C++ result can be checked
# exactly against this reference (isolating gate order / recurrence correctness),
# separately from the loose bound vs. PyTorch's exact activations.
def rat_sig(x):
    return np.where(x <= -10, 0.0, np.where(x >= 10, 1.0, 0.5 + x / (2.0 * (1.0 + np.abs(x)))))


def rat_tanh(x):
    return np.where(x <= -10, -1.0, np.where(x >= 10, 1.0, x / (1.0 + np.abs(x))))


def lstm_forward_approx(x, W, U, b, H):
    # x:(B,T,D); W:(4H,D); U:(4H,H); b:(4H,1) — all in our gate order i,f,o,g.
    B, T, _ = x.shape
    out = np.zeros((B, T, H), np.float32)
    for bi in range(B):
        h = np.zeros(H, np.float32)
        c = np.zeros(H, np.float32)
        for t in range(T):
            pre = W @ x[bi, t] + U @ h + b[:, 0]
            i = rat_sig(pre[0:H]); f = rat_sig(pre[H:2 * H])
            o = rat_sig(pre[2 * H:3 * H]); g = rat_tanh(pre[3 * H:4 * H])
            c = f * c + i * g
            h = o * rat_tanh(c)
            out[bi, t] = h
    return out


LSTM_CASES = [(2, 3, 4, 5), (1, 4, 3, 6)]
put("lstm_num", np.array([len(LSTM_CASES)], np.int64))
for idx, (B, T, D, H) in enumerate(LSTM_CASES):
    lstm = torch.nn.LSTM(input_size=D, hidden_size=H, num_layers=1, batch_first=True)
    for pn, pp in lstm.named_parameters():
        torch.nn.init.normal_(pp, std=0.3)
    x = torch.randn(B, T, D)
    with torch.no_grad():
        out, _ = lstm(x)  # (B, T, H), h0=c0=0, exact activations
    W = to_our_gate_order(lstm.weight_ih_l0, H)                 # (4H, D)
    U = to_our_gate_order(lstm.weight_hh_l0, H)                 # (4H, H)
    b = to_our_gate_order(
        (lstm.bias_ih_l0 + lstm.bias_hh_l0).reshape(4 * H, 1), H)  # (4H, 1)

    approx = lstm_forward_approx(
        x.numpy().astype(np.float32), W.detach().numpy().astype(np.float32),
        U.detach().numpy().astype(np.float32), b.detach().numpy().astype(np.float32), H)

    p = f"lstm_{idx}_"
    put(p + "dims", np.array([B, T, D, H], np.int64))
    put(p + "input", x)
    put(p + "W", W)
    put(p + "U", U)
    put(p + "b", b)
    put(p + "output", out)              # PyTorch exact (loose bound)
    put(p + "output_approx", approx)    # rational-approx (tight, our activations)


# ── LIF spiking neuron vs snnTorch Leaky ──────────────────────────────────────
# Our LifBPTT recurrence  v[t] = beta*v[t-1] + input[t];  spike when v > V_th;
# reset either subtract (v -= V_th) or zero (v = 0). This is exactly snnTorch's
# snn.Leaky(beta, threshold, reset_mechanism). We pick R=C=1 and time_step=-ln(beta)
# in C++ so beta = exp(-time_step/(R*C)) reproduces the beta given to snnTorch.
#
# Cases: (T, B, F, beta, V_th, reset_mechanism, readout)
#   readout=1 emits the membrane (no spike/reset) — compared against a snnTorch
#   neuron with an unreachable threshold (pure leaky integrator).
LIF_CASES = [
    (6, 2, 3, 0.9, 1.0, "subtract", 0),
    (6, 2, 3, 0.9, 0.8, "zero", 0),
    (8, 1, 4, 0.8, 1.0, "subtract", 1),  # membrane (readout) parity
]
put("lif_num", np.array([len(LIF_CASES)], np.int64))
for idx, (T, B, F, beta, vth, mech, readout) in enumerate(LIF_CASES):
    torch.manual_seed(100 + idx)
    x = torch.randn(T, B, F)
    thr = 1.0e9 if readout else vth  # readout: never spikes → pure integrator
    lif = snn.Leaky(beta=float(beta), threshold=float(thr), reset_mechanism=mech,
                    init_hidden=False)
    mem = torch.zeros(B, F)
    spk_out = torch.zeros(T, B, F)
    mem_out = torch.zeros(T, B, F)
    for t in range(T):
        spk, mem = lif(x[t], mem)
        spk_out[t] = spk
        mem_out[t] = mem
    time_step = -math.log(beta)  # so exp(-time_step/(R=1 * C=1)) == beta
    p = f"lif_{idx}_"
    put(p + "dims", np.array([T, B, F], np.int64))
    # [time_step, R, C, V_th, reset_zero(1=zero/0=subtract), readout]
    put(p + "params", np.array(
        [time_step, 1.0, 1.0, vth, 1.0 if mech == "zero" else 0.0, float(readout)], np.float32))
    put(p + "input", x)
    put(p + "spk", spk_out)
    put(p + "mem", mem_out)


# ── Conv1d / Conv2d vs torch ──────────────────────────────────────────────────
# Weights are stored already permuted into OUR im2col layout so the C++ test can
# set them directly via set_weights():
#   Conv1d weights_ (Cin*K, Cout):     our[ic*K + k, oc]           = torch[oc,ic,k]
#   Conv2d weights_ (Cin*K*K, Cout):   our[ic*K*K + ky*K + kx, oc] = torch[oc,ic,ky,kx]
#   bias_ (1, Cout).
# Stride 1, no padding, square kernels (our Conv2d only supports square kernels).
CONV1D_CASES = [(1, 2, 3, 8, 3), (2, 3, 4, 10, 5)]  # (N, Cin, Cout, L, K)
put("conv1d_num", np.array([len(CONV1D_CASES)], np.int64))
for idx, (N, Cin, Cout, L, K) in enumerate(CONV1D_CASES):
    torch.manual_seed(200 + idx)
    conv = torch.nn.Conv1d(Cin, Cout, K, stride=1, padding=0, bias=True)
    torch.nn.init.normal_(conv.weight, std=0.5)
    torch.nn.init.normal_(conv.bias, std=0.5)
    x = torch.randn(N, Cin, L)
    with torch.no_grad():
        y = conv(x)  # (N, Cout, L-K+1)
    w = conv.weight.detach().numpy()                 # (Cout, Cin, K)
    w_ours = np.transpose(w, (1, 2, 0)).reshape(Cin * K, Cout)  # (Cin*K, Cout)
    p = f"conv1d_{idx}_"
    put(p + "dims", np.array([N, Cin, Cout, L, K], np.int64))
    put(p + "weight", w_ours)
    put(p + "bias", conv.bias.detach().reshape(1, Cout))
    put(p + "input", x)
    put(p + "output", y)

CONV2D_CASES = [(1, 2, 3, 6, 6, 3), (2, 3, 2, 7, 7, 2)]  # (N, Cin, Cout, H, W, K) square kernel
put("conv2d_num", np.array([len(CONV2D_CASES)], np.int64))
for idx, (N, Cin, Cout, H, W, K) in enumerate(CONV2D_CASES):
    torch.manual_seed(300 + idx)
    conv = torch.nn.Conv2d(Cin, Cout, K, stride=1, padding=0, bias=True)
    torch.nn.init.normal_(conv.weight, std=0.5)
    torch.nn.init.normal_(conv.bias, std=0.5)
    x = torch.randn(N, Cin, H, W)
    with torch.no_grad():
        y = conv(x)  # (N, Cout, H-K+1, W-K+1)
    w = conv.weight.detach().numpy()                 # (Cout, Cin, K, K)
    w_ours = np.transpose(w, (1, 2, 3, 0)).reshape(Cin * K * K, Cout)  # (Cin*K*K, Cout)
    p = f"conv2d_{idx}_"
    put(p + "dims", np.array([N, Cin, Cout, H, W, K], np.int64))
    put(p + "weight", w_ours)
    put(p + "bias", conv.bias.detach().reshape(1, Cout))
    put(p + "input", x)
    put(p + "output", y)


# ── LifBPTT backward (readout / leaky integrator) vs snnTorch autograd ─────────
# Exact BPTT temporal-gradient check: with an unreachable threshold the neuron is
# a pure leaky integrator v[t]=beta*v[t-1]+input[t] (no spike/reset/surrogate).
# loss = sum(all membranes) → dL/dinput backpropagated through the recurrence.
# Our LifBPTT in readout_mode(=emit v_mem) + grad_output=ones must reproduce it.
LIFBW_CASES = [(5, 2, 3, 0.9), (6, 1, 4, 0.8)]  # (T, B, F, beta)
put("lifbw_num", np.array([len(LIFBW_CASES)], np.int64))
for idx, (T, B, F, beta) in enumerate(LIFBW_CASES):
    torch.manual_seed(700 + idx)
    x = torch.randn(T, B, F, requires_grad=True)
    lif = snn.Leaky(beta=float(beta), threshold=1.0e9, reset_mechanism="subtract",
                    init_hidden=False)
    mem = torch.zeros(B, F)
    total = torch.zeros(())
    for t in range(T):
        _, mem = lif(x[t], mem)
        total = total + mem.sum()
    total.backward()
    time_step = -math.log(beta)
    p = f"lifbw_{idx}_"
    put(p + "dims", np.array([T, B, F], np.int64))
    put(p + "params", np.array([time_step, 1.0, 1.0], np.float32))  # time_step, R, C
    put(p + "grad_input", x.grad)  # (T,B,F) == (T*B, F)


# ── CrossEntropyLoss (softmax + CE, mean) vs torch ────────────────────────────
# Our CrossEntropyLoss takes one-hot targets (N,C); torch takes class indices.
CE_CASES = [(3, 4), (5, 2)]  # (N, C)
put("ce_num", np.array([len(CE_CASES)], np.int64))
for idx, (N, Cc) in enumerate(CE_CASES):
    torch.manual_seed(400 + idx)
    logits = torch.randn(N, Cc, requires_grad=True)
    tgt_idx = torch.randint(0, Cc, (N,))
    loss = torch.nn.functional.cross_entropy(logits, tgt_idx, reduction="mean")
    loss.backward()
    onehot = torch.zeros(N, Cc)
    onehot[torch.arange(N), tgt_idx] = 1.0
    p = f"ce_{idx}_"
    put(p + "logits", logits)
    put(p + "target", onehot)                 # one-hot (N,C) for our API
    put(p + "loss", loss.reshape(1, 1))
    put(p + "grad_logits", logits.grad)


# ── MaxPool1d / MaxPool2d (forward) vs torch ──────────────────────────────────
MP1_CASES = [(1, 2, 8, 2, 2), (2, 3, 9, 3, 2)]  # (N, C, L, k, stride)
put("maxpool1d_num", np.array([len(MP1_CASES)], np.int64))
for idx, (N, C, L, k, s) in enumerate(MP1_CASES):
    torch.manual_seed(500 + idx)
    x = torch.randn(N, C, L)
    y = torch.nn.functional.max_pool1d(x, kernel_size=k, stride=s)
    p = f"maxpool1d_{idx}_"
    put(p + "params", np.array([k, s], np.int64))
    put(p + "input", x)
    put(p + "output", y)

MP2_CASES = [(1, 2, 6, 6, 2, 2), (2, 3, 7, 5, 3, 2)]  # (N, C, H, W, k, stride)
put("maxpool2d_num", np.array([len(MP2_CASES)], np.int64))
for idx, (N, C, H, W, k, s) in enumerate(MP2_CASES):
    torch.manual_seed(600 + idx)
    x = torch.randn(N, C, H, W)
    y = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=s)
    p = f"maxpool2d_{idx}_"
    put(p + "params", np.array([k, s], np.int64))
    put(p + "input", x)
    put(p + "output", y)


os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **A)
print(f"wrote {len(A)} arrays -> {OUT}")
