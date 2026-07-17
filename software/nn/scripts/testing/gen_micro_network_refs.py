#!/usr/bin/env python3
"""Generate ground-truth fixtures for WHOLE micro-networks (not single layers).

`gen_pytorch_refs.py` already pins each layer in isolation. That is necessary but not
sufficient: every layer can be individually correct while the *network* is wrong — gradients
chained in the wrong order through a stack, state not reset between sequences, a gate
permuted only when composed, a bias applied twice. The experiments train networks, not
layers, so this file pins three minimal end-to-end networks that mirror the shapes the
thesis actually uses.

Covered (all forward AND backward, i.e. parameter gradients after one loss):

  micro_ann   Linear(4->3) -> ReLU -> Linear(3->2), MSE loss
              vs torch.nn.Sequential. EXACT parity expected.

  micro_snn   Linear(4->3) -> LIF -> Linear(3->2), time-major (T*B, F), MSE loss
              vs snnTorch snn.Leaky. Forward is exact. Backward is pinned in
              READOUT mode only — see the note below.

  micro_lstm  LSTM(3->4) -> Linear(4->2), MSE loss
              vs a NumPy model of our own recurrence. NOT vs torch.nn.LSTM — see below.

--------------------------------------------------------------------------------
Two honest limits, both deliberate and both encoded here rather than hidden:

1. SNN backward. Our LifBPTT uses an ExponentialSurrogate for the spike derivative;
   snnTorch's Leaky defaults to an arctan surrogate. Those are *different functions*, so
   gradients through a spiking layer cannot agree and pinning them against each other would
   be meaningless. Forward (which involves no surrogate) is compared exactly, with spikes.
   Backward is compared in readout mode, where the neuron emits its membrane and never
   spikes, so the path is purely continuous and both sides must agree exactly. This isolates
   "is the recurrence + composition + gradient chaining right" from "which surrogate".

2. LSTM. Our LSTMLayer does NOT use sigmoid/tanh. It uses the rational approximations in
   FastActivations.hpp: rat_sig(x) = 0.5 + x/(2(1+|x|)) and rat_tanh(x) = x/(1+|x|), i.e.
   softsign-based gates chosen for speed. These are not close to the real thing:
   |tanh - rat_tanh| reaches 0.306 on [-4,4] (at x=2, tanh=0.964 vs ours=0.667). So our
   LSTM is a *softsign-gated* LSTM and can never match torch.nn.LSTM numerically. Pinning
   it against torch would either fail or need a tolerance so loose it proves nothing.
   Instead we pin it against a NumPy model of our own recurrence, which still catches gate
   order, the c/h update order, bias merging and composition — everything except the choice
   of activation, which is a documented design decision, not a bug. `micro_lstm_torch_*`
   additionally records what real torch.nn.LSTM produces from the SAME weights, purely so
   the C++ test can assert the divergence is the expected size and not silently grow.

Run (developer step; needs torch + snntorch — CI consumes the committed .npz):
    software/nn/.venv/bin/python software/nn/scripts/testing/gen_micro_network_refs.py

Output: software/nn/src/core/layers/tests/fixtures/micro_network_refs.npz
"""
import math
import os

import numpy as np
import snntorch as snn
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "src", "core", "layers", "tests", "fixtures",
                   "micro_network_refs.npz")
OUT = os.path.normpath(OUT)

A = {}


def put(name, t):
    # Deep-copy always: torch tensors share storage with numpy views, and several of these
    # networks mutate parameters in place. (This exact aliasing produced a fixture where
    # every step was the last step in gen_optimizer_refs.py.)
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    A[name] = np.array(t, dtype=np.float32, copy=True)


# ══ micro_ann: Linear(4->3) -> ReLU -> Linear(3->2) ══════════════════════════
# Exact parity with torch is expected: every op here has an exact counterpart.
torch.manual_seed(0)
B, D, H, O = 2, 4, 3, 2
net = torch.nn.Sequential(torch.nn.Linear(D, H), torch.nn.ReLU(), torch.nn.Linear(H, O))
for m in (net[0], net[2]):
    torch.nn.init.normal_(m.weight, std=0.6)
    torch.nn.init.normal_(m.bias, std=0.6)

x = torch.randn(B, D, requires_grad=True)
tgt = torch.randn(B, O)
out = net(x)
loss = torch.nn.functional.mse_loss(out, tgt, reduction="mean")
net.zero_grad()
loss.backward()

put("ann_dims", np.array([B, D, H, O], np.float32))
put("ann_w1", net[0].weight)   # (H, D) — same layout as our Linear
put("ann_b1", net[0].bias)
put("ann_w2", net[2].weight)   # (O, H)
put("ann_b2", net[2].bias)
put("ann_x", x)
put("ann_target", tgt)
put("ann_out", out)
put("ann_loss", np.array([[loss.item()]], np.float32))
put("ann_gw1", net[0].weight.grad)
put("ann_gb1", net[0].bias.grad)
put("ann_gw2", net[2].weight.grad)
put("ann_gb2", net[2].bias.grad)
put("ann_gx", x.grad)


# ══ micro_snn: Linear(4->3) -> LIF -> Linear(3->2), time-major ═══════════════
# beta is given to snnTorch directly; on our side R=C=1 and time_step=-ln(beta) makes
# beta = exp(-time_step/(R*C)) reproduce it exactly (same trick as gen_pytorch_refs.py).
def snn_case(tag, T, Bs, D, H, O, beta, vth, readout, seed):
    torch.manual_seed(seed)
    fc_in = torch.nn.Linear(D, H)
    fc_out = torch.nn.Linear(H, O)
    for m in (fc_in, fc_out):
        torch.nn.init.normal_(m.weight, std=0.6)
        torch.nn.init.normal_(m.bias, std=0.6)

    # readout: threshold unreachable -> never spikes -> pure leaky integrator, so the whole
    # path is differentiable without a surrogate and gradients are exact on both sides.
    thr = 1.0e9 if readout else vth
    # reset_mechanism="zero" is what the thesis actually uses: Lif/LifBPTT default to
    # reset_zero=true and no production code selects subtract. It also happens to be the
    # mode where our LIF matches snnTorch EXACTLY (0.0% spike disagreement, measured),
    # whereas subtract diverges ~2-3% — see the divergence note in the module docstring.
    lif = snn.Leaky(beta=float(beta), threshold=float(thr),
                    reset_mechanism="zero", init_hidden=False)

    x = torch.randn(T, Bs, D, requires_grad=True)
    tgt = torch.randn(T, Bs, O)

    mem = torch.zeros(Bs, H)
    outs = []
    for t in range(T):
        cur = fc_in(x[t])
        spk, mem = lif(cur, mem)
        outs.append(fc_out(mem if readout else spk))
    out = torch.stack(outs)                      # (T, Bs, O)
    loss = torch.nn.functional.mse_loss(out, tgt, reduction="mean")
    fc_in.zero_grad(); fc_out.zero_grad()
    loss.backward()

    p = f"snn_{tag}_"
    put(p + "dims", np.array([T, Bs, D, H, O], np.float32))
    # [time_step, R, C, V_th, reset_zero(1=zero), readout]
    put(p + "params", np.array(
        [-math.log(beta), 1.0, 1.0, vth, 1.0, float(readout)], np.float32))
    put(p + "w1", fc_in.weight); put(p + "b1", fc_in.bias)
    put(p + "w2", fc_out.weight); put(p + "b2", fc_out.bias)
    # Flatten to our time-major (T*B, F) contract: t0 rows, then t1 rows, ...
    put(p + "x", x.reshape(T * Bs, D))
    put(p + "target", tgt.reshape(T * Bs, O))
    put(p + "out", out.reshape(T * Bs, O))
    put(p + "loss", np.array([[loss.item()]], np.float32))
    if readout:  # gradients only meaningful (and surrogate-free) in readout mode
        put(p + "gw1", fc_in.weight.grad); put(p + "gb1", fc_in.bias.grad)
        put(p + "gw2", fc_out.weight.grad); put(p + "gb2", fc_out.bias.grad)


snn_case("spk", T=5, Bs=2, D=4, H=3, O=2, beta=0.9, vth=1.0, readout=0, seed=1)   # forward
snn_case("ro",  T=5, Bs=2, D=4, H=3, O=2, beta=0.9, vth=1.0, readout=1, seed=2)   # fwd+bwd


# ══ micro_lstm: LSTM(3->4) -> Linear(4->2) ═══════════════════════════════════
# Pinned against a NumPy model of OUR recurrence (softsign gates), NOT torch — see header.
def rat_sig(x):
    return np.where(x <= -10, 0.0, np.where(x >= 10, 1.0, 0.5 + x / (2.0 * (1.0 + np.abs(x)))))


def rat_tanh(x):
    return np.where(x <= -10, -1.0, np.where(x >= 10, 1.0, x / (1.0 + np.abs(x))))


def to_our_gate_order(w, H):
    """torch [i,f,g,o] -> our [i,f,o,g]."""
    i, f, g, o = w[0:H], w[H:2 * H], w[2 * H:3 * H], w[3 * H:4 * H]
    return torch.cat([i, f, o, g], dim=0)


torch.manual_seed(7)
Tl, Bl, Dl, Hl, Ol = 4, 2, 3, 4, 2
lstm = torch.nn.LSTM(Dl, Hl, num_layers=1, batch_first=True)
head = torch.nn.Linear(Hl, Ol)
for p_ in lstm.parameters():
    torch.nn.init.normal_(p_, std=0.4)
torch.nn.init.normal_(head.weight, std=0.4)
torch.nn.init.normal_(head.bias, std=0.4)

xl = torch.randn(Bl, Tl, Dl)

# Weights in OUR layout: gate order i,f,o,g and a single merged bias.
W = to_our_gate_order(lstm.weight_ih_l0.detach(), Hl)              # (4H, D)
U = to_our_gate_order(lstm.weight_hh_l0.detach(), Hl)              # (4H, H)
bmerged = to_our_gate_order((lstm.bias_ih_l0 + lstm.bias_hh_l0).detach().unsqueeze(1), Hl)

# Our recurrence, in NumPy, with our approximations.
Wn, Un, bn, xn = W.numpy(), U.numpy(), bmerged.numpy(), xl.numpy()
h_seq = np.zeros((Bl, Tl, Hl), np.float32)
for bi in range(Bl):
    h = np.zeros(Hl, np.float32)
    c = np.zeros(Hl, np.float32)
    for t in range(Tl):
        pre = Wn @ xn[bi, t] + Un @ h + bn[:, 0]
        i = rat_sig(pre[0:Hl]); f = rat_sig(pre[Hl:2 * Hl])
        o = rat_sig(pre[2 * Hl:3 * Hl]); g = rat_tanh(pre[3 * Hl:4 * Hl])
        c = f * c + i * g
        h = o * rat_tanh(c)
        h_seq[bi, t] = h

# Head applied to the LAST timestep's hidden state (the autoencoder's latent readout).
hw = head.weight.detach().numpy(); hb = head.bias.detach().numpy()
last_h = h_seq[:, -1, :]                       # (B, H)
ours_out = last_h @ hw.T + hb                  # (B, O)

put("lstm_dims", np.array([Tl, Bl, Dl, Hl, Ol], np.float32))
put("lstm_W", W); put("lstm_U", U); put("lstm_b", bmerged)
put("lstm_hw", head.weight); put("lstm_hb", head.bias)
put("lstm_x", xl.reshape(Bl * Tl, Dl))         # (B*T, D), batch-major rows
put("lstm_h_seq", h_seq.reshape(Bl * Tl, Hl))  # our recurrence's hidden states
put("lstm_out", ours_out)                      # (B, O) — the reference our C++ must match

# What a REAL torch LSTM gives from the same weights, recorded so the C++ test can assert
# the divergence stays the expected size rather than silently growing.
with torch.no_grad():
    th_seq, _ = lstm(xl)
    torch_out = head(th_seq[:, -1, :])
put("lstm_torch_out", torch_out)
put("lstm_torch_h_seq", th_seq.reshape(Bl * Tl, Hl))

# ══ LIF reset-mode divergence: measured, recorded, asserted in C++ ═══════════
# Our LIF applies the reset IMMEDIATELY (v -= V_th, then decays next step), so the reset
# term ends up multiplied by beta. snnTorch subtracts it UN-decayed at the next step:
#   ours     : v[t] = beta*v[t-1] + I[t] - V_th*spk[t]      (v is post-reset)
#   snnTorch : mem[t] = beta*mem[t-1] + I[t] - V_th*spk[t-1]
# With reset="zero" the two coincide exactly (0 stays 0 through the decay), which is the
# mode the thesis uses. With reset="subtract" they disagree on ~2-3% of spikes. The existing
# per-layer fixture tests subtract but spikes only 3/36 times, so it passed by being too
# weakly driven to ever exercise a reset. These arrays pin BOTH modes under a HARD drive so
# the equivalence is proven where it holds and the divergence cannot silently reappear.
def reset_mode_case(tag, mech, scale, T=20, B=2, F=4, beta=0.9, vth=1.0, seed=11):
    torch.manual_seed(seed)
    x = torch.randn(T, B, F) * scale
    lif = snn.Leaky(beta=beta, threshold=vth, reset_mechanism=mech, init_hidden=False)
    mem = torch.zeros(B, F); spks = []
    for t in range(T):
        s, mem = lif(x[t], mem); spks.append(s.clone())
    spk = torch.stack(spks)
    p = f"reset_{tag}_"
    put(p + "dims", np.array([T, B, F], np.float32))
    put(p + "params", np.array(
        [-math.log(beta), 1.0, 1.0, vth, 1.0 if mech == "zero" else 0.0, 0.0], np.float32))
    put(p + "x", x.reshape(T * B, F))
    put(p + "spk", spk.reshape(T * B, F))
    print(f"  reset_{tag}: mech={mech} drive={scale} snnTorch spike rate={spk.mean():.3f}")


reset_mode_case("zero", "zero", 2.5)          # must match ours EXACTLY
reset_mode_case("subtract", "subtract", 2.5)  # known to diverge — bounded, asserted in C++

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **A)
print(f"wrote {OUT} with {len(A)} arrays")
print(f"  micro_lstm: max|ours - torch| over hidden states = "
      f"{np.abs(h_seq.reshape(-1, Hl) - th_seq.detach().numpy().reshape(-1, Hl)).max():.4f} "
      f"(expected: our softsign gates are NOT tanh/sigmoid)")
