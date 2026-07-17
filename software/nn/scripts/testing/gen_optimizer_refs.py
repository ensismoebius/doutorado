#!/usr/bin/env python3
"""Generate optimizer ground-truth fixtures for the C++ parity tests.

For each optimizer this drives the *reference* implementation (PyTorch's own, or the
authors' released package) over a fixed sequence of seeded parameters and gradients, and
records the parameter value after every step. The C++ test `optimizer_parity_gtest` replays
the identical parameters/gradients through our port and asserts EXPECT_NEAR against these.

Why this exists: these optimizers are easy to implement *plausibly* and wrong -- the operation
order in Lion (decay before update, momentum advanced after) and the three-sequence structure
of Schedule-Free are exactly the details that a from-memory port silently gets wrong. Diffing
against the authors' own code is what makes the port trustworthy rather than merely tidy.

Covered (reference -> our class):
  torch.optim.Adam      (weight_decay=0)  -> Adam
  torch.optim.AdamW                       -> Adam with weight_decay>0 (decoupled)
  torch.optim.SGD                         -> SGD
  lion_pytorch.Lion                       -> Lion
  schedulefree.AdamWScheduleFreeReference -> ScheduleFreeAdamW

Deliberate deviation from upstream, encoded here on purpose (see each C++ header):
  * weight_decay: this project applies decoupled decay only to 2-D weight matrices, so that
    SNN biophysical scalars (R, C, V_th -- 1x1) never have tau=R*C or the threshold decayed.
    Every decay case below therefore uses a 2-D parameter, where the two agree exactly.

Run (developer step; requires torch + the reference packages -- CI consumes the committed .npz):
    software/nn/.venv/bin/python software/nn/scripts/testing/gen_optimizer_refs.py

Output: software/nn/src/core/optimizers/tests/fixtures/optimizer_refs.npz
"""
import os

import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "src", "core", "optimizers", "tests", "fixtures",
                   "optimizer_refs.npz")
OUT = os.path.normpath(OUT)

A = {}


def put(name, arr):
    # MUST deep-copy. np.asarray() on a torch tensor shares its storage, and the optimizers
    # mutate parameters in place -- without copy=True every recorded step aliases the final
    # parameter value, and the fixture silently becomes "the last step, N times".
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    A[name] = np.array(arr, dtype=np.float32, copy=True)


def run_case(prefix, make_opt, shape, steps=5, seed=0, grad_scale=1.0):
    """Drive `make_opt` over `steps` fixed gradients; record params after each step.

    Saves: <prefix>_p0 (initial param), <prefix>_g{k} (gradient at step k),
           <prefix>_p{k+1} (param after step k), <prefix>_steps, <prefix>_shape.
    """
    g = torch.Generator().manual_seed(seed)
    p0 = torch.randn(shape, generator=g, dtype=torch.float32) * 0.5
    grads = [torch.randn(shape, generator=g, dtype=torch.float32) * grad_scale
             for _ in range(steps)]

    p = torch.nn.Parameter(p0.clone())
    opt = make_opt([p])

    put(f"{prefix}_p0", p0)
    put(f"{prefix}_shape", np.array(list(shape), np.float32))
    put(f"{prefix}_steps", np.array([steps], np.float32))

    for k, gr in enumerate(grads):
        put(f"{prefix}_g{k}", gr)
        opt.zero_grad()
        p.grad = gr.clone()
        opt.step()
        put(f"{prefix}_p{k+1}", p.detach())
    return opt


# ── Adam / AdamW / SGD (PyTorch's own implementations) ────────────────────────
run_case("adam_2x3", lambda ps: torch.optim.Adam(ps, lr=0.01, betas=(0.9, 0.999), eps=1e-8),
         (2, 3))
# AdamW == our Adam with decoupled weight_decay, on a 2-D matrix (see header note).
run_case("adamw_2x3", lambda ps: torch.optim.AdamW(ps, lr=0.01, betas=(0.9, 0.999), eps=1e-8,
                                                   weight_decay=0.1), (2, 3))
run_case("sgd_2x3", lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.0), (2, 3))
run_case("sgdm_2x3", lambda ps: torch.optim.SGD(ps, lr=0.05, momentum=0.9), (2, 3))

# ── Lion (authors' lion-pytorch) ──────────────────────────────────────────────
from lion_pytorch import Lion  # noqa: E402

run_case("lion_2x3", lambda ps: Lion(ps, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0), (2, 3))
run_case("lion_wd_2x3",
         lambda ps: Lion(ps, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1), (2, 3))

# ── Schedule-Free AdamW (authors' schedulefree, reference variant) ────────────
from schedulefree import AdamWScheduleFreeReference  # noqa: E402


def _sf(ps, **kw):
    o = AdamWScheduleFreeReference(ps, **kw)
    o.train()
    return o


run_case("sfadamw_2x3",
         lambda ps: _sf(ps, lr=0.0025, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
         (2, 3))
run_case("sfadamw_wd_2x3",
         lambda ps: _sf(ps, lr=0.0025, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1),
         (2, 3))
run_case("sfadamw_warm_2x3",
         lambda ps: _sf(ps, lr=0.0025, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                        warmup_steps=3),
         (2, 3))

# Schedule-Free eval iterate (x): train for N steps, then eval() and record x.
# This pins the train/eval swap our OptimizerEvalScope relies on.
_g = torch.Generator().manual_seed(7)
_p0 = torch.randn((2, 3), generator=_g, dtype=torch.float32) * 0.5
_grads = [torch.randn((2, 3), generator=_g, dtype=torch.float32) for _ in range(5)]
_p = torch.nn.Parameter(_p0.clone())
_o = _sf([_p], lr=0.0025, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
put("sfeval_p0", _p0)
put("sfeval_steps", np.array([len(_grads)], np.float32))
for k, gr in enumerate(_grads):
    put(f"sfeval_g{k}", gr)
    _o.zero_grad()
    _p.grad = gr.clone()
    _o.step()
put("sfeval_y", _p.detach())   # training iterate after the last step
_o.eval()
put("sfeval_x", _p.detach())   # averaged/evaluation iterate

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, **A)
print(f"wrote {OUT} with {len(A)} arrays")
