"""Optional PyTorch reference implementation of the STE trick.

Not imported by the GUI or by any DemoModule — the application runs
entirely on numpy (see bitnet/ste.py). This module exists only as the
"implementação representativa" PyTorch fragment from
ESPECIFICACAO_DLVL.md #13/#14, kept runnable on its own for anyone who
wants to see the same idea expressed with autograd:

    python -m efficient_nn_lab.bitnet.ste_torch_reference

Requires the optional `torch-reference` extra (`pip install
.[torch-reference]`); it is not a dependency of the base package.
"""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only when torch absent
    raise ImportError(
        "ste_torch_reference requires PyTorch. Install with: "
        "pip install '.[torch-reference]'"
    ) from exc


def ternary_quantize(w: "torch.Tensor", threshold: float = 0.5) -> "torch.Tensor":
    q = torch.zeros_like(w)
    q = torch.where(w > threshold, torch.ones_like(w), q)
    q = torch.where(w < -threshold, -torch.ones_like(w), q)
    return q


def ste_quantize(w: "torch.Tensor", threshold: float = 0.5) -> "torch.Tensor":
    """Forward: q. Backward: gradient w.r.t. w behaves like the identity.

    `w + (q - w).detach()` is numerically equal to `q` (since `q - q = 0`
    contributes nothing extra), but because `(q - w).detach()` is treated
    as a constant by autograd, the gradient of this expression w.r.t. `w`
    is exactly 1 — the STE.
    """
    q = ternary_quantize(w, threshold)
    return w + (q - w).detach()


def run_scalar_example() -> None:
    """Reproduces ESPECIFICACAO_DLVL.md #14 exactly and prints every value."""
    w = torch.tensor(0.8, requires_grad=True)
    target = torch.tensor(4.0)
    x = torch.tensor(2.0)

    q = ste_quantize(w)
    y = x * q

    loss = 0.5 * (y - target) ** 2
    loss.backward()

    print("w real:", w.item())
    print("w quantizado:", q.item())
    print("y:", y.item())
    print("loss:", loss.item())
    print("grad:", w.grad.item())


if __name__ == "__main__":
    run_scalar_example()
