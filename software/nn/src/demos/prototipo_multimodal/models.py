"""Modelos do protótipo: autoencoder denso e spiking (snnTorch)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    import snntorch as snn
    from snntorch import utils as snn_utils
except ImportError:
    snn = None
    snn_utils = None


class DenseAutoencoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: tuple[int, ...], latent_dim: int
    ) -> None:
        super().__init__()
        enc: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc.extend([nn.Linear(prev, h), nn.GELU()])
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec.extend([nn.Linear(prev, h), nn.GELU()])
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class ExponentialSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, sharpness: float) -> torch.Tensor:
        ctx.save_for_backward(inp)
        ctx.sharpness = sharpness
        return (inp > 0).to(inp.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (inp,) = ctx.saved_tensors
        s = float(ctx.sharpness)
        grad = (1.0 / s) * torch.exp(-torch.abs(inp) / s)
        return grad_output * grad, None


def exp_surrogate(sharpness: float):
    def _apply(x: torch.Tensor) -> torch.Tensor:
        return ExponentialSurrogate.apply(x, sharpness)

    return _apply


class SpikingAutoencoder(nn.Module):
    """Versão simples em snnTorch, compatível com ideia de LeakyBPTT.

    Entrada no forward: (B, F)
    Internamente repete T vezes e processa como (T, B, F).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...],
        time_steps: int,
        dt: float,
        resistance: float,
        capacitance: float,
        threshold: float,
        surrogate_sharpness: float,
    ) -> None:
        super().__init__()
        if snn is None or snn_utils is None:
            raise ImportError(
                "snnTorch não instalado. Instale com: pip install snntorch"
            )

        tau = resistance * capacitance
        beta = math.exp(-dt / tau)
        spike_grad = exp_surrogate(surrogate_sharpness)

        self.time_steps = int(time_steps)
        self.in_fc = nn.Linear(input_dim, hidden_dims[0])
        self.in_lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            init_hidden=True,
            reset_mechanism="zero",
        )

        self.mid_fc = nn.Linear(hidden_dims[0], latent_dim)
        self.mid_lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            init_hidden=True,
            reset_mechanism="zero",
        )

        self.out_fc = nn.Linear(latent_dim, input_dim)
        self.out_lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            init_hidden=True,
            output=True,
            reset_mechanism="none",
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        snn_utils.reset(self)
        x_t = x.unsqueeze(0).repeat(self.time_steps, 1, 1)

        latents = []
        recons = []
        for t in range(self.time_steps):
            cur = self.in_fc(x_t[t])
            spk = self.in_lif(cur)

            cur = self.mid_fc(spk)
            z = self.mid_lif(cur)

            cur = self.out_fc(z)
            _, mem = self.out_lif(cur)

            latents.append(z)
            recons.append(mem)

        # média temporal para sair no mesmo contrato (B,F) / (B,D)
        recon = torch.stack(recons, dim=0).mean(dim=0)
        z = torch.stack(latents, dim=0).mean(dim=0)
        return recon, z


def build_autoencoder(
    model_type: str,
    input_dim: int,
    latent_dim: int,
    hidden_dims: tuple[int, ...],
    snn_time_steps: int,
    snn_dt: float,
    snn_resistance: float,
    snn_capacitance: float,
    snn_threshold: float,
    snn_surrogate_sharpness: float,
) -> nn.Module:
    if model_type == "dense":
        return DenseAutoencoder(
            input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim
        )
    if model_type == "spiking":
        return SpikingAutoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            time_steps=snn_time_steps,
            dt=snn_dt,
            resistance=snn_resistance,
            capacitance=snn_capacitance,
            threshold=snn_threshold,
            surrogate_sharpness=snn_surrogate_sharpness,
        )
    raise ValueError(f"model_type inválido: {model_type}")
