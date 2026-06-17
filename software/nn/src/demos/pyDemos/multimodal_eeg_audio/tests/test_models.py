"""Unit tests for multimodal_eeg_audio/models.py — DenseAutoencoder."""

import sys
from pathlib import Path

import pytest
import torch

# Make the package importable without install
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models import DenseAutoencoder  # noqa: E402


class TestDenseAutoencoder:
    INPUT_DIM = 32
    HIDDEN = (16,)
    LATENT = 8
    BATCH = 4

    @pytest.fixture()
    def model(self):
        return DenseAutoencoder(self.INPUT_DIM, self.HIDDEN, self.LATENT)

    def test_forward_output_shape(self, model):
        x = torch.rand(self.BATCH, self.INPUT_DIM)
        recon, z = model(x)
        assert recon.shape == (self.BATCH, self.INPUT_DIM)
        assert z.shape == (self.BATCH, self.LATENT)

    def test_latent_smaller_than_input(self, model):
        assert self.LATENT < self.INPUT_DIM

    def test_reconstruction_is_finite(self, model):
        x = torch.rand(self.BATCH, self.INPUT_DIM)
        recon, _ = model(x)
        assert torch.all(torch.isfinite(recon))

    def test_loss_decreases_over_ten_epochs(self, model):
        torch.manual_seed(0)
        x = torch.rand(self.BATCH, self.INPUT_DIM)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        first, last = None, None
        for epoch in range(10):
            recon, _ = model(x)
            loss = torch.nn.functional.mse_loss(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch == 0:
                first = loss.item()
            last = loss.item()
        assert last < first, f"Loss did not decrease: {first:.6f} → {last:.6f}"

    def test_encoder_output_shape(self, model):
        x = torch.rand(self.BATCH, self.INPUT_DIM)
        z = model.encoder(x)
        assert z.shape == (self.BATCH, self.LATENT)

    def test_different_batch_sizes(self, model):
        for b in (1, 8, 32):
            x = torch.rand(b, self.INPUT_DIM)
            recon, z = model(x)
            assert recon.shape == (b, self.INPUT_DIM)
            assert z.shape == (b, self.LATENT)
