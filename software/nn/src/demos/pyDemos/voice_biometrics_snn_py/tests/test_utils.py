"""Unit tests for voice_biometrics_snn_py/utils — codificacao and janelamento."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.codificacao import calcular_taxa_max_adaptativa, codificar_poisson  # noqa: E402


class TestCalcularTaxaMaxAdaptativa:
    def test_mean_half_gives_approx_point_two(self):
        sinal = torch.full((1, 16), 0.5)
        rate = calcular_taxa_max_adaptativa(sinal, qtde_de_spikes_esperada_por_passo=0.1)
        assert abs(rate - 0.2) < 1e-4

    def test_near_zero_returns_max_max_rate(self):
        sinal = torch.zeros(1, 16)
        rate = calcular_taxa_max_adaptativa(
            sinal, frequencia_max_max=0.5, qtde_de_spikes_esperada_por_passo=0.1
        )
        assert abs(rate - 0.5) < 1e-6

    def test_result_always_in_bounds(self):
        for v in [0.0, 0.1, 0.5, 0.9, 1.0]:
            sinal = torch.full((1, 8), v)
            rate = calcular_taxa_max_adaptativa(sinal)
            assert 0.02 <= rate <= 0.50, f"Out of bounds for mean={v}: {rate}"

    def test_high_expected_clamped_to_max(self):
        sinal = torch.full((1, 4), 0.5)
        rate = calcular_taxa_max_adaptativa(
            sinal, qtde_de_spikes_esperada_por_passo=10.0, frequencia_max_max=0.5
        )
        assert abs(rate - 0.5) < 1e-6


class TestCodificarPoisson:
    FEATURES = 16
    BATCH = 2
    STEPS = 20

    @pytest.fixture()
    def frame(self):
        return torch.full((self.BATCH, self.FEATURES), 0.5)

    def test_output_shape(self, frame):
        spikes = codificar_poisson(frame, passos=self.STEPS)
        assert spikes.shape == (self.STEPS, self.BATCH, self.FEATURES)

    def test_values_are_binary(self, frame):
        spikes = codificar_poisson(frame, passos=self.STEPS)
        assert torch.all((spikes == 0) | (spikes == 1))

    def test_invalid_steps_raises(self, frame):
        with pytest.raises(ValueError):
            codificar_poisson(frame, passos=0)

    def test_zero_input_no_spikes(self):
        zero = torch.zeros(1, self.FEATURES)
        # With fixed max rate=0, p=0 → never fires
        spikes = codificar_poisson(zero, passos=50, frequencia_max=0.0, adaptativo=False)
        assert spikes.sum().item() == 0

    def test_adaptive_density_near_expected(self):
        frame = torch.full((1, 64), 0.5)
        spikes = codificar_poisson(
            frame, passos=500, adaptativo=True, qtde_de_spikes_esperada_por_passo=0.1
        )
        density = spikes.float().mean().item()
        assert 0.06 < density < 0.14, f"Spike density {density:.4f} not near 0.10"

    def test_single_step_output_shape(self, frame):
        spikes = codificar_poisson(frame, passos=1)
        assert spikes.shape == (1, self.BATCH, self.FEATURES)
