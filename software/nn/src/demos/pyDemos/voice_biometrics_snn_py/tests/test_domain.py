"""Unit tests for voice_biometrics_snn_py/domain/regras.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain.regras import aplicar_limiar_desconhecido  # noqa: E402


class TestAplicarLimiarDesconhecido:
    def test_above_threshold_returns_person(self):
        result = aplicar_limiar_desconhecido("alice", 0.9, limiar=0.5)
        assert result == "alice"

    def test_below_threshold_returns_unknown(self):
        result = aplicar_limiar_desconhecido("alice", 0.3, limiar=0.5)
        assert result == "desconhecido"

    def test_at_threshold_returns_person(self):
        # confianca == limiar: NOT below → identity returned
        result = aplicar_limiar_desconhecido("bob", 0.5, limiar=0.5)
        assert result == "bob"

    def test_custom_unknown_label(self):
        result = aplicar_limiar_desconhecido("carol", 0.1, limiar=0.5, rotulo_desconhecido="??")
        assert result == "??"

    def test_zero_confidence_is_unknown(self):
        result = aplicar_limiar_desconhecido("dave", 0.0, limiar=0.01)
        assert result == "desconhecido"

    def test_threshold_zero_never_unknown(self):
        # Any confidence >= 0 is above threshold=0
        result = aplicar_limiar_desconhecido("eve", 0.0, limiar=0.0)
        assert result == "eve"

    def test_threshold_one_always_unknown(self):
        # No confidence < 1 triggers unknown
        result = aplicar_limiar_desconhecido("frank", 0.9999, limiar=1.0)
        assert result == "desconhecido"
