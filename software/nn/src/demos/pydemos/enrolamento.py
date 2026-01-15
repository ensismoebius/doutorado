"""Compatibilidade: o módulo foi renomeado para `cadastro`.

Este arquivo permanece apenas para não quebrar imports antigos.
Use `from cadastro import capturar_e_salvar_amostra`.
"""

# Reexporta a função principal para não quebrar código antigo.
from cadastro import capturar_e_salvar_amostra  # noqa: F401
