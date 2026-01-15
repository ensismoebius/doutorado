"""Modelo SNN simples em snnTorch para o demo de biometria por voz."""

try:
    import torch
    import torch.nn as nn
    import snntorch as snn
except ImportError:
    raise ImportError("As bibliotecas 'torch' e 'snntorch' são necessárias.")


class ModeloSNN(nn.Module):
    def __init__(
        self,
        num_inputs: int = 100,
        num_outputs: int | None = None,
        hidden: int = 100,
        beta: float = 0.9,
    ):
        super().__init__()
        self.num_entradas = num_inputs
        self.num_ocultos = hidden

        # Saída: por padrão, mantém o comportamento antigo (mesmo tamanho da entrada).
        # Para classificação (biometria por voz), use `num_outputs = num_classes`.
        self.num_saidas = int(num_outputs if num_outputs is not None else num_inputs)

        # Escala opcional da entrada.
        # Observação didática:
        # - Se você alimentar *spikes* na entrada (0/1), a escala atua como um ganho de corrente.
        # - Se você alimentar valores contínuos, a escala evita saturação.
        self.escala_entrada = 1.0

        # Camadas densas + neurônios LIF (Leaky Integrate-and-Fire).
        self.fc1 = nn.Linear(num_inputs, hidden)
        self.lif1 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = snn.Leaky(beta=beta)

        self.fc3 = nn.Linear(hidden, self.num_saidas)
        self.lif3 = snn.Leaky(beta=beta)

    def inicializar_estado(self, tamanho_lote: int, device=None, dtype=None):
        # Estado explícito permite processamento stateful entre janelas.
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return {
            "mem1": torch.zeros(
                tamanho_lote, self.num_ocultos, device=device, dtype=dtype
            ),
            "mem2": torch.zeros(
                tamanho_lote, self.num_ocultos, device=device, dtype=dtype
            ),
            "mem3": torch.zeros(
                tamanho_lote, self.num_saidas, device=device, dtype=dtype
            ),
        }

    def _forward_passo(self, x, state=None):
        # x: [lote, características] (pode ser contínuo ou spikes)
        # state: dicionário com memórias para propagação temporal

        if state is None:
            state = self.inicializar_estado(
                tamanho_lote=x.shape[0], device=x.device, dtype=x.dtype
            )

        mem1 = state["mem1"]
        mem2 = state["mem2"]
        mem3 = state["mem3"]

        # Entrada como corrente: maior valor -> maior corrente -> mais spikes.
        corrente_entrada = x * self.escala_entrada

        # Cada camada calcula corrente -> gera spike + atualiza memória.
        cur1 = self.fc1(corrente_entrada.float())
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)

        state = {"mem1": mem1, "mem2": mem2, "mem3": mem3}
        return spk3, state

    def forward(self, x, state=None):
        """Forward compatível com dois formatos de entrada.

        1) Passo único (como no demo antigo):
           - x: [lote, features]
           - retorna: (spk, state)

        2) Sequência temporal (para codificação Poisson/latência dentro da janela):
           - x: [T, lote, features]
           - retorna: (spk_seq, state)
             onde spk_seq tem shape [T, lote, num_saidas]
        """

        # Caso 1: passo único (sem dimensão temporal).
        if x.dim() == 2:
            return self._forward_passo(x, state)

        if x.dim() != 3:
            raise ValueError(
                "Entrada deve ter shape [lote, features] ou [T, lote, features]."
            )

        # Caso 2: sequência temporal de spikes.
        T = x.shape[0]
        spk_seq = []
        for t in range(T):
            spk_t, state = self._forward_passo(x[t], state)
            spk_seq.append(spk_t)
        return torch.stack(spk_seq, dim=0), state


def criar_modelo_snn(*, num_inputs: int = 100, num_outputs: int | None = None):
    # Regra de inicialização determinística (reprodutibilidade do demo).
    torch.manual_seed(42)
    return ModeloSNN(num_inputs=num_inputs, num_outputs=num_outputs)


# Wrappers (compatibilidade)
SNNModel = ModeloSNN
create_snn_model = criar_modelo_snn


if __name__ == "__main__":
    # Uma entrada de exemplo para teste rápido do modelo.
    model = criar_modelo_snn()
    input_data = torch.randn(1, 100)
    spk, state = model(input_data)
    print(spk)
