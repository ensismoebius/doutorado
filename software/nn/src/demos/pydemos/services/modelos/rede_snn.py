"""Modelo SNN simples em snnTorch para o demo de biometria por voz."""

try:
    import torch
    import torch.nn as nn
    import snntorch as snn
except ImportError:
    raise ImportError("As bibliotecas 'torch' e 'snntorch' são necessárias.")


# Bloco residual para SNN: Linear -> LIF -> Linear -> LIF + skip connection
class ResidualSNNBlock(nn.Module):
    def __init__(self, dim: int, beta: float = 0.9):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(dim, dim)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x, mem1, mem2):
        out = self.fc1(x)
        spk1, mem1 = self.lif1(out, mem1)
        out = self.fc2(spk1)
        spk2, mem2 = self.lif2(out, mem2)
        # Residual connection: soma entrada original após segunda LIF
        return spk2 + x, mem1, mem2


class ModeloSNN(nn.Module):
    def __init__(
        self,
        tamanho_da_entrada: int = 100,
        quantidade_de_saidas: int | None = None,
        tamanho_da_camada_escondida: int = 100,
        beta: float = 0.9,
        numero_de_blocos_residuais: int = 3,
    ):
        super().__init__()
        self.num_entradas = tamanho_da_entrada
        self.num_ocultos = tamanho_da_camada_escondida

        if quantidade_de_saidas is None or quantidade_de_saidas <= 0:
            raise ValueError("`quantidade_de_saidas` não pode ser None ou <= 0.")

        self.num_saidas = quantidade_de_saidas

        self.escala_entrada = 1.0

        # Camada de entrada
        self.fc_in = nn.Linear(tamanho_da_entrada, tamanho_da_camada_escondida)
        self.lif_in = snn.Leaky(beta=beta)

        # Blocos residuais profundos
        self.num_blocos_residuais = numero_de_blocos_residuais
        self.res_blocks = nn.ModuleList(
            [
                ResidualSNNBlock(tamanho_da_camada_escondida, beta=beta)
                for _ in range(numero_de_blocos_residuais)
            ]
        )

        # Camada de saída
        self.fc_out = nn.Linear(tamanho_da_camada_escondida, self.num_saidas)
        self.lif_out = snn.Leaky(beta=beta)

    def inicializar_estado(self, tamanho_lote: int, device=None, dtype=None):
        # Estado explícito para todas as memórias dos blocos residuais
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        state = {
            "mem_in": torch.zeros(
                tamanho_lote, self.num_ocultos, device=device, dtype=dtype
            ),
            "mem_out": torch.zeros(
                tamanho_lote, self.num_saidas, device=device, dtype=dtype
            ),
        }

        # Para cada bloco residual, duas memórias (uma para cada LIF)
        for i in range(self.num_blocos_residuais):
            state[f"mem_res{i}_1"] = torch.zeros(
                tamanho_lote, self.num_ocultos, device=device, dtype=dtype
            )
            state[f"mem_res{i}_2"] = torch.zeros(
                tamanho_lote, self.num_ocultos, device=device, dtype=dtype
            )
        return state

    def _forward_passo(self, x, state=None):
        # x: [lote, características] (pode ser contínuo ou spikes)
        if state is None:
            state = self.inicializar_estado(
                tamanho_lote=x.shape[0], device=x.device, dtype=x.dtype
            )

        corrente_entrada = x * self.escala_entrada
        mem_in = state["mem_in"]
        out = self.fc_in(corrente_entrada.float())
        spk, mem_in = self.lif_in(out, mem_in)

        # Passa por todos os blocos residuais
        for i, bloco in enumerate(self.res_blocks):
            mem1 = state[f"mem_res{i}_1"]
            mem2 = state[f"mem_res{i}_2"]
            spk, mem1, mem2 = bloco(spk, mem1, mem2)
            state[f"mem_res{i}_1"] = mem1
            state[f"mem_res{i}_2"] = mem2

        mem_out = state["mem_out"]
        out = self.fc_out(spk)
        spk_out, mem_out = self.lif_out(out, mem_out)
        state["mem_in"] = mem_in
        state["mem_out"] = mem_out
        return spk_out, state

    def forward(self, x, estado=None):
        """Forward compatível com 2 formatos de entrada (2D ou 3D).

        Resumo
        -----
        - Aceita um único passo (2D) ou uma sequência temporal (3D).
        - Retorna uma tupla ``(spikes, state)`` onde ``state`` é o dicionário de
          memórias atualizado (mapeado por amostra do batch).

        Formatos
        -------
        1) Passo único (2D)
           - Entrada: ``x`` com shape ``(batch, features)``.
             Ex.: ``(8, 100)`` para ``batch=8`` e ``features=100``.
           - Saída: ``spk`` com shape ``(batch, num_saidas)``.
           - Exemplo rápido::

             x = torch.randn(8, 100)         # shape (8,100)
             spk, state = model(x)
             # spk.shape == (8, num_saidas)

        2) Sequência temporal (3D)
           - Entrada: ``x`` com shape ``(num_passos, batch, features)``.
             Ex.: ``(10, 8, 100)`` para 10 passos, batch=8, features=100.
           - Saída: ``spk_seq`` com shape ``(num_passos, batch, num_saidas)``.
           - Exemplo rápido::

             # exemplo concreto com num_passos=3, batch=2, features=4
             x = torch.tensor([
               [[0,0,1,0], [1,0,0,0]],   # t=0 -> shape (2,4)
               [[0,1,0,0], [0,0,1,0]],   # t=1
               [[1,0,0,0], [0,1,0,1]],   # t=2
             ])  # shape (3,2,4)
             spk_seq, state = model(x)
             # spk_seq.shape == (3,2,num_saidas)

        Observações
        ----------
        - ``state`` (opcional) permite continuação da dinâmica entre janelas;
          se passado, o modelo reusa as memórias; caso contrário, o estado é
          inicializado com zeros via ``inicializar_estado``.
        - Internamente a operação em um passo é a mesma usada em cada passo da
          sequência temporal; na versão temporal iteramos sobre o eixo 0.
        """

        # Caso 1: passo único (sem dimensão temporal).
        # Aqui o `x` já representa o vetor de features do lote atual.
        # Ex.: x.shape == (8, 100)
        if x.dim() == 2:
            return self._forward_passo(x, estado)

        if x.dim() != 3:
            raise ValueError(
                "Entrada deve ter shape [lote, features] ou [T, lote, features]."
            )

        # Caso 2: sequência temporal de spikes.
        # Iteramos ao longo do tempo e acumulamos a saída em `spk_seq`.
        # Ex.: x.shape == (10, 8, 100) -> spk_seq terá 10 tensores [8, num_saidas].
        num_passos = x.shape[0]
        sequencia_de_pulsos = []
        for passo_index in range(num_passos):
            # Cada passo reutiliza o estado atualizado da etapa anterior.
            pulsos_t, estado = self._forward_passo(x[passo_index], estado)
            sequencia_de_pulsos.append(pulsos_t)
        return torch.stack(sequencia_de_pulsos, dim=0), estado


def criar_modelo_snn(
    *,
    numero_de_entradas: int = 100,
    numero_de_saidas: int | None = None,
    tamanho_da_camada_escondida: int = 100,
    qtde_de_blocos_residuais: int = 3,
):
    # Regra de inicialização determinística (reprodutibilidade do demo).
    torch.manual_seed(42)

    # Parâmetros padrão.
    if qtde_de_blocos_residuais is None:
        qtde_de_blocos_residuais = 3

    # Cria e retorna o modelo SNN.
    return ModeloSNN(
        tamanho_da_entrada=numero_de_entradas,
        quantidade_de_saidas=numero_de_saidas,
        tamanho_da_camada_escondida=tamanho_da_camada_escondida,
        numero_de_blocos_residuais=int(qtde_de_blocos_residuais),
    )


if __name__ == "__main__":
    # Uma entrada de exemplo para teste rápido do modelo.
    model = criar_modelo_snn()
    input_data = torch.randn(1, 100)
    spk, state = model(input_data)
    print(spk)
