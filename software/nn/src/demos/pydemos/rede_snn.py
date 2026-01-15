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
        tamanho_da_entrada: int = 100,
        quantidade_de_classes: int | None = None,
        tamanho_da_camada_escondida: int = 100,
        beta: float = 0.9,
    ):
        super().__init__()
        self.num_entradas = tamanho_da_entrada
        self.num_ocultos = tamanho_da_camada_escondida

        # Saída: por padrão, mantém o comportamento antigo (mesmo tamanho da entrada).
        # Para classificação (biometria por voz), use `num_outputs = num_classes`.
        if quantidade_de_classes is None or quantidade_de_classes <= 0:
            raise ValueError("`quantidade_de_classes` não pode ser None ou <= 0.")

        self.num_saidas = quantidade_de_classes

        # Escala opcional da entrada (ganho de corrente).
        # - Se `x` já é spike (0/1), controla quanta corrente cada spike injeta.
        #   Ex.: x=1.0 e escala=0.5 -> corrente=0.5; escala=2.0 -> corrente=2.0.
        # - Se `x` é contínuo (ex.: [0,1]), evita saturação (muito alto) ou inatividade
        #   (muito baixo) dos neurônios LIF.
        # - Valores > 1.0 tendem a aumentar a taxa de spikes; < 1.0 reduzem a atividade.
        self.escala_entrada = 1.0

        # Camadas densas + neurônios LIF (Leaky Integrate-and-Fire).
        self.fc1 = nn.Linear(tamanho_da_entrada, tamanho_da_camada_escondida)
        self.lif1 = snn.Leaky(beta=beta)

        self.fc2 = nn.Linear(tamanho_da_camada_escondida, tamanho_da_camada_escondida)
        self.lif2 = snn.Leaky(beta=beta)

        self.fc3 = nn.Linear(tamanho_da_camada_escondida, self.num_saidas)
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
        # Ex.: lote=2 e features=4 -> x.shape == (2, 4)
        # Exemplo de valores (matriz):
        # x = torch.tensor([
        #   [0.1, 0.0, 1.0, 0.5],  # amostra 0 (4 features)
        #   [0.0, 1.0, 0.2, 0.3],  # amostra 1 (4 features)
        # ])  # shape -> (2, 4)
        #
        # state: dicionário com memórias para propagação temporal
        # Exemplo de `state` para lote=2 e num_ocultos=3, num_saidas=5:
        # state = {
        #   'mem1': torch.zeros(2, 3),  # shape (lote, num_ocultos)
        #   'mem2': torch.zeros(2, 3),  # shape (lote, num_ocultos)
        #   'mem3': torch.zeros(2, 5),  # shape (lote, num_saidas)
        # }

        if state is None:
            state = self.inicializar_estado(
                tamanho_lote=x.shape[0], device=x.device, dtype=x.dtype
            )

        mem1 = state["mem1"]
        mem2 = state["mem2"]
        mem3 = state["mem3"]

        # Entrada como corrente: maior valor -> maior corrente -> mais spikes.
        # Ex.: x=0.3 e escala=1.0 -> corrente=0.3; escala=2.0 -> corrente=0.6.
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


def criar_modelo_snn(*, num_inputs: int = 100, num_outputs: int | None = None):
    # Regra de inicialização determinística (reprodutibilidade do demo).
    torch.manual_seed(42)
    return ModeloSNN(tamanho_da_entrada=num_inputs, quantidade_de_classes=num_outputs)


if __name__ == "__main__":
    # Uma entrada de exemplo para teste rápido do modelo.
    model = criar_modelo_snn()
    input_data = torch.randn(1, 100)
    spk, state = model(input_data)
    print(spk)
