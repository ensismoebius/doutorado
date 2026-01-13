try:
    import torch
    import torch.nn as nn
    import snntorch as snn
except ImportError:
    raise ImportError("As bibliotecas 'torch' e 'snntorch' são necessárias.")

class ModeloSNN(nn.Module):
    def __init__(self, num_inputs=100, hidden=100, beta=0.9):
        super().__init__()
        self.num_entradas = num_inputs
        self.num_ocultos = hidden
        
        # Codificação (injeção direta de corrente):
        # Como as energias do WPT podem ter faixa dinâmica grande, usamos um fator de escala.
        self.escala_entrada = 0.1 

        # Camadas
        self.fc1 = nn.Linear(num_inputs, hidden)
        self.lif1 = snn.Leaky(beta=beta)
        
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = snn.Leaky(beta=beta)
        
        self.fc3 = nn.Linear(hidden, num_inputs) 
        self.lif3 = snn.Leaky(beta=beta)

    def inicializar_estado(self, tamanho_lote: int, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return {
            "mem1": torch.zeros(tamanho_lote, self.num_ocultos, device=device, dtype=dtype),
            "mem2": torch.zeros(tamanho_lote, self.num_ocultos, device=device, dtype=dtype),
            "mem3": torch.zeros(tamanho_lote, self.num_entradas, device=device, dtype=dtype),
        }

    def forward(self, x, state=None):
        # x: [lote, características] (vetor de energia)
        # state: dicionário com memórias para propagação temporal (stateful)

        if state is None:
            state = self.inicializar_estado(
                tamanho_lote=x.shape[0], device=x.device, dtype=x.dtype
            )

        mem1 = state["mem1"]
        mem2 = state["mem2"]
        mem3 = state["mem3"]

        # Codificação: maior energia -> maior corrente -> mais spikes.
        corrente_entrada = x * self.escala_entrada

        cur1 = self.fc1(corrente_entrada.float())
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)

        state = {"mem1": mem1, "mem2": mem2, "mem3": mem3}
        return spk3, state

def criar_modelo_snn():
    # Regra de inicialização determinística
    torch.manual_seed(42)
    return ModeloSNN()


# Wrappers (compatibilidade)
SNNModel = ModeloSNN
create_snn_model = criar_modelo_snn

if __name__ == "__main__":
    # Uma entrada de exemplo para teste
    model = criar_modelo_snn()
    input_data = torch.randn(1, 100)
    spk, state = model(input_data)
    print(spk)
