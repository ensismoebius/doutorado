try:
    import torch
    import torch.nn as nn
    import snntorch as snn
except ImportError:
    raise ImportError("As bibliotecas 'torch' e 'snntorch' são necessárias.")

class SNNModel(nn.Module):
    def __init__(self, num_inputs=100, hidden=100, beta=0.9):
        super().__init__()
        self.num_inputs = num_inputs
        self.hidden = hidden
        
        # Encoding: Direct Current Injection scale
        # Como as energias Wavelet podem ser grandes, podemos ter um fator de escala
        self.input_scale = 0.1 

        # Initialize weights deterministically
        self.fc1 = nn.Linear(num_inputs, hidden)
        self.lif1 = snn.Leaky(beta=beta)
        
        self.fc2 = nn.Linear(hidden, hidden)
        self.lif2 = snn.Leaky(beta=beta)
        
        self.fc3 = nn.Linear(hidden, num_inputs) 
        self.lif3 = snn.Leaky(beta=beta)

    def init_state(self, batch_size: int, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return {
            "mem1": torch.zeros(batch_size, self.hidden, device=device, dtype=dtype),
            "mem2": torch.zeros(batch_size, self.hidden, device=device, dtype=dtype),
            "mem3": torch.zeros(batch_size, self.num_inputs, device=device, dtype=dtype),
        }

    def forward(self, x, state=None):
        # x: [Batch, Features] (Energy Vector)
        # state: dicionário com memórias (estado interno) para propagação temporal

        if state is None:
            state = self.init_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        mem1 = state["mem1"]
        mem2 = state["mem2"]
        mem3 = state["mem3"]

        # Encoding: Scale input current directly
        # x representa a energia da banda. Maior energia -> Maior corrente -> Mais spikes.
        current_input = x * self.input_scale

        cur1 = self.fc1(current_input.float())
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc3(spk2)
        spk3, mem3 = self.lif3(cur3, mem3)

        state = {"mem1": mem1, "mem2": mem2, "mem3": mem3}
        return spk3, state

def create_snn_model():
    # Deterministic Initialization Rule
    torch.manual_seed(42)
    return SNNModel()

if __name__ == "__main__":
    # Uma entrada de exemplo para teste
    model = create_snn_model()
    input_data = torch.randn(1, 100)
    spk, state = model(input_data)
    print(spk)
