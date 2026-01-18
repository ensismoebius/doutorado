"""snnTorch parity demo.

This script is a Python reference implementation used to compare against the C++
spiking autoencoder demo.

What it is for:
- Validate that the *training dynamics* (surrogate gradient shape, LIF parameters,
    and overall loss trend) are broadly consistent between C++ and snnTorch.

Important conventions:
- Inputs are treated as (T, B, F) = (time, batch, features).
- The surrogate gradient below is customized to mirror the C++ exponential surrogate:
    (1/sharpness) * exp(-|v - v_th| / sharpness).

Running:
- Typically executed from the repo root as part of manual comparisons.
- Requires `torch` and `snntorch` to be installed in the chosen Python environment.
"""

import torch
import torch.nn as nn
import snntorch as snn
import snntorch.utils
import numpy as np
import math
import os

# Custom Exponential Surrogate to match C++
# C++ implementation: (1.0/sharpness) * exp(-|v - threshold| / sharpness)
def exponential_surrogate(slope=1.0):
    class Exponential(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return (input > 0).float()
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            # Input is (mem - threshold)
            grad = (1/slope) * torch.exp(-torch.abs(input) / slope)
            return grad_input * grad
    return Exponential.apply

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Parameters matching C++
        self.input_dim = 100
        self.hidden_dim1 = 50
        self.hidden_dim2 = 40
        self.hidden_dim3 = 30
        self.hidden_dim4 = 20
        self.hidden_dim5 = 10
        self.bottleneck_dim = 10
        
        # Leaky parameters
        dt = 0.001
        R = 5.0
        C = 1.0
        tau = R * C
        self.beta = math.exp(-dt / tau)
        self.thresh = 0.01
        
        # Surrogate
        spike_grad = exponential_surrogate(slope=1.0) # sharpness=1.0 in C++
        
        # Initialization Helper
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                # Apply the 0.01 scaling used in C++ fix
                with torch.no_grad():
                    m.weight.data *= 0.01

        # Encoder Layers
        # Linear -> Leaky
        self.enc1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.lif1 = snn.Leaky(beta=self.beta, threshold=self.thresh, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero')
        
        self.enc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.lif2 = snn.Leaky(beta=self.beta, threshold=self.thresh, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero')
        
        self.enc3 = nn.Linear(self.hidden_dim2, self.hidden_dim3)
        self.lif3 = snn.Leaky(beta=self.beta, threshold=self.thresh, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero')
        
        self.enc4 = nn.Linear(self.hidden_dim3, self.hidden_dim4)
        self.lif4 = snn.Leaky(beta=self.beta, threshold=self.thresh, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero')
        
        self.enc5 = nn.Linear(self.hidden_dim4, self.hidden_dim5)
        self.lif5 = snn.Leaky(beta=self.beta, threshold=self.thresh, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero')
        
        self.enc6 = nn.Linear(self.hidden_dim5, self.bottleneck_dim)
        self.lif6 = snn.Leaky(beta=self.beta, threshold=self.thresh, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero')
        
        # Decoder Layers
        self.dec1 = nn.Linear(self.bottleneck_dim, self.hidden_dim5)
        self.lif_dec1 = snn.Leaky(beta=self.beta, threshold=self.thresh, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero')
        
        self.dec2 = nn.Linear(self.hidden_dim5, self.hidden_dim4)
        self.lif_dec2 = snn.Leaky(beta=self.beta, threshold=self.thresh, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero')
        
        self.dec3 = nn.Linear(self.hidden_dim4, self.hidden_dim3)
        self.lif_dec3 = snn.Leaky(beta=self.beta, threshold=self.thresh, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero')
        
        self.dec4 = nn.Linear(self.hidden_dim3, self.hidden_dim2)
        self.lif_dec4 = snn.Leaky(beta=self.beta, threshold=self.thresh, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero')
        
        self.dec5 = nn.Linear(self.hidden_dim2, self.hidden_dim1)
        self.lif_dec5 = snn.Leaky(beta=self.beta, threshold=self.thresh, spike_grad=spike_grad, init_hidden=True, reset_mechanism='zero')
        
        self.dec6 = nn.Linear(self.hidden_dim1, self.input_dim)
        # Final Readout: Output=True, No Reset (effectively), Learn Beta=False
        # In C++: reset=false.
        self.lif_dec6 = snn.Leaky(beta=self.beta, threshold=self.thresh, spike_grad=spike_grad, init_hidden=True, output=True, reset_mechanism='none')
        
        # Apply initialization
        self.apply(init_weights)

    def forward(self, x):
        # x shape: (Time, Batch, Features)
        
        # Initialize hidden states
        snn.utils.reset(self)
        
        spk1_rec = []
        spk2_rec = []
        spk3_rec = []
        spk4_rec = []
        spk5_rec = []
        spk6_rec = []
        
        spk_dec1_rec = []
        spk_dec2_rec = []
        spk_dec3_rec = []
        spk_dec4_rec = []
        spk_dec5_rec = []
        mem_dec6_rec = [] # Readout
        
        time_steps = x.size(0)
        
        for step in range(time_steps):
            cur1 = self.enc1(x[step])
            spk1 = self.lif1(cur1)
            
            cur2 = self.enc2(spk1)
            spk2 = self.lif2(cur2)
            
            cur3 = self.enc3(spk2)
            spk3 = self.lif3(cur3)
            
            cur4 = self.enc4(spk3)
            spk4 = self.lif4(cur4)
            
            cur5 = self.enc5(spk4)
            spk5 = self.lif5(cur5)
            
            cur6 = self.enc6(spk5)
            spk6 = self.lif6(cur6)
            
            # Decoder
            cur_dec1 = self.dec1(spk6)
            spk_dec1 = self.lif_dec1(cur_dec1)
            
            cur_dec2 = self.dec2(spk_dec1)
            spk_dec2 = self.lif_dec2(cur_dec2)
            
            cur_dec3 = self.dec3(spk_dec2)
            spk_dec3 = self.lif_dec3(cur_dec3)
            
            cur_dec4 = self.dec4(spk_dec3)
            spk_dec4 = self.lif_dec4(cur_dec4)
            
            cur_dec5 = self.dec5(spk_dec4)
            spk_dec5 = self.lif_dec5(cur_dec5)
            
            cur_dec6 = self.dec6(spk_dec5)
            spk_final, mem_final = self.lif_dec6(cur_dec6)
            
            mem_dec6_rec.append(mem_final)
            
        return torch.stack(mem_dec6_rec, dim=0)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters matches C++:
    batch_size = 32 # Not used 
    n_samples = 10 
    n_steps = 100
    input_dim = 100
    epochs = 400
    lr = 0.001
    
    device = torch.device("cpu") 
    
    model = AutoEncoder().to(device)
    
    if not os.path.exists("weights"):
        os.makedirs("weights")
        
    enc_weights = {}
    dec_weights = {}
    
    layer_map_enc = {
        'enc1': '0', 'enc2': '2', 'enc3': '4', 'enc4': '6', 'enc5': '8', 'enc6': '10'
    }
    layer_map_dec = {
        'dec1': '0', 'dec2': '2', 'dec3': '4', 'dec4': '6', 'dec5': '8', 'dec6': '10'
    }
            
    for name, param in model.named_parameters():
        parts = name.split('.')
        layer_name = parts[0]
        type_name = parts[1]
        
        val = param.detach().numpy()
        
        if layer_name in layer_map_enc:
            key_w = f"{layer_map_enc[layer_name]}.weight"
            key_b = f"{layer_map_enc[layer_name]}.bias"
            if type_name == 'weight':
                 enc_weights[key_w] = val
            elif type_name == 'bias':
                 enc_weights[key_b] = val
                 
        elif layer_name in layer_map_dec:
            key_w = f"{layer_map_dec[layer_name]}.weight"
            key_b = f"{layer_map_dec[layer_name]}.bias"
            if type_name == 'weight':
                 dec_weights[key_w] = val
            elif type_name == 'bias':
                 dec_weights[key_b] = val

    np.savez("weights/encoder_spike_model_weights.npz", **enc_weights)
    np.savez("weights/decoder_spike_model_weights.npz", **dec_weights)
    print("Weights saved to weights/ directory.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    inputs = torch.ones((n_steps, n_samples, input_dim)).to(device)
    targets = inputs.clone()
    
    print(f"Starting Training: Epochs={epochs}, LR={lr}")
    
    log_file = open("python_loss_log.txt", "w")

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if epoch % 10 == 0 or epoch < 5:
            # Calculate grad norm for comparison
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Epoch {epoch} Grad Norm: {total_norm:.6f}")
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {loss.item():.6f}")
            
        log_file.write(f"{epoch},{loss.item()}\n")
            
    print(f"Final Loss: {loss.item():.6f}")
    log_file.close()

if __name__ == "__main__":
    main()
