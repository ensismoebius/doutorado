# Multi-Pass Forward

**Contexto:**
Estou trabalhando em um projeto Python de **classificação de locutor usando Spiking Neural Networks (SNN)** em PyTorch/snntorch. O código contém as funções:

* `treinar_classificador_locutor`
* `identificar_locutor_por_microfone`
* `identificar_locutor_por_wav`

Atualmente, o *loss* é calculado assim:

```python
spk_out_seq, _ = model(spk_in, None)
contagem = spk_out_seq.sum(dim=0)
loss = F.cross_entropy(contagem, yb)
```

Quero **refatorar o código** para que:

1. **O método de cálculo do loss seja configurável via CLI** (`--loss_mode`)
2. **O cálculo do loss seja feito sobre múltiplas passadas do mesmo input pela rede** (multi-trial averaging), reduzindo ruído estocástico da SNN
3. O pipeline original continue funcional

---

## 🎯 Objetivo técnico

Adicionar argumento CLI:

```
--loss_mode=<modo>
--num_passes=<int>
```

Onde:

* `loss_mode` controla a estratégia de readout + loss
* `num_passes` define **quantas vezes o mesmo batch deve ser passado pela SNN**
* O loss final deve ser calculado **sobre a média agregada das múltiplas execuções**

---

## 📌 Regra obrigatória — Multi-Pass Aggregation

Durante o treino, para cada batch:

```python
outputs = []
for _ in range(num_passes):
    spk_out_seq, mem_trace = model(spk_in, None)
    outputs.append((spk_out_seq, mem_trace))

# Agregar estatísticas entre múltiplas execuções
spk_out_seq = mean(outputs.spikes)
mem_trace   = mean(outputs.membrane)
```

**O loss SEMPRE deve ser calculado após essa agregação**, e **não por execução individual**.

Objetivo: **reduzir variância induzida por spikes, refratariedade e Poisson noise**.

---

## 📌 Função central obrigatória

Criar:

```python
def compute_loss(loss_mode, spk_out_seq, mem_trace, target, cfg_snn):
    ...
```

---

## 📌 Modos obrigatórios (TODOS devem ser implementados)

---

### 1️⃣ `rate` — Rate coding clássico (modo atual)

```python
contagem = spk_out_seq.sum(dim=0)
loss = F.cross_entropy(contagem, target)
```

---

### 2️⃣ `monte_carlo` — Monte Carlo sampling (nativo)

Já embutido no **multi-pass forward**:

```python
contagem = spk_out_seq.sum(dim=0)
loss = F.cross_entropy(contagem, target)
```

---

### 3️⃣ `temporal_pooling` — Pooling por janelas temporais

```python
pooled = spk_out_seq.view(num_windows, window_size, B, C).mean(dim=1)
readout = pooled.mean(dim=0)
loss = F.cross_entropy(readout, target)
```

---

### 4️⃣ `van_rossum` — PSC / Van Rossum kernel

```python
filtered = apply_psc_kernel(spk_out_seq, tau=cfg_snn.tau_psc)
readout = filtered.sum(dim=0)
loss = F.mse_loss(readout, target_vector)
```

---

### 5️⃣ `membrane` — Potencial de membrana como saída

```python
readout = mem_trace.sum(dim=0)
loss = F.mse_loss(readout, target_vector)
```

---

### 6️⃣ `cosine` — Cosine similarity sobre vetor médio contínuo

```python
readout = mem_trace.mean(dim=0)
loss = 1 - F.cosine_similarity(readout, target_vector).mean()
```

---

### 7️⃣ `mse_vector` — MSE sobre vetor médio contínuo

```python
readout = mem_trace.mean(dim=0)
loss = F.mse_loss(readout, target_vector)
```

---

## 📌 Atualizações necessárias no código real

### Em `treinar_classificador_locutor`

Substituir forward único por **multi-forward**:

```python
spk_runs = []
mem_runs = []

for _ in range(cfg_snn.num_passes):
    spk_seq, mem = model(spk_in, None)
    spk_runs.append(spk_seq)
    mem_runs.append(mem)

spk_out_seq = torch.stack(spk_runs).mean(dim=0)
mem_trace   = torch.stack(mem_runs).mean(dim=0)

loss = compute_loss(loss_mode, spk_out_seq, mem_trace, yb, cfg_snn)
```

---

## 📌 Compatibilidade obrigatória

### ✅ Não quebrar inferência

Inferência pode continuar usando **uma passada**, sem custo extra.

### ✅ Default seguro

```
loss_mode = "rate"
num_passes = 1
```

### ✅ GPU-safe

Nada deve mover tensores para CPU.

---

## 📌 Helpers esperados

```python
aggregate_spikes()
aggregate_membrane()
apply_psc_kernel()
multi_forward_pass()
```

---

## 📌 Log desejado

```python
print(f"[INFO] Loss mode = {loss_mode} | num_passes = {num_passes}")
```

---

## 📦 Saída esperada do Copilot

1. Código modificado completo
2. Implementação de `compute_loss`
3. Multi-pass forward integrado
4. CLI parsing atualizado
5. Explicação curta do patch

