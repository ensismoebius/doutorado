### Planejamento metodológico revisado pra protótipo de extração de características multimodais (EEG + Áudio)

#### 1. Aquisição de dados

* **Sinais utilizados**:

  * EEG multicanal
  * Áudio simultâneo
* **Sincronização temporal** obrigatória entre EEG e áudio.

---

#### 2. Padronização e redução inicial de dimensionalidade

**Segmentação temporal (windowing)**

* Tamanho da janela: **0,1 s (100 ms)**
* Sobreposição opcional: **50%** (avaliar empiricamente).

Justificativa técnica:

* Permite capturar **dinâmica temporal curta** relevante para EEG e fala.

**Downsampling**

| Sinal | Taxa original típica | Taxa utilizada |
| ----- | -------------------- | -------------- |
| Áudio | 44.1–48 kHz          | **16 kHz**     |
| EEG   | 500–1000 Hz          | **200 Hz**     |

Objetivos:

* Redução de dimensionalidade
* Preservação da banda informacional relevante

Fundamentos:

* Fala inteligível está majoritariamente abaixo de **8 kHz** (Teorema de Nyquist).
* EEG informativo está abaixo de **~100 Hz**.

---

#### 3. Estruturação das entradas

Para cada janela de **100 ms**:

**Áudio**

[
N_{audio} = 16000 \times 0.1 = 1600 \text{ amostras}
]

**EEG**

[
N_{eeg} = 200 \times 0.1 = 20 \text{ amostras por canal}
]

Se houver (C) canais EEG:

[
N_{total} = 1600 + (20 \times C)
]

Esses vetores são mantidos **sem extração manual de features** para o autoencoder.

---

#### 4. Pipeline de extração de características

O sistema terá **duas rotas paralelas de extração**:

### 4.1 Autoencoder

Entrada:

[
x \in \mathbb{R}^{N_{total}}
]

Objetivo:

[
z = f_{\theta}(x)
]

onde:

* (z) = vetor latente (características)
* (f_{\theta}) = encoder

Arquiteturas a testar:

1. **Autoencoder denso**
2. **Autoencoder convolucional temporal**
3. **Autoencoder variacional (VAE)**

Parâmetros experimentais:

| Parâmetro          | Valores candidatos |
| ------------------ | ------------------ |
| dimensão latente   | 16, 32, 64, 128    |
| número de camadas  | 2–6                |
| função de ativação | ReLU / GELU        |
| regularização      | dropout / L2       |

---

### 4.2 Extração por Wavelets

Aplicação separada para:

* EEG
* Áudio

Transformada:

[
W_x(a,b)
]

onde:

* (a) = escala
* (b) = posição temporal

Famílias de wavelet a testar:

| Família              |
| -------------------- |
| Daubechies (db4–db8) |
| Symlets              |
| Coiflets             |
| Morlet               |
| Mexican Hat          |

Características derivadas possíveis:

* energia por escala
* entropia wavelet
* variância por banda

---

#### 5. Engenharia Paraconsistente de Características

Objetivo:
avaliar **qual conjunto de características possui maior poder discriminativo**.

Base teórica:

Lógica Paraconsistente Anotada (LPA).

Cada vetor de características gera:

* **grau de evidência favorável ((\mu))**
* **grau de evidência contrária ((\lambda))**

A partir disso calcula-se:

**Grau de certeza**

[
Gc = \mu - \lambda
]

**Grau de contradição**

[
Gct = \mu + \lambda - 1
]

Interpretação:

| Região           | Significado                    |
| ---------------- | ------------------------------ |
| Alta certeza     | boa separação entre classes    |
| Alta contradição | características inconsistentes |

Métricas derivadas:

* **Distância paraconsistente**
* **Índice de separabilidade de classes**
* **Índice de inconsistência**

---

#### 6. Comparação dos métodos

Serão comparados:

1. Features do **autoencoder**
2. Features de **wavelets**
3. **Combinação dos dois**

Critério de avaliação:

* Engenharia paraconsistente de características
* Performance em classificador downstream (opcional)

Classificadores candidatos:

* SVM
* MLP
* SNN

---

#### 7. Seleção final do modelo

O modelo escolhido será aquele que apresentar:

1. **Maior separabilidade paraconsistente**
2. **Menor contradição**
3. **Melhor generalização**

---

# Evidência científica

### Uso de janelas temporais

Estudos de EEG e áudio utilizam **50–200 ms**.

Evidência: **A**

Referências:

* Cohen (2014) *Analyzing Neural Time Series Data*
* O'Shaughnessy (Speech Processing)

---

### Downsampling áudio para 16 kHz

Amplamente adotado em ASR.

Evidência: **A**

Referência:

* Rabiner & Schafer (2011) *Theory and Applications of DSP*

---

### Uso de autoencoders para EEG/áudio

Aplicado para aprendizado de representação.

Evidência: **B**

Exemplos:

* Waytowich et al., 2018 — Deep learning for EEG
* Hsu et al., 2017 — Unsupervised speech representation

---

### Engenharia Paraconsistente de Características

Aplicada em análise de padrões e sistemas inteligentes.

Evidência: **B**

Referências:

* Abe, J. M. (2015) — *Paraconsistent Intelligent Based Systems*
* Da Costa, Newton C. A. — lógica paraconsistente

---

# Limitações

1. **Janela fixa de 100 ms** pode não capturar eventos EEG mais longos.
2. **Autoencoders podem aprender ruído** se não houver regularização.
3. Engenharia paraconsistente **depende da definição correta de evidência**.
4. Wavelets ideais dependem fortemente do domínio espectral dos sinais.

---

# Possíveis extensões (fortemente recomendadas)

1. **Autoencoder multimodal com atenção**
2. **Contraste learning entre EEG e áudio**
3. **Autoencoder esparso**
4. **Representações latentes alinhadas (CCA ou Deep CCA)**

---

✔ **Nível geral de evidência:** **B**

✔ **Incerteza residual:** moderada, principalmente na escolha da arquitetura do autoencoder e da wavelet ótima.
