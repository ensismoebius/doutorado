# Experiment 05 — Autenticação Biométrica de Locutores Drasticamente Disfônicos Aprimorada pela *Imagined Speech*

**Tese de doutorado — experimento principal.**

- **Tese:** Autenticação Biométrica de Locutores Drasticamente Disfônicos Aprimorada pela *Imagined Speech*
- **Autor:** André Furlan — UNESP
- **Orientador:** Prof. Dr. Eng. Rodrigo Capobianco Guido
- **Financiamento:** FAPESP 2021/12407-4

---

## Objetivo

Projetar e implementar algoritmos biométricos capazes de autenticar, em princípio por meio da
fala, indivíduos que sejam capazes de produzir somente locuções severamente degradadas
(disfonias laríngeas severas — DLSs), adicionando informações extraídas dos sinais cerebrais
durante a fonação, isto é, da *imagined speech*, ao conjunto daquelas que são acústicas e
oriundas da voz prejudicada.

Esta implementação cobre os experimentos **E3** (extração de características) e **E4**
(autenticação) do projeto de pesquisa, operando sobre a base de dados pública
`10.1117/12.2255697`.

---

## O que o experimento implementa

### E3 — Extração de Características (EC)

As características dos sinais de voz e de EEG são extraídas de duas maneiras:

**Método manual (*handcrafted extraction*)**

Aplica a Transformada *Wavelet-Packet* de Tempo Discreto (DTWPT) com filtro Daubechies-4 em
`handcrafted.dtwpt_level` níveis. Em cada sub-banda são calculados os descritores selecionados
em `handcrafted.descriptors`:

| Descritor | Fórmula |
|---|---|
| Energia (`energy`) | $E_k = \sum_n c_k[n]^2$ |
| Taxa de cruzamentos por zero (`zcr`) | $\frac{1}{N}\sum_n \mathbb{1}[\text{sinal}(x[n]) \neq \text{sinal}(x[n-1])]$ |
| Entropia (`entropy`) | $-\sum_k p_k \log_2 p_k$ (coeficientes normalizados) |
| Teager–Kaiser (`teager`) | $\Psi[x(n)] = x^2(n) - x(n+1)\cdot x(n-1)$ |
| Jitter (`jitter`) | $\frac{\overline{|T_i - T_{i+1}|}}{\bar{T}}$ (período a período, baseado em picos) |
| Shimmer (`shimmer`) | $\frac{\overline{|A_i - A_{i+1}|}}{\bar{A}}$ (amplitude a amplitude, baseado em picos) |

O campo `handcrafted.scale` é armazenado como sufixo no rótulo do conjunto de características
(`"handcrafted-<scale>"`). **Ele não altera o algoritmo de extração** — serve para identificar
a rodada no CSV/JSON/DAT de saída. As escalas previstas no manuscrito (Bark, MEL, LFCC) são
comparadas rodando o experimento com diferentes perfis.

O sinal processado por amostra é determinado por `dataset.modality`:
- `"voice"` → tensor de áudio da amostra (`sample.audio`)
- `"eeg"` → tensor de EEG da amostra (`sample.eeg`)
- `"fused"` → tensor de áudio se disponível, caso contrário tensor de EEG

O sinal é completado com zeros até a próxima potência de dois antes da DTWPT. A extração é
paralelizada por OpenMP (`schedule(dynamic,4)`) sobre as amostras.

**Método automatizado (*feature learning*)**

Previstos autoencoders LSTM-AE e SNN-AE para extração de VsCs latentes. **Ainda não
implementado neste binário.** O perfil retorna um `FeatureSet` vazio (marcador), que é
ignorado pela etapa de paraconsistência e pela classificação.

### EPC — Engenharia Paraconsistente de Características

Se `paraconsistent.enabled = true`, todos os conjuntos de características não-vazios são
avaliados pela lógica paraconsistente antes de qualquer classificador ser treinado:

1. Agrupamento das amostras por `subject_id` → mapa de classes.
2. Truncamento para tamanho uniforme (mínimo entre classes).
3. Normalização dos vetores de características em [0, 1] por componente.
4. Cálculo de **α** (similaridade intraclasse) e **β** (sobreposição interclasse) via
   `include/paraconsistent/`.
5. Mapeamento no plano paraconsistente: G₁ = α − β, G₂ = α + β − 1.
6. **D_verdade** = √((G₁ − 1)² + G₂²) — distância ao vértice "Verdade" (1, 0).

D_verdade menor indica maior separabilidade natural dos VsCs. O ranking é escrito no CSV de
saída e impresso no stdout. Todos os conjuntos não-vazios são passados à classificação
independentemente do ranking.

### E4 — Autenticação

**Arquitetura do classificador** (`cfg.classifier.type = "rnn"` — Residual Neural Network):

```
Entrada (feat_dim)
  → Linear(feat_dim, 128) → ReLU
  → BlocoResidual(128) × 2   [Linear→ReLU→Linear + ligação direta]
  → Linear(128, n_locutores)
```

Implementado como `SimpleResNetImpl<nn::Backend>` com `hidden_dim=128`, `depth=2`.
Perda: entropia cruzada (`CrossEntropyLossImpl`). Otimizador: Adam via `Trainer`.

> **DSNN (Deep Spiking Neural Network)** está previsto no manuscrito como segunda opção
> de classificador (`cfg.classifier.type = "dsnn"`). **Ainda não implementado.**

**Validação cruzada**

`GroupKFoldPolicy` com `k_folds` dobras garante que todas as locuções de um mesmo locutor
fiquem na mesma dobra, evitando vazamento de dados entre treino e teste. A dobra interna
(primeira divisão interna da dobra externa) fornece o conjunto de validação para a parada
antecipada.

Divisão de texto dentro de cada dobra:
- `"dependent"` — 80 % das locuções para treino, 20 % para teste (embaralhado).
- `"independent"` — frases divididas ~50/50 por identidade da frase; treino e teste usam
  conjuntos de frases disjuntos.

**Métricas avaliadas por dobra** (avaliação em lote, um único `forward`):

| Métrica | Implementada | Fórmula |
|---|---|---|
| Acurácia | ✓ | Acertos argmax / N |
| F1 macro | ✓ | Média não-ponderada de 2PR/(P+R) por classe |
| Precisão macro | ✓ | Média de TP/(TP+FP) por classe |
| Recall / Sensibilidade | ✓ | Média de TP/(TP+FN) por classe |
| EER | ✓ | Cruzamento FAR=FRR (`GenuineImpostorEERScorer`: similaridade cosseno genuíno/impostor) |
| AUC-ROC | ✓ | P(score_genuíno > score_impostor) — estimador Wilcoxon–Mann–Whitney |
| MSE | — | Usado no treino de autoencoders (E3 automático, não implementado ainda) |
| Especificidade | — | Planejada |

Agregação sobre dobras externas: média, desvio padrão populacional, IC95 % (1,96 × dp / √n)
para acurácia e EER; média e dp para F1, precisão, recall e AUC.

**Protocolo de EER** (`GenuineImpostorEERScorer`):
1. Por locutor: primeira locução → template normalizado L2.
2. Locuções restantes → sondas (*probes*).
3. Cada sonda pontuada contra todos os templates por similaridade cosseno.
4. Distribuição de pontuações genuínas vs. impostoras → varredura de limiar → cruzamento FAR/FRR = EER.

**Salvamento de modelos**: após cada dobra externa, o `state_dict` do modelo treinado é
gravado em `<results_dir>/models/<run_tag>/<feature_label>/fold_<N>.bin` via
`nn::io::save_state_dict`.

---

## Base de dados utilizada

**`10.1117/12.2255697`** (validação pública):

- 15 locutores falantes de espanhol
- Estímulos: vogais (/a/ /e/ /i/ /o/ /u/) e comandos direcionais (arriba / abajo / izquierda / derecha / adelante)
- Três modalidades: fala fonada, fala imaginada (EEG), mista
- Áudio: 22 050 Hz, 16 bits PCM WAV
- EEG: 800 Hz, 14 canais, sistema 10-20 (Emotiv EPOC)
- Pré-processamento EEG: passa-banda 1–800 Hz, notch 60 Hz

> **Base de dados própria** (locutores com DLS, protocolo de coleta do Capítulo 3 da monografia)
> **não está disponível ainda**. Este experimento opera exclusivamente sobre a base pública.

### Carregamento dos dados

`load_dataset()` sempre carrega **ambos** os sinais (áudio e EEG) para cada ensaio, usando o
campo `eeg_index` da sessão de áudio (índice 1-base) para identificar a linha EEG
correspondente. Locutores sem ambos os arquivos MAT e ensaios com índice EEG fora do intervalo
são descartados silenciosamente. O campo `dataset.modality` não afeta o carregamento — apenas
a extração de características.

Isso garante que rodadas voz-somente, EEG-somente e fusão operem sobre o **conjunto idêntico**
de locutores e ensaios, tornando os resultados diretamente comparáveis.

---

## Perfil JSON

```jsonc
{
  "experiment": {
    "run_tag": "e05_handcrafted_voice",  // prefixo dos arquivos de saída
    "seed": 42,
    "repeats": 1,
    "seed_deterministic": false
  },
  "dataset": {
    "root": "/caminho/para/10.1117/",
    "results_dir": "results/",
    "modality": "voice",        // "voice" | "eeg" | "fused"
                                // determina qual tensor é passado para a extração;
                                // áudio e EEG são SEMPRE carregados e pareados
    "max_samples": 0            // 0 = ilimitado; valor pequeno (ex.: 20) para depuração
  },
  "feature_extraction": {
    "strategy": "handcrafted",  // "handcrafted" | "autoencoder" (autoencoder = marcador, não treinado)
    "handcrafted": {
      "transform": "dtwpt",     // único valor implementado
      "scale": "bark",          // "bark" | "mel" | "lfcc" — apenas rótulo, não altera o algoritmo
      "descriptors": ["energy", "zcr", "entropy", "teager"]
                                // opcionais: "jitter", "shimmer"
    },
    "autoencoder": {
      "model": "lstm-ae",       // "lstm-ae" | "snn-ae" — não treinado neste binário
      "encoder_layer_spec": [],
      "decoder_layer_spec": []
    }
  },
  "paraconsistent": {
    "enabled": true             // calcula D_verdade antes da classificação
  },
  "classifier": {
    "type": "rnn",              // "rnn" implementado; "dsnn" planejado
    "layer_spec": [],           // não parseado; arquitetura fixada: Linear→ReLU→2×Residual→Linear
    "text_mode": "independent"  // "dependent" | "independent"
  },
  "training": {
    "epochs": 50,
    "learning_rate": 1e-3,
    "samples_per_batch": 32,
    "early_stop_patience": 10,  // 0 = desativado
    "k_folds": 5,
    "nested_cv": true           // não parseado; validação cruzada aninhada sempre usada
  }
}
```

---

## Pipeline completo

```
Base pública 10.1117/12.2255697
  │
  ├── Fala fonada (22 050 Hz WAV)       ← sempre carregada
  └── EEG imagined speech (800 Hz, 14 ch) ← sempre carregada e pareada por eeg_index
          │
          ▼   dataset.modality → seleciona qual sinal vai para a EC
  ┌────────────────────────────────────────┐
  │  E3 — Extração de Características (EC) │
  │  DTWPT + descritores (handcrafted)      │
  │  OU autoencoder latente (planejado)     │
  └────────────────────┬───────────────────┘
                       │  VsCs por amostra
          ▼
  ┌────────────────────────────────────────┐
  │  EPC — Engenharia Paraconsistente      │
  │  α (intraclasse) + β (interclasse)     │
  │  → D_verdade ranks os conjuntos de VsC │
  └────────────────────┬───────────────────┘
                       │  ranking impresso + salvo em CSV
          ▼
  ┌────────────────────────────────────────┐
  │  E4 — Autenticação                     │
  │  Classificador RNN (ResNet) ou DSNN*   │
  │  Validação cruzada aninhada k=5        │
  │  Modos: text-dependent / independent   │
  └────────────────────┬───────────────────┘
                       │
          Saída: locutor autêntico ou inautêntico
          Métricas: acurácia, F1, precisão, recall,
                    EER, AUC-ROC por dobra + agregados

* DSNN planejado; apenas RNN implementado atualmente.
```

---

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `results/e05_<tag>_metrics.csv` | Por dobra: `feature_set, classifier, text_mode, fold, accuracy, f1, precision, recall, eer, auc, model_path` |
| `results/e05_<tag>_paraconsistent.csv` | Por conjunto: `label, alpha, beta, g1, g2, d_truth` |
| `results/e05_<tag>_summary.json` | Configuração + média/dp/IC95 de todas as métricas + caminhos dos modelos por dobra |
| `results/e05_<tag>_comparison.dat` | DAT pgfplots: `x label accuracy std_accuracy ci95_accuracy f1 std_f1 precision recall eer std_eer ci95_eer auc std_auc` |
| `results/models/<tag>/<feature_label>/fold_<N>.bin` | State dict binário do modelo por dobra externa |

---

## Layout do diretório

```
05/
├── README.md                             ← este arquivo
├── CMakeLists.txt
├── experiment05.cpp                      ← ponto de entrada CLI (pipeline de 6 etapas)
├── lib/
│   ├── include/
│   │   ├── E05Config.hpp                 ← parser JSON + validate()
│   │   ├── E05Dataset.hpp                ← E05Sample, E05DatasetView, load_dataset()
│   │   ├── E05FeatureExtraction.hpp      ← extract_features(), extract_handcrafted()
│   │   ├── E05Paraconsistent.hpp         ← score_feature_set(), rank_feature_sets()
│   │   ├── E05Classifiers.hpp            ← FoldResult, ClassificationResult, run_classifier()
│   │   └── E05Output.hpp                 ← escritores CSV, JSON, DAT
│   └── src/
│       ├── E05Config.cpp
│       ├── E05Dataset.cpp                ← carregamento pareado via eeg_index
│       ├── E05FeatureExtraction.cpp      ← DTWPT + loop OpenMP paralelo
│       ├── E05Paraconsistent.cpp         ← cálculo α/β/D_verdade
│       ├── E05Classifiers.cpp            ← SimpleResNet + GroupKFold + EER/AUC
│       └── E05Output.cpp
├── profiles/
│   ├── debug.json                        ← teste rápido (2 épocas, poucas amostras)
│   ├── handcrafted-voice.json            ← DTWPT na voz (scale=bark)
│   ├── handcrafted-eeg.json              ← DTWPT no EEG
│   ├── handcrafted-fused.json            ← DTWPT na voz+EEG
│   ├── autoencoder-voice.json            ← marcador (autoencoder não treinado)
│   ├── autoencoder-eeg.json
│   ├── autoencoder-fused.json
│   └── article-full.json                 ← rodada completa de comparação
└── tests/
    ├── e05_profile_audit_gtest.cpp       ← 48 testes: 8 perfis parseiam e validam
    ├── e05_feature_extraction_gtest.cpp  ← funções de descritor + extract_handcrafted
    └── e05_classifiers_gtest.cpp         ← compute_aggregate_stats + run_classifier (sintético)
```

---

## Build e execução

```bash
# Configurar (uma vez)
cmake --preset=max-performance

# Compilar
cmake --build out/build/max-performance --target experiment05 -j$(nproc)

# Rodar perfil único
./out/build/max-performance/src/experiments/05/experiment05 \
  --config src/experiments/05/profiles/handcrafted-voice.json

# Rodar todos os testes
cmake --build out/build/max-performance --target e05_profile_audit_gtest -j$(nproc)
ctest --test-dir out/build/max-performance -R e05 --output-on-failure
```

---

## Recarregar modelo salvo

```cpp
#include "io/StateIO.hpp"
#include "layers/residual/SimpleResNet.hpp"

SimpleResNetImpl<nn::Backend> model(feat_dim, /*hidden=*/128, n_locutores, /*depth=*/2);
auto sd = nn::io::load_state_dict("results/models/<run_tag>/<feature_label>/fold_0.bin");
model.load_state_dict(sd);
```

---

## Diferenças em relação ao Experimento 04

| Aspecto | Experimento 04 | Experimento 05 |
|---|---|---|
| Objetivo | Artigo de congresso (reconstrução SNN vs LSTM) | Experimento principal da tese |
| Base de dados | FSDD (dígitos, 8 kHz, áudio) | 10.1117/12.2255697 (EEG + voz, 22 050 Hz) |
| Tarefa | Reconstrução autoencoder (MSE) | Autenticação de locutores (acurácia, EER, AUC) |
| Sinais | Somente áudio | Voz + EEG, sempre carregados como ensaios pareados |
| Seleção de características | Varredura de arquitetura fixa | Ranking paraconsistente EPC (α/β) antes de qualquer classificador |
| Classificador | Perda de reconstrução | ResNet densa (SimpleResNet) + entropia cruzada |
| Modalidades de texto | N/A | Text-dependent + text-independent |
| Estratégia de CV | KFold estratificado | GroupKFold (agrupado por locutor, sem vazamento) |
| Modelos salvos | Não | State dict binário por dobra |

---

## Veja também

- [Wiki Experimento 05](./../../../.wiki/Experiments/Experiment05.md) — teoria completa, fórmulas, armadilhas
- [Contexto de Pesquisa](./../../../.wiki/Research-Context.md) — objetivos da tese e pipeline
- [Core/Paraconsistente](./../../../.wiki/Core/Paraconsistent.md) — derivação de α/β/D_verdade
- [Conceitos/Imagined-Speech-and-EEG](./../../../.wiki/Concepts/Imagined-Speech-and-EEG.md)
- [Conceitos/LFCC](./../../../.wiki/Concepts/LFCC.md)
