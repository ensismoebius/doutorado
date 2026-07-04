# Experiment 05 — Autenticação Biométrica de Locutores Drasticamente Disfônicos Aprimorada pela *Imagined Speech*

**Tese de doutorado — experimento principal.**

Auditoria de aderencia implementacao vs documentos: [GAP_ANALYSIS.md](./GAP_ANALYSIS.md).

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

## Passo a passo para concluir a tese

Use esta sequencia como plano de execucao. Cada etapa tem entregavel e criterio de pronto.

### 1) Congelar escopo cientifico

- [ ] Definir pergunta final, hipoteses H1/H2/H3 e contribuicoes declaradas da tese.
- [x] Escopo atual de classificadores executaveis: `RNN + DSNN`.
- [ ] Decidir escopo final de E3 automatico: implementar `LSTM-AE/SNN-AE` ou registrar como trabalho futuro.

**Entregavel:** secao de escopo final no capitulo de metodologia + lista de contribuicoes.

**Pronto quando:** nao existir item "talvez" no escopo experimental.

### 2) Fechar base de dados e protocolo

- [ ] Consolidar inventario da base publica usada (`10.1117/12.2255697`): sujeitos, modalidades, filtros, exclusoes.
- [ ] Registrar claramente no texto se base propria com DLS entrou ou nao nos resultados desta versao.
- [ ] Congelar regras de split: `text-dependent` e `text-independent`.

**Entregavel:** tabela de dados no capitulo de resultados + subsecao de protocolo replicavel.

**Pronto quando:** qualquer leitor consegue reproduzir o mesmo conjunto de amostras.

### 3) Fechar implementacao do Experimento 05

- [ ] Garantir que todos os perfis JSON usados no artigo/tese parseiam e validam.
- [ ] Se escopo incluir E3 automatico: implementar treino e extracao latente real (hoje ainda marcador).
- [x] Caminho `cfg.classifier.type = "dsnn"` implementado e coberto por teste.
- [ ] Manter testes de regressao para feature extraction, classifier e agregacao de metricas.

**Entregavel:** codigo compilando + testes `e05_*` passando.

**Pronto quando:** `ctest -R e05 --output-on-failure` sem falhas.

### 4) Rodar matriz experimental completa

- [ ] Definir matriz minima: `voice`, `eeg`, `fused` x `dependent`, `independent` x estrategias de features escolhidas.
- [ ] Rodar com `k_folds=5`, sementes fixas e `repeats` definido no escopo.
- [ ] Salvar todos os artefatos em `results/` (CSV, JSON, DAT, modelos por dobra).

**Entregavel:** pacote de resultados finais por `run_tag`.

**Pronto quando:** nenhuma combinacao da matriz faltar.

### 5) Fazer auditoria de qualidade dos resultados

- [ ] Verificar consistencia entre `metrics.csv`, `summary.json` e `comparison.dat`.
- [ ] Confirmar medias, desvios e IC95 para acuracia e EER.
- [ ] Revisar sinais de vazamento de dados (split por locutor, frases disjuntas no modo independente).

**Entregavel:** checklist de validacao assinado no repositorio (arquivo markdown em `results/`).

**Pronto quando:** sem inconsistencias numericas entre arquivos.

### 6) Gerar figuras e tabelas finais da tese

- [ ] Tabela principal: comparacao por modalidade (voice/eeg/fused) e por modo de texto.
- [ ] Figura principal: barras de `accuracy`, `F1`, `EER`, `AUC` com erro (dp ou IC95).
- [ ] Tabela de ranking EPC: `alpha`, `beta`, `g1`, `g2`, `d_truth`.

**Entregavel:** figuras e tabelas prontas em `documentation/00-thesis/monography/`.

**Pronto quando:** todas as figuras citadas no texto estao geradas e versionadas.

### 7) Escrever capitulo 9 (Testes e Resultados)

- [ ] Descrever protocolo experimental completo (dados, preprocessamento, modelos, metricas).
- [ ] Inserir resultados por dobra e agregados, com interpretacao critica.
- [ ] Incluir comparacao com estado da arte e limites do estudo.

**Entregavel:** `chapters/09-testsAndResults.tex` completo.

**Pronto quando:** capitulo compila sem TODO/FIXME e sem referencias quebradas.

### 8) Escrever capitulo 10 (Conclusoes)

- [ ] Responder explicitamente cada objetivo especifico da tese.
- [ ] Declarar contribuicoes tecnicas e cientificas validadas por dados.
- [ ] Listar ameacas a validade e trabalhos futuros concretos.

**Entregavel:** `chapters/10-conclusions.tex` completo.

**Pronto quando:** conclusoes batem 1:1 com resultados apresentados.

### 9) Fechar bibliografia e citacoes

- [ ] Revisar `bibliography.bib` para DOI, ano, venue e duplicatas.
- [ ] Garantir que toda afirmacao tecnica relevante tenha citacao.
- [ ] Remover referencias nao usadas.

**Entregavel:** bibliografia final limpa.

**Pronto quando:** build LaTeX sem avisos graves de citacao.

### 10) Atualizar arquivo principal da monografia

- [ ] Descomentar inclusoes finais em `monografia.tex`:
  - `\include{chapters/09-testsAndResults}`
  - `\include{chapters/10-conclusions}`
- [ ] Validar sumario, lista de figuras/tabelas e apendices.

**Entregavel:** PDF final completo da tese.

**Pronto quando:** `monografia.pdf` compila do inicio ao fim sem erro.

### 11) Preparar pacote de reproducibilidade

- [ ] Incluir comandos de build/run exatos e perfis usados.
- [ ] Congelar seeds, versoes e hash de commit da rodada final.
- [ ] Organizar resultados finais por pasta de submissao/defesa.

**Entregavel:** pacote replicavel para banca e publicacoes.

**Pronto quando:** terceiro reproduz resultados principais com instrucoes do repositorio.

### 12) Preparar defesa e submissao

- [ ] Gerar narrativa de defesa: problema, metodo, resultados, contribuicoes.
- [ ] Criar 3 blocos de evidencias: tecnica, estatistica, impacto pratico.
- [ ] Preparar versao curta (artigo) e versao longa (tese).

**Entregavel:** slide deck + roteiro + tese final.

**Pronto quando:** historia cientifica fecha sem lacunas entre objetivo, metodo e resultado.

---

## Ordem minima recomendada (execucao rapida)

1. Fechar escopo (Passos 1-3).
2. Rodar matriz final e auditoria (Passos 4-5).
3. Gerar tabelas/figuras (Passo 6).
4. Escrever capitulos 9 e 10 (Passos 7-8).
5. Fechar bibliografia + build final (Passos 9-10).
6. Empacotar reproducibilidade e defesa (Passos 11-12).

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
(`"handcrafted-<scale>"`). As escalas Bark/MEL/LFCC **alteram o agrupamento espectral das
sub-bandas DTWPT** antes do cálculo dos descritores, e também identificam a rodada no
CSV/JSON/DAT de saída.

O sinal processado por amostra é determinado por `dataset.modality`:
- `"voice"` → tensor de áudio da amostra (`sample.audio`), com pré-ênfase
- `"eeg"` → tensor de EEG da amostra (`sample.eeg`)
- `"fused"` → combina voz + EEG; *quando* combina depende de `dataset.fusion_mode`:
  - `"early"` → sinais brutos concatenados (voz seguida de EEG) e extraídos numa única passagem
  - `"late"` (padrão) → voz e EEG extraídos independentemente; os vetores de características resultantes são concatenados por amostra (`[voz ‖ eeg]`)

O sinal é completado com zeros até a próxima potência de dois antes da DTWPT. A extração é
paralelizada por OpenMP (`schedule(dynamic,4)`) sobre as amostras.

**Método automatizado (*feature learning*)**

Implementado com **LSTM-AE** (`feature_extraction.autoencoder.model = "lstm-ae"`).
Cada amostra é padronizada para comprimento fixo por zero-padding, o autoencoder é treinado
com MSE via `Trainer::fit_autoencoder`, e o vetor latente de `encode()` é usado como VsC.
`snn-ae` permanece fora de escopo neste binário.

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

> **DSNN (Deep Spiking Neural Network)** implementado como classificador alternativo
> (`cfg.classifier.type = "dsnn"`) no mesmo pipeline de validacao cruzada.

**Validação cruzada**

`GroupKFoldPolicy` com `k_folds` dobras garante que todas as locuções de um mesmo locutor
fiquem na mesma dobra, evitando vazamento de dados entre treino e teste. No modo `nested_cv`,
todas as dobras internas da dobra externa são percorridas; o modelo com melhor acurácia de
validação interna é selecionado para avaliação na dobra externa.

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
| Especificidade | ✓ | Média de TN/(TN+FP) por classe |
| EER | ✓ | Cruzamento FAR=FRR (`GenuineImpostorEERScorer`: similaridade cosseno genuíno/impostor) |
| AUC-ROC | ✓ | P(score_genuíno > score_impostor) — estimador Wilcoxon–Mann–Whitney |
| MSE | ✓ (somente E3 autoencoder) | Função de reconstrução no treino LSTM-AE |

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
- Estímulos: vogais (/a/ /e/ /i/ /o/ /u/) e 6 comandos direcionais (arriba / abajo / adelante / atras / derecha / izquierda)
- Duas condições de fala: fonada (áudio + EEG simultâneos) e imaginada (apenas EEG)
- Áudio: 44 100 Hz, canal único (presente apenas nos trials de fala fonada)
- EEG: 1024 Hz, 6 canais (F3, F4, C3, C4, P3, P4), sistema 10-20; trial de 4 s = 4096 amostras/canal
- Fonte: Pressel Coretto, Gareis, Rufiner, "Open access database of EEG signals recorded during imagined speech", Proc. SPIE 10160, 2017 (DOI 10.1117/12.2255697)

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
    "fusion_mode": "late",      // "early" | "late" — só usado quando modality=fused
    "max_samples": 0            // 0 = ilimitado; valor pequeno (ex.: 20) para depuração
  },
  "feature_extraction": {
    "strategy": "handcrafted",  // "handcrafted" | "autoencoder"
    "handcrafted": {
      "transform": "dtwpt",     // único valor implementado
      "scale": "bark",          // "bark" | "mel" | "lfcc" — altera agrupamento das sub-bandas
      "descriptors": ["energy", "zcr", "entropy", "teager"]
                                // opcionais: "jitter", "shimmer"
    },
    "autoencoder": {
      "model": "lstm-ae",       // único valor aceito neste binário
      "encoder_layer_spec": [],
      "decoder_layer_spec": []
    }
  },
  "paraconsistent": {
    "enabled": true             // calcula D_verdade antes da classificação
  },
  "classifier": {
    "type": "rnn",              // "rnn" | "dsnn" implementados
    "layer_spec": [],            // validado (primeiro linear, residual presente, último linear:N_speakers)
    "text_mode": "independent"  // "dependent" | "independent"
  },
  "training": {
    "epochs": 50,
    "learning_rate": 1e-3,
    "samples_per_batch": 32,
    "early_stop_patience": 10,  // 0 = desativado
    "k_folds": 5,
    "nested_cv": true           // parseado; usa nested CV quando true, flat K-fold quando false
  }
}
```

---

## Pipeline completo

```
Base pública 10.1117/12.2255697
  │
  ├── Fala fonada (44 100 Hz, canal único) ← carregada quando há áudio
  └── EEG imagined speech (1024 Hz, 6 ch)   ← sempre carregada e pareada por eeg_index
          │
          ▼   dataset.modality → seleciona qual sinal vai para a EC
  ┌────────────────────────────────────────┐
  │  E3 — Extração de Características (EC) │
  │  DTWPT + descritores (handcrafted)      │
  │  OU autoencoder latente (LSTM-AE)       │
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

* DSNN implementado junto do baseline RNN.
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
│   ├── debug.json                        ← teste rápido (RNN, 3 épocas, poucas amostras)
│   ├── phase00/                          ← FASE 00: construção do vetor + ranking paraconsistente
│   │   ├── p00_hc_<wavelet>_<scale>_<fonte>.json   varredura handcrafted
│   │   │     wavelet ∈ {haar, daub4, daub6, ..., daub46}  (23, ver Types.hpp)
│   │   │     scale   ∈ {bark, mel, lfcc};  fonte ∈ {voice, eeg}
│   │   │     23 × 3 × 2 = 138 perfis
│   │   └── p00_ae_<tam>_<fonte>.json               autoencoder LSTM compacto → 6 perfis
│   │         tam ∈ {tiny, small, base}  (latente = 8 / 16 / 32; oculto 2:1)
│   │       (classifier.enabled=false; para após o ranking. 144 perfis no total)
│   │       (fundido é construído DEPOIS, do vencedor de cada lado)
│   └── phase01/                          ← FASE 01: autenticação DSNN (só o MELHOR combo)
│       └── p01_dsnn_<fonte>_<texto>_<cv>.json   (todos classifier=dsnn)
│           fonte ∈ {voice, eeg, fused-early, fused-late}
│           texto ∈ {dep, indep}, cv ∈ {nested, flat}
│           4 × 2 × 2 = 16 perfis
│           extrator = PLACEHOLDER (handcrafted/lfcc/daub4) — trocar pelo vencedor da Fase 00
└── tests/
    ├── e05_profile_audit_gtest.cpp       ← 1131 testes: 161 perfis (144 fase00 + 16 fase01 + debug) parseiam e validam
    ├── e05_feature_extraction_gtest.cpp  ← descritores + extract_handcrafted + fusão early/late + varredura de wavelets
    └── e05_classifiers_gtest.cpp         ← compute_aggregate_stats + run_classifier (sintético)
```

**Duas fases** (`classifier.enabled` controla o gate):
- **Fase 00** (`phase00/`): extrai características por sinal (voz, EEG) e roda o ranking paraconsistente para escolher o melhor extrator por sinal. Varre a **wavelet-mãe** (`handcrafted.wavelet`, 23 opções de `Types.hpp`) × escala × sinal (138 perfis), mais um **sweep de autoencoders LSTM compactos** (6 perfis). `classifier.enabled=false` → o pipeline para após o ranking; escreve apenas o CSV paraconsistente + JSON de resumo. Não precisa de `layer_spec`.

  **Autoencoders compactos (SOTA para dispositivos de baixo poder).** Três tamanhos por sinal, todos `lstm-ae` de uma camada (raso, adequado a borda), com razão de compressão 2:1 entre a camada oculta e o gargalo latente. O latente é o próprio vetor de características, então um gargalo menor gera vetor menor → classificador a jusante mais barato e melhor generalização:

  | tamanho | oculto | latente (= dim. do vetor) |
  |---|---|---|
  | `tiny`  | 16 | 8  |
  | `small` | 32 | 16 |
  | `base`  | 64 | 32 |

  Princípios (gargalo agressivo, razão ~2:1, pilha rasa) seguem a literatura de autoencoders leves para extração de características em borda. A escolha do modelo é limitada pelo código (apenas `lstm-ae`); variantes esparsas/quantizadas exigiriam extensão do `E05FeatureExtraction`.
- **Fase 01** (`phase01/`): alimenta **apenas o melhor combo** (escolhido pela engenharia paraconsistente na Fase 00) no classificador DSNN e mede EER/AUC. `classifier.enabled=true`, `paraconsistent.enabled=false`. O bloco `feature_extraction` é um placeholder — substitua pela wavelet/escala (ou `strategy=autoencoder`) vencedora antes de rodar. Cruza fonte × modo de texto × esquema de CV.

**Automação da transição Fase 00 → Fase 01** (dois scripts em `scripts/pipeline/`):

```bash
# 1. Rode todos os perfis phase00/ (grava results/phase00/*_paraconsistent.csv)

# 2. Ranqueie e escolha o vencedor por sinal (menor D_truth):
python3 scripts/pipeline/e05_phase00_rank.py \
    --profiles-dir src/experiments/05/profiles/phase00 \
    --results-dir  results/phase00 \
    --out          results/phase00/winners.json

# 3. Injete o vencedor nos 16 perfis phase01/ (fused usa o vencedor da voz por padrão):
python3 scripts/pipeline/e05_apply_winner.py \
    --winners      results/phase00/winners.json \
    --profiles-dir src/experiments/05/profiles/phase01 \
    --fused        voice      # ou: eeg

# 4. Rode os perfis phase01/ (já com o extrator vencedor).
```

O `rank` agrega os CSV por (wavelet × escala × sinal), faz média das repetições e reporta a tabela ordenada + o vencedor. O `apply` é idempotente: `voice→vencedor-voz`, `eeg→vencedor-eeg`, `fused-*→--fused`. Use `--dry-run` no `apply` para revisar sem gravar.

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
| Base de dados | FSDD (dígitos, 8 kHz, áudio) | 10.1117/12.2255697 (EEG 1024 Hz + voz 44 100 Hz) |
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
