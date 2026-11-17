# Revisão integral da tese pelo Antigravity (Gemini)

Instruções para repetir, com o agente do Google Antigravity, a **leitura integral do PDF** feita
em 2026-11-17 (registrada em `issues.md`, seção "Leitura integral do PDF"). O objetivo é uma
**segunda opinião independente**: outro modelo lendo a mesma tese com o mesmo método, para achar
o que a primeira leitura deixou passar.

## 1. Preparação (feita por você, antes de abrir o agente)

1. Compile a versão a revisar e anote o commit:
   ```bash
   cd documentation/00-thesis/monography
   latexmk -pdf -interaction=nonstopmode monografia.tex
   git rev-parse --short HEAD
   ```
2. No Antigravity, abra **a pasta raiz do repositório** (`doutorado/`) como workspace, para que o
   agente possa conferir o texto contra o código em `software/nn/`.
3. Escolha o modelo Gemini mais capaz disponível e o modo de planejamento (*Planning*), não o
   modo rápido: a tarefa é longa e exige verificação.
4. **Não deixe o agente ler `issues.md` antes de terminar** (§ 4). Se ler, ele ancora na lista
   que já existe e deixa de ser uma revisão independente.

## 2. Prompt para colar no agente

> Você vai revisar a tese de doutorado em
> `documentation/00-thesis/monography/monografia.pdf` (fonte LaTeX em `chapters/*.tex`,
> bibliografia em `bibliography.bib`). Leia as instruções completas em
> `documentation/00-thesis/monography/review_antigravity.md` e siga-as à risca. **Não leia
> `issues.md` até a etapa final indicada nelas.** Não altere nenhum arquivo além do relatório
> que essas instruções pedem.

## 3. Instruções para o agente

### 3.1 Regras

- **Não edite** a tese, o código nem os resultados. A única saída é o relatório do § 3.5.
- **Não rode experimentos.** Nada de `thesis`, `run_thesis_profiles.sh`, `sbatch` ou qualquer
  treino; eles levam horas. Ler arquivos, `pdftotext`, `grep` e compilar o LaTeX é permitido.
- **Não faça commit nem push.**
- **Toda afirmação de defeito tem que ter evidência**: a página impressa do PDF e, quando o
  problema é "o texto diz X mas o método faz Y", o arquivo de código que mostra Y. Se não
  conseguir confirmar, marque o item como *suspeita* e diga o que faltou conferir. Não invente
  referências, números nem linhas de código.
- Os números de resultado do texto vêm de chaves geradas (`\tn{chave}`, definidas em
  `tables/thesis_numbers.tex`). Um número digitado à mão no meio do texto é por si só um
  achado (exceção declarada: o estudo preliminar citado no Apêndice A).
- Referências bibliográficas: confira autor, título, veículo, ano, volume e páginas pela DOI ou
  pela página do editor. Sem fonte verificável, marque *não verificada*; não "corrija" de memória.
- Escreva o relatório em **português**.

### 3.2 Como ler

1. Extraia o texto com a numeração de páginas:
   ```bash
   cd documentation/00-thesis/monography
   pdftotext -layout monografia.pdf /tmp/tese.txt
   ```
   Leia **todas** as páginas, em ordem, em blocos de ~20 páginas. Use a numeração impressa na
   página (não a do leitor de PDF) ao citar.
2. Para figuras, tabelas e equações, olhe a página renderizada (por exemplo,
   `pdftoppm -f N -l N -r 80 -png monografia.pdf /tmp/p`). Texto dentro de imagens não aparece
   no `pdftotext`.
3. Para localizar o fonte de um trecho, procure-o em `chapters/*.tex`.

### 3.3 O que conferir

**Contra o código (o texto descreve o método que rodou?).** Os pontos principais e onde
conferir:

| Assunto do texto | Onde está a verdade |
|---|---|
| extrator manual (DTWPT, escalas, descritores, categorias 1/2) | `software/nn/src/experiments/thesis/lib/src/ThesisHandcraftedFeatures.cpp`, `software/nn/scripts/pipeline/thesis/00_thesis_generate_profiles.py` |
| autoencoders e PCA (entrada, tamanhos) | `ThesisFeatureExtraction.cpp`, `ThesisPca.cpp` (mesma pasta) |
| critério paraconsistente e escolha do vencedor | `ThesisParaconsistent.cpp`, `scripts/pipeline/thesis/01_thesis_phase00_rank.py` |
| classificadores (DSNN, gêmea ANN, cosseno), padronização, validação cruzada, EER | `ThesisDsnnClassifier.hpp`, `ThesisAnnClassifier.hpp`, `ThesisClassifiers.cpp`, `ThesisClassifiersInternal.hpp` |
| bases (Coretto, BIOMEX-DB), filtros, exclusões | `ThesisDataset.cpp`, `EegPairing.cpp`, `software/nn/scripts/data/biomex_to_sqlite.py` |
| disfonia simulada (sussurro LPC, perfis WORLD) | `VoiceDegradation.cpp`, `VoiceDysphonia.cpp` |
| testes estatísticos e regras pré-registradas | docstrings de `scripts/pipeline/thesis/0[3-9]_*` e `thesis_cross_profile_significance.py` |
| perfis e contagens | `software/nn/src/experiments/thesis/profiles/profiles.manifest` |

A documentação técnica do experimento está em `software/nn/.wiki/Experiments/Thesis.md`.

**Contra os dados (a afirmação é sustentada?).** Por exemplo: não significância lida como
igualdade; "mede" onde só se estima; generalização além do que foi testado; hipótese escrita
como resultado; contagens e números que não batem entre texto, tabela e figura.

**Forma.** Ortografia e concordância; termos inconsistentes entre capítulos; siglas sem
definição; vírgula decimal (o texto é em português); legendas e fontes das figuras; remissões
quebradas ("??") ou que apontam para a seção errada; aspas LaTeX; voz impessoal; bibliografia
(duplicatas, caixa de siglas nos títulos, campos faltando).

### 3.4 Classificação dos achados

| Grupo | Significado |
|---|---|
| **A** | O texto descreve um método diferente do que foi executado (uma banca pode exigir correção ou reexecução). |
| **B** | Afirmação mais forte do que os dados sustentam, inconsistência entre partes, ou explicação que falta. |
| **C** | Forma: ortografia, termos, figuras, tabelas, bibliografia. |

### 3.5 Relatório

Grave em `documentation/00-thesis/monography/review_antigravity_report.md`:

1. Cabeçalho: data, commit revisado, modelo usado e número de páginas lidas.
2. Uma tabela por grupo, com estas colunas: **#** (A1, A2, …, B1, …, C1, …),
   **Onde** (página impressa e seção), **Problema** (o que o texto diz e o que deveria dizer),
   **Evidência** (arquivo e linha, ou cálculo) e **Status** (*confirmado* ou *suspeita*).
3. Uma lista curta das páginas ou seções que você **não** conseguiu verificar, com o motivo.

### 3.6 Etapa final: comparar com a primeira leitura

Só depois de gravar o relatório, leia `issues.md` (seções "Leitura integral do PDF" e as
seguintes) e acrescente ao relatório uma seção **"Comparação"** com:

- achados novos (não estão em `issues.md`);
- achados que já constavam e que o texto atual **ainda** não corrige;
- itens de `issues.md` dados como corrigidos que você considera **não** corrigidos, com a
  página.

As marcações `[PENDENTE: …]` no PDF já são conhecidas (itens que só o autor resolve); não as
liste como achados.

## 4. Depois da revisão (você)

1. Leia `review_antigravity_report.md` e descarte o que não se confirmar.
2. Leve o que restar para o Claude Code, que confere cada item no código e aplica as correções
   por grupo, um commit por grupo, como na primeira leitura.
