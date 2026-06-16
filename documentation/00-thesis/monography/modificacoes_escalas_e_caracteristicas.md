# Modificações — Escalas e Características

Registro de todas as alterações realizadas durante a revisão técnica (guide00.md).

---

## IC-A — Referência cruzada errada na seção MEL

**Localização:** `chapters/07-bibliographicRevision.tex`, parágrafo após fig:barkfeaturevect (linha ~71)

**Trecho original:**
```
como mostrado na  \autoref{fig:barkfeaturevect}.
```

**Trecho modificado:**
```
como mostrado na  \autoref{fig:melfeaturevect}.
```

**Justificativa:** O parágrafo está dentro da subseção da escala MEL e descreve a figura do pipeline MEL. A referência apontava para a figura Bark por engano.

---

## IC-C — Relação incorreta entre escalas Mel e Bark

**Localização:** `chapters/07-bibliographicRevision.tex`, início da subseção "A escala MEL" (linha ~67)

**Trecho original:**
```
\par Escala Mel, advinda do termo \textit{melody}, é uma adaptação da escala Bark para sinais de voz. Dentre as várias implementações de bandas críticas a escolhida foi a implementação que contém os valores em Hz:
```

**Trecho modificado:**
```
\par A escala Mel, advinda do termo \textit{melody}, é uma escala perceptual independente que mapeia frequências físicas em alturas tonais percebidas pelo ouvido humano, sem derivar da escala Bark. Dentre as várias implementações de bandas críticas a escolhida foi a implementação que contém os valores em Hz:
```

**Justificativa:** As escalas Mel e Bark são escalas perceptuais independentes. A escala Mel não é uma adaptação da escala Bark; ambas modelam a percepção auditiva humana mas foram desenvolvidas separadamente.

---

## IC-D + IC-E — Parágrafo MFCC: categorias, número de coeficientes e escopo de Δ/ΔΔ

**Localização:** `chapters/07-bibliographicRevision.tex`, parágrafo seguinte ao anterior (linha ~69)

**Trecho original:**
```
\par A variante que será usada neste trabalho é conhecida como \textit{Mel-frequency cepstral coefficients}(MFCC) a qual inclui, além dos intervalos definidos, uma diminuição da correlação entre os componentes gerados via aplicação da Transformada Discreta Cosseno (DCT) \cite{salomon2007data} ou da Análise de Componentes Principais (PCA) \cite{jolliffe2006principal} seguida de duas derivações no vetor de características resultando em um total de 11 coeficientes. Nesse trabalho foi escolhida a DCT, no entanto, PCA poderia também ser escolhida sem prejuízos, o uso de uma ou outra depende da preferência do autor.
```

**Trecho modificado:**
```
\par Neste trabalho, a escala Mel é avaliada em duas configurações distintas. Na primeira (Categoria~1), as energias calculadas nos intervalos Mel constituem diretamente o vetor de características, com 13 componentes ($V_1, V_2, \ldots, V_{13}$). Na segunda (Categoria~2), denominada \textit{Mel-frequency cepstral coefficients} (MFCC), aplica-se o logaritmo às energias por banda e em seguida a Transformada Discreta Cosseno (DCT) \cite{salomon2007data} para decorrelacionar os componentes, resultando em 13 coeficientes cepstrais. Adicionalmente, sobre os coeficientes da Categoria~2, são calculadas as primeiras e segundas derivadas temporais ($\Delta$ e $\Delta\Delta$), estendendo o vetor de características com informação dinâmica; essa etapa de derivação \textbf{não} é aplicada às características da Categoria~1.
```

**Justificativa:**
- IC-D (a): texto anterior caracterizava MFCC como "a variante usada", ignorando que o trabalho avalia separadamente Mel-band energies (Categoria 1) e MFCC (Categoria 2) — guide00.md define 6 tipos de características em duas categorias independentes.
- IC-D (b): texto anterior indicava 11 coeficientes; a figura `melFeatureVect.tex` exibe $V_1$ a $V_{13}$ (13 coeficientes).
- IC-E: Δ/ΔΔ aparecia sem restrição de escopo; a figura `barkFeatureVect.tex` não possui Δ/ΔΔ, enquanto `melFeatureVect.tex` os possui. A derivação temporal aplica-se exclusivamente à Categoria 2 (coeficientes cepstrais).
- PCA removida: guide00.md não inclui PCA em nenhum pipeline definido.

---

## IC-F — Listagem incompleta de características na metodologia proposta

**Localização:** `chapters/08-proposedApproach.tex`, linha ~144

**Trecho original:**
```
serão realizados os agrupamentos energéticos nas bandas Bark e MEL para a extração de características.
```

**Trecho modificado:**
```
serão realizados os agrupamentos energéticos em três esquemas de particionamento espectral --- bandas lineares, bandas Mel e bandas Bark --- gerando, na Categoria~1, os vetores de \textit{Linear-band energies}, \textit{Mel-band energies} e \textit{Bark-band energies}, respectivamente. Para a Categoria~2, aplica-se adicionalmente o logaritmo e a DCT sobre cada um desses vetores, produzindo os coeficientes LFCC, MFCC e BFCC.
```

**Justificativa:** O texto original omitia: Linear-band energies, LFCC e BFCC. guide00.md define 6 tipos de características a avaliar (3 da Categoria 1 + 3 da Categoria 2). A metodologia proposta deve enumerá-los todos.

---

## IC-G — PCA no box DCT da figura MEL

**Localização:** `images/melFeatureVect.tex`, nó dctbox e labels adjacentes

**Trecho original:**
```latex
\node[mbox, minimum height=1.0cm] (dctbox) at (0,-7.4) {Aplicação da DCT ou da PCA};
\node[right, font=\footnotesize, align=left] at (\xLbl,-7.1)
    {DCT -- \textit{Discrete cosine transformation}};
\node[right, font=\footnotesize, align=left] at (\xLbl,-7.7)
    {PCA -- \textit{Principal component analysis}};
```

**Trecho modificado:**
```latex
\node[mbox, minimum height=1.0cm] (dctbox) at (0,-7.4) {Aplicação da DCT};
\node[right, font=\footnotesize, align=left] at (\xLbl,-7.4)
    {DCT -- \textit{Discrete cosine transformation}};
```

**Justificativa:** guide00.md define o pipeline MFCC como: energias → log → DCT → vetor cepstral. PCA não faz parte de nenhum dos seis pipelines definidos. A figura deve refletir exclusivamente o fluxo adotado.

---

## Reestruturação da subseção `\subsection{Escalas e energias dos sinais}`

**Localização:** `chapters/07-bibliographicRevision.tex`, linhas 55–92 (parágrafo introdutório + 3 subsubseções antigas)

**Estrutura anterior:**
- Parágrafo introdutório mencionando apenas BARK e MEL
- `\subsubsection{A escala BARK}` — Bark-band energies
- `\subsubsection{A escala MEL}` — Mel-band energies + MFCC misturados em uma seção
- `\subsubsection{A escala LFCC}` — LFCC

**Estrutura nova (6 subsubseções):**
1. `\subsubsection{Energias em bandas lineares}` — figura `linearBandEnergies.tex`
2. `\subsubsection{Energias em bandas Mel}` — figura `melBandEnergies.tex`
3. `\subsubsection{Energias em bandas Bark}` — figura `barkFeatureVect.tex` (mantida)
4. `\subsubsection{Coeficientes LFCC}` — figura `lfccFeatureVect.tex` (mantida; frase "ou os coeficientes pós-DCT" removida)
5. `\subsubsection{Coeficientes MFCC}` — figura `melFeatureVect.tex` (legenda corrigida para MFCC)
6. `\subsubsection{Coeficientes BFCC}` — figura `bfccFeatureVect.tex` (nova)

**Justificativa:** guide00.md define 6 tipos de características em duas categorias independentes. A estrutura anterior agrupava tipos distintos e omitia Linear-band energies, MFCC como entidade própria e BFCC completamente. Cada tipo requer definição separada e figura própria para clareza.

**Figuras novas criadas:**
- `images/linearBandEnergies.tex` — pipeline bandas lineares (sem log/DCT)
- `images/melBandEnergies.tex` — pipeline bandas Mel (sem log/DCT)
- `images/bfccFeatureVect.tex` — pipeline BFCC (bandas Bark → log → DCT → coeficientes)

---

## IC-H — Questões científicas ausentes do documento

**Localização:** `chapters/06-Introduction.tex` (subsection Objetivos) e `chapters/08-proposedApproach.tex` (section Estrutura da estratégia proposta)

**Problema:** guide00.md exige que o documento reflita explicitamente as duas questões científicas independentes que a metodologia foi estruturada para responder. Nenhum capítulo formulava essas questões.

**Correções aplicadas:**

1. `06-Introduction.tex` — adicionado `\label{chap:intro}` ao capítulo e inserido parágrafo com `\begin{description}` formulando Q1 e Q2 explicitamente ao final de `\subsection{Objetivos}`.

2. `08-proposedApproach.tex` — parágrafo da estratégia proposta expandido com sentença que vincula a organização em Categoria~1/Categoria~2 às Questão~1 e Questão~2 via `\autoref{chap:intro}`. Também corrigido "configuração 1/2" → "Categoria~1/2" no mesmo parágrafo.

**Questões formuladas:**
- Q1 (agrupamento espectral): Linear-band energies × Mel-band energies × Bark-band energies
- Q2 (transformação cepstral): Linear-band energies × LFCC; Mel-band energies × MFCC; Bark-band energies × BFCC

---

## Inconsistências não corrigidas

### IC-B — DFT na definição padrão de LFCC

**Localização:** `chapters/07-bibliographicRevision.tex`, linha ~84

**Decisão:** Mantido. A seção descreve o método LFCC conforme a literatura (onde DFT é a transformada padrão). A substituição por DTWPT é uma decisão metodológica do trabalho descrita na seção de metodologia, não na fundamentação teórica.