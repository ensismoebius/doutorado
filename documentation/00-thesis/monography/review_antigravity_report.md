# Relatório de Revisão Integral da Tese de Doutorado

**Data da revisão:** 17 de novembro de 2026  
**Commit revisado:** `5a35becec`  
**Modelo avaliador:** Gemini 2.5 Pro (Google Antigravity)  
**Total de páginas lidas:** 179 páginas (PDF completo, da capa aos apêndices; páginas impressas 19 a 178 para o corpo principal, com convenção estrita `página impressa = página PDF - 1`).

---

## 1. Metodologia e Escopo da Leitura

A revisão foi executada de forma independente sobre o documento compilado `documentation/00-thesis/monography/monografia.pdf`, os códigos-fonte LaTeX em `chapters/*.tex`, as tabelas em `tables/` e a bibliografia em `bibliography.bib`, confrontando sistematicamente as afirmações do texto com o código de implementação em `software/nn/` (`ThesisHandcraftedFeatures.cpp`, `ThesisFeatureExtraction.cpp`, `ThesisParaconsistent.cpp`, `ThesisClassifiers.cpp`, `ThesisClassifiersInternal.hpp`, `ThesisDsnnClassifier.hpp`, `ThesisAnnClassifier.hpp`, `eer_scorer.cpp`, `VoiceDegradation.cpp`, `VoiceDysphonia.cpp`, scripts de pipeline e manifestos de perfis).

Nenhum arquivo além deste relatório foi editado, nenhum experimento ou script de treino foi disparado e nenhum commit ou push foi realizado, em estrito cumprimento às instruções de `review_antigravity.md`. As marcações preexistentes `[PENDENTE: ...]` foram preservadas e não constam como defeitos.

---

## 2. Achados por Grupo

### Grupo A — Divergência entre Texto e Método Executado
*(O texto descreve um método diferente do que foi implementado/executado no código)*

| # | Onde | Problema | Evidência | Status |
|---|------|----------|-----------|--------|
| **A1** | Pág. 108 (PDF 109), Seção 3.3.1.1.1, Tabela 10 | A Tabela 10 e o texto associado afirmam que a entrada dos autoencoders para "Fusão precoce" possui "256 valores" idênticos aos da voz e do EEG ("Entrada (256 valores)"). No entanto, a fusão precoce é construída pela concatenação da forma de onda temporal bruta da voz (176.400 amostras) com os seis canais do EEG (24.576 amostras), totalizando 200.976 amostras, antes da aplicação da estimativa espectral de Welch a 44,1 kHz. Embora o vetor Welch final tenha 256 bandas acima do DC, o texto omite que a taxa de amostragem da voz (44,1 kHz) foi imposta a todo o vetor (distorcendo as frequências físicas do EEG em ~43x), o que torna a entrada conceitualmente heterogênea em relação aos 256 bins das fontes isoladas. | `08-proposedApproach.tex:79, 129-136`; `software/nn/src/experiments/thesis/lib/src/ThesisFeatureExtraction.cpp:319-335`. | Confirmado |
| **A2** | Pág. 31 (PDF 32), Seção 2.1.4.1, Equação (2.4) | Na Equação (2.4), a DCT do LFCC é definida como $c_n = \sum_{k=1}^K \log(E_k) \cos\big(\pi n \frac{2k-1}{2K}\big)$, com $n = 0, \ldots, N-1$. No texto, diz-se que $E_k$ é a energia da banda e $N=K=16$. Contudo, na implementação C++ (`ThesisHandcraftedFeatures.cpp:189-205`), a fórmula implementada para a DCT-II é $C[k] = \sum_{n=0}^{N-1} x[n] \cos\big(\frac{\pi}{N}(n+0{,}5)k\big)$. Embora matematicamente equivalentes sob mudança de índice ($n+0{,}5 = \frac{2n+1}{2} = \frac{2(n+1)-1}{2}$), a troca dos papéis das variáveis $n$ (ordem cepstral na tese, índice temporal/banda no código) e $k$ (índice da banda na tese, ordem cepstral no código) gera confusão na leitura cruzada entre a especificação da tese e o código. | `07-bibliographicRevision.tex:129-131` vs `software/nn/src/experiments/thesis/lib/src/ThesisHandcraftedFeatures.cpp:189-205`. | Confirmado |

---

### Grupo B — Inconsistências Internas, Afirmações sem Sustentação e Falhas de Notação
*(Afirmação mais forte do que os dados sustentam, contradição entre trechos, ou cálculo aritmético incorreto)*

| # | Onde | Problema | Evidência | Status |
|---|------|----------|-----------|--------|
| **B1** | Pág. 46 (PDF 47), Seção 2.1.7.1, Linhas 514–515 | No "Exemplo numérico" de $D_{\text{penalizado}}$, o texto afirma: *"Para o caso degenerado discutido acima ($\alpha=\beta=1$, $G_1=0$, $G_2=1$, $D_{1,0}=\sqrt{2}$): $D_{\text{penalizado}} = \sqrt{2} + 0{,}5858 \times 1 \approx 2{,}4142$ — o pior valor possível, e não mais um dos melhores."* O cálculo aritmético está incorreto: $\sqrt{2} + 0{,}5858 \times 1 = 1{,}41421 + 0{,}58579 = 2{,}0000$ (e não $2{,}4142$). O valor $2{,}4142$ corresponderia a $\sqrt{2} + 1{,}0$, resquício de quando a fórmula usava penalidade unitária ($\lambda=1$). Além disso, a afirmação contradiz diretamente a dedução matemática apresentada quatro linhas acima (Equação 2.21 e texto imediatamente anterior), onde se prova expressamente que $\lambda = 2 - \sqrt{2}$ foi escolhido justamente para que Ambiguidade atinja exatamente $D_{\text{penalizado}} = 2{,}0$. | `07-bibliographicRevision.tex:508-515`; cálculo $\sqrt{2} + (2-\sqrt{2}) = 2{,}0000$. | Confirmado |
| **B2** | Pág. 59 (PDF 60), Seção 2.1.10.2, Linha 920 | No exemplo numérico da discretização de Euler da equação diferencial do potencial de membrana, o texto define $R=5$, $C=1$, portanto $\tau = R \cdot C = 5$, e escolhe passo de integração $dt=1$. Na linha 920, porém, o texto afirma: *"um passo de Euler tão grosseiro ($dt=\tau$) já se aproxima da assíntota..."*. Como $dt=1$ e $\tau=5$, tem-se $dt \neq \tau$ (na verdade $dt = \tau/5$). A escrita "($dt=\tau$)" contradiz os valores numéricos recém-calculados na mesma frase. | `07-bibliographicRevision.tex:916-921`. | Confirmado |
| **B3** | Pág. 31 (PDF 32), Seção 2.1.4.1, Linhas 128–132 | Redundância e imprecisão na definição das energias logarítmicas: o texto introduz *"Seja $E_k$ a energia logarítmica da banda $k$..."* e, logo em seguida, na Equação (2.4), aplica o logaritmo novamente: $c_n = \sum_{k=1}^K \log(E_k) \cos(...)$. Se $E_k$ já representasse a energia logarítmica, $\log(E_k)$ seria o logaritmo do logaritmo. Mais adiante (linha 132), o texto escreve $\log(E_1)=1, \ldots$, e na linha 136 esclarece que $E_k$ é a energia da sub-banda e a DCT opera sobre o logaritmo dessas energias. O texto da linha 128 deveria dizer *"Seja $E_k$ a energia da banda $k$..."*. | `07-bibliographicRevision.tex:128-136`. | Confirmado |
| **B4** | Pág. 114 (PDF 115), Seção 3.3.1.2.3, Linha 219 | O texto define a AUC como $\text{AUC} = P(\text{pontuação genuína} > \text{pontuação impostora})$. Embora seja a forma textual comum da estatística de Mann-Whitney, para distribuições discretas ou escores com empates a formulação exata executada pelo algoritmo (`eer_scorer.cpp:170`) atribui peso $0{,}5$ aos empates: $\text{AUC} = P(S_{\text{gen}} > S_{\text{imp}}) + 0{,}5\,P(S_{\text{gen}} = S_{\text{imp}})$. A omissão do termo de desempate na definição formal é imprecisa. | `08-proposedApproach.tex:219` vs `software/nn/src/core/statistics/eer_scorer.cpp:167-171`. | Confirmado |
| **B5** | Pág. 154 (PDF 155), Seção 5.2, Linha 200 | Presença de número com dado empírico hardcoded no corpo do texto sem chave `\tn{...}`: *"entre 2 e 40\,Hz, 98,1\% da potência cai na primeira sub-banda..."*. O valor $98{,}1\%$ provém dos cálculos de densidade espectral da Tabela 9 (`tab:potenciaEeg`), mas foi digitado diretamente no LaTeX do Capítulo 5 (e repetido no Capítulo 2, linhas 1513, 1522, 1537) sem passar por macro gerada em `tables/thesis_numbers.tex`. | `10-conclusions.tex:200` e `07-bibliographicRevision.tex:1513, 1522, 1537`. | Confirmado |

---

### Grupo C — Forma, Gramática, Tipografia e Bibliografia
*(Ortografia, normas ABNT, referências bibliográficas, formatação de fórmulas e tabelas)*

| # | Onde | Problema | Evidência | Status |
|---|------|----------|-----------|--------|
| **C1** | Pág. 38 (PDF 39), Seção 2.1.6.1, Equação (2.7), Linha 310 | Na Equação (2.7) (`eq:haarMultiplic`), na 4ª linha da matriz de coeficientes da transformada de Haar, o quarto elemento foi digitado incorretamente sem o sinal negativo: `0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}`, ficando idêntica à 3ª linha. Multiplicando essa linha por $(1, 2, 3, 4)^T$, o produto resultaria em $7/\sqrt{2}$; no entanto, o vetor resultado exibe corretamente $-1/\sqrt{2}$, o qual só decorre se a linha for `0 & 0 & \frac{1}{\sqrt{2}} & \frac{-1}{\sqrt{2}}`, tal como grafado na Equação (2.6) (linha 295). Trata-se de erro tipográfico na fórmula matemática. | `07-bibliographicRevision.tex:310`, impresso na pág. 38. | Confirmado |
| **C2** | Pág. 38 (PDF 39), Seção 2.1.6.1, Linha 289 | Erro de sintaxe com pleonasmo pronominal: *"Se o tamanho do sinal a ser tratado é quatro e se pretende-se aplicar o filtro Haar..."*. Ocorrência de duplo pronome reflexivo/apassivador "se" (proclítico e enclítico na mesma oração). Deve ser *"e se pretende aplicar"* ou *"e pretende-se aplicar"*. | `07-bibliographicRevision.tex:289`, pág. impressa 38. | Confirmado |
| **C3** | Págs. 7 e 8 (PDF 8 e 9), Resumo e Abstract | Citação ABNT incorreta no meio do texto: `avaliada na base pública de \cite{10.1117/12.2255697}` no Resumo e Abstract gera *"avaliada na base pública de (CORETTO; GAREIS; RUFINER, 2017)"* no Resumo em português e *"evaluated on the public database of (CORETTO; GAREIS; RUFINER, 2017)"* no Abstract em inglês. Pela norma ABNT e pela gramática, quando a citação faz parte da oração deve-se usar `\citeonline{...}`, produzindo *"avaliada na base pública de Coretto, Gareis e Rufiner (2017)"*. | `04-abstract.tex:2` e `04-abstract.tex:9`. | Confirmado |
| **C4** | Pág. 14 (PDF 15), Lista de abreviaturas e siglas | A sigla `STDP` está posicionada após `SVM` na Lista de abreviaturas e siglas. A NBR 14724 da ABNT exige ordem alfabética estrita, de modo que `STDP` deve preceder `SVM` (`S-N` < `S-T` < `S-V`). O próprio cabeçalho do arquivo ressalta a exigência da ordem alfabética como norma obrigatória. | `05-abreviations.tex:42-45` e PDF pág. 15. | Confirmado |
| **C5** | Arquivo `bibliography.bib`, linhas 404 e 415 | Chave bibliográfica duplicada: a entrada `@ARTICLE{6400235, ...}` (artigo de Shahidi Zandi et al., 2013) está declarada duas vezes consecutivas no arquivo `bibliography.bib` com a mesma chave e os mesmos campos. | `documentation/00-thesis/monography/bibliography.bib:404, 415`. | Confirmado |
| **C6** | Pág. 163 (PDF 164), Referências Bibliográficas | Título em caixa baixa para sigla não protegida por chaves no BibTeX: na entrada `MahapatraNrushinghCharan2023Ecoi`, o campo `title` contém `EEG-based classification...` sem chaves em `{EEG}`. Como o estilo BibTeX `abntex2-alf` converte títulos para minúsculas, a referência impressa no PDF saiu como *"MAHAPATRA, N. C.; BHUYAN, P. Eeg-based classification of imagined digits..."*, com a sigla corrompida para "Eeg". O mesmo ocorre com `{BCI}`, `{SVM}` e `{CNN}` em outras entradas como `CHINTA2023101477` e `rusnac2021eeg`. | `bibliography.bib:2307` e PDF impresso pág. 163. | Confirmado |
| **C7** | Pág. 174 (PDF 175), Apêndice C, Seção C.2, Equação (C.3) | Na fórmula da divergência KL para esparsidade em autoencoders, a média de ativação da unidade $i$ é definida no texto como $\hat{\rho}_i$, mas na fórmula do somatório foi grafada sem o índice: $\Omega(h) = \sum_i \big(\rho \log(\rho / \hat{\rho}) + (1 - \rho) \log((1 - \rho) / (1 - \hat{\rho}))\big)$, omitindo o subscrito $i$ em ambas as ocorrências de $\hat{\rho}$. | `11-appendices.tex:534-537`. | Confirmado |
| **C8** | Pág. 84 (PDF 85), Seção 2.2, Tabela 8, Linha 4 | Na Tabela 8 de trabalhos do estado da arte, a 4ª linha (artigo de \citeonline{math11194205}) contém o texto `base CN-Celeb` na coluna cujo cabeçalho é o número de sujeitos/locutores, quebrando a padronização numérica da tabela (onde todas as outras linhas contêm contagens como 21, 25, 23, mais de 3800, etc.). | `07-bibliographicRevision.tex:1487`. | Confirmado |

---

## 3. Seções e Aspectos Não Verificados

1. **Hardware neuromórfico físico e medição física de corrente/potência:**
   A tese não realizou medição em chips neuromórficos (como Loihi ou TrueNorth) nem ensaios de bancada física. Toda a energia de inferência reportada provém de um modelo analítico computado a partir de contagem de operações e de acessos à memória SRAM/DRAM em processo de 45 nm (`ThesisClassifiersInternal.hpp:74-105`), conforme explicitamente advertido no texto.
2. **Avaliação subjetiva/perceptual com ouvintes dos perfis WORLD:**
   Os níveis de rugosidade, tremor e quebras de sonoridade foram calibrados exclusivamente por métricas acústicas automatizadas (via Praat e rotinas em C++), sem painel de fonoaudiólogos ou testes MUSHRA com ouvintes humanos, conforme expressamente declarado pelo autor.
3. **Avaliação clínica com pacientes disfônicos reais:**
   As duas bases avaliadas (Coretto e BIOMEX-DB) contêm apenas sujeitos saudáveis; a disfonia avaliada é 100% simulada sinteticamente (por substituição de fonte via ruído LPC ou por edição de contorno F0/aperiodicidade no WORLD).
4. **Campos pré-textuais e apêndice pendentes:**
   Páginas marcadas com `[PENDENTE: ...]`, incluindo ficha catalográfica (p. 3), banca (p. 4), dedicatória (p. 5), agradecimentos (p. 6), epígrafe (p. 7) e figura do Apêndice D (p. 178), foram deixadas intocadas por dependerem de definições do autor e da universidade.

---

## 4. Comparação com a Primeira Leitura (`issues.md`)

Conforme estipulado na Seção 3.6 do protocolo, após a conclusão e gravação do relatório independente acima, realizou-se a leitura integral de `documentation/00-thesis/monography/issues.md` (com foco na seção *"Leitura integral do PDF (2026-11-17)"*). A confrontação entre os dois levantamentos revela o seguinte panorama:

### 4.1 Achados novos (não constavam em `issues.md`)

Os seguintes 11 achados identificados nesta revisão são inteiramente novos:

1. **B1 — Cálculo aritmético e contradição conceitual em $D_{\text{penalizado}}$ (pág. 46 / PDF 47):** A conta $\sqrt{2} + 0{,}5858 \times 1 \approx 2{,}4142$ está aritmeticamente errada ($= 2{,}0000$) e contradiz a dedução teórica recém-apresentada de que $\lambda = 2 - \sqrt{2}$ iguala Ambiguidade a exatamente $2{,}0$. `issues.md` tratou do valor de $\lambda$ e de Ambiguidade conceitualmente, mas não detectou esse erro de conta e contradição no exemplo numérico.
2. **B2 — Contradição interna no passo de integração da discretização de Euler ($dt=\tau$) (pág. 59 / PDF 60):** O texto afirma `dt = \tau` logo após calcular com $dt=1$ e $\tau=5$. Não constava em `issues.md`.
3. **B3 — Redundância conceitual entre $E_k$ e $\log(E_k)$ no LFCC (pág. 31 / PDF 32):** Definição de $E_k$ como "energia logarítmica" seguida de nova aplicação do logaritmo na fórmula. Não constava em `issues.md`.
4. **B4 — Omissão de empates na definição formal da AUC (pág. 114 / PDF 115):** A fórmula omite o termo de $0{,}5 \times P(\text{empate})$ implementado no código. Não constava em `issues.md`.
5. **B5 — Dado empírico hardcoded sem macro no Capítulo 5 (pág. 154 / PDF 155):** O número "98,1\%" foi digitado como texto puro no Capítulo 5 e no Capítulo 2, descumprindo a geração automatizada via `thesis_numbers.tex`. Não constava em `issues.md`.
6. **A1 — Heterogeneidade conceitual da entrada de 256 valores para a fusão precoce na Tabela 10 (pág. 108 / PDF 109):** Omissão, na Tabela 10, de que a taxa da voz de 44,1 kHz foi imposta ao sinal conjunto, distorcendo as bandas do EEG em ~43x. Não constava em `issues.md`.
7. **C2 — Pleonasmo pronominal "e se pretende-se" (pág. 38 / PDF 39):** Erro gramatical em `07-bibliographicRevision.tex:289`. Não constava em `issues.md`.
8. **C3 — Citação no meio da oração com `\cite{...}` no Resumo e Abstract (págs. 7 e 8 / PDF 8 e 9):** Gera "(CORETTO; GAREIS; RUFINER, 2017)" como sujeito gramatical em vez de usar `\citeonline{...}`. Não constava em `issues.md`.
9. **C4 — Ordem alfabética violada por STDP na Lista de Abreviaturas e Siglas (pág. 14 / PDF 15):** STDP posicionado após SVM. Não constava em `issues.md`.
10. **C5 — Chave `@ARTICLE{6400235}` duplicada no BibTeX:** Duplicata consecutiva no arquivo `bibliography.bib` (linhas 404 e 415). `issues.md` registrou apenas a duplicata de `ABDULGHANI 2023`.
11. **C7 — Subscrito $i$ omitido em $\hat{\rho}$ na penalidade de esparsidade do Apêndice C (pág. 174 / PDF 175):** O somatório sobre $i$ traz $\hat{\rho}$ sem o índice $i$. Não constava em `issues.md`.
12. **C8 — Quebra de padrão numérico na Tabela 8 com "base CN-Celeb" (pág. 84 / PDF 85):** Texto preenchendo coluna numérica de sujeitos. Não constava em `issues.md`.

---

### 4.2 Achados que já constavam em `issues.md` e que o texto atual AINDA não corrige

1. **C1 (Matriz de Haar na Equação 2.7, pág. 38 / PDF 39):** Em `issues.md` (linha 830), apontou-se *"Eq. 2.6 parece sem o sinal de menos na 4ª linha"*. Na versão atual (commit `5a35becec`), a Equação 2.6 está correta, mas a Equação 2.7 (`eq:haarMultiplic`, linha 310) **permanece incorreta**, com a 4ª linha `0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}}` (positiva), produzindo incongruência com o vetor resultante.
2. **C6 (Siglas em caixa baixa no PDF por desproteção de chaves no BibTeX):** Em `issues.md` (linha 845), apontou-se *"caixa sem chaves ('eeg', 'Cn-celeb', 'broca’s', 'russian')"*. A versão atual ainda possui entradas sem chaves em `{EEG}`, `{SVM}`, `{BCI}` e `{CNN}` (como `MahapatraNrushinghCharan2023Ecoi`, `CHINTA2023101477` e `rusnac2021eeg`), resultando na impressão de *"Eeg-based"* no PDF na página 163.

---

### 4.3 Itens de `issues.md` dados como corrigidos que se confirmam NÃO corrigidos

1. **Item C — Linha 848 de `issues.md`:** A nota declara *"C corrigido em 2026-11-17 (commit do grupo C)"*. No entanto, conforme verificado no PDF e nos fontes LaTeX compilados:
   - O sinal negativo na 4ª linha da matriz de coeficientes de Haar da Equação (2.7) (**não** foi corrigido);
   - A proteção de caixa alta para títulos BibTeX de artigos de periódicos (**não** foi aplicada a todas as entradas, vide pág. 163 do PDF);
   - A entrada duplicada de Shahidi Zandi et al. (`6400235`) no arquivo `bibliography.bib` (**não** foi removida).
