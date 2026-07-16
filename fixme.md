# TODO — ativas

## Pipeline Experiment05 (crítico para a tese, ordem de prioridade)

1. Rodar `01_e05_phase00_rank.py` sobre o conjunto de 300 perfis agora completo para produzir `winners.json`.
2. Rodar `02_e05_apply_winner.py` para injetar o vencedor real nos 32 perfis da phase01, substituindo o bloco `feature_extraction` provisório.
3. Executar os 32 perfis `classifier.type=dsnn` da phase01 (`run_e05_profiles.sh phase01`) --- este é o experimento de autenticação real da tese e atualmente não tem nenhum resultado.
4. Com resultados reais do DSNN em mãos, considerar uma rodada explícita de ablação para `weight_decay`, `firing_rate_reg_lambda` e tdBN --- nenhum dos três jamais foi exercitado fora do perfil debug/smoke:
   - [ ] `training.weight_decay` > 0 --- só configurado em `debug.json`/`smoke/debug.json`; não está em nenhum perfil real da phase01
   - [ ] `training.firing_rate_reg_lambda` > 0 --- mesma situação: apenas debug/smoke
   - [ ] `training.batch_normalization = "threshold-dependent"` (tdBN) --- caminho de código implementado (`E05Config.hpp:135`, `ThresholdDependentBatchNorm`), mas nunca configurado em nenhum perfil publicado, debug incluso --- ainda genuinamente não testado de ponta a ponta

Status completo verificado: ver "Log de status do Experiment05" abaixo.

## Questões em aberto

- Estou vendo que você vai testar apenas o perfil de autoencoder com 4 camadas, mas acho bem provável que o estado da arte para dispositivos de baixo poder computacional (tipo Raspberry Pi B) use mais camadas. Estou errado?

## Aprofundar (revisão de texto/tese)

- [x] 24. Verificar consistência entre as equações do LIF e o código utilizado. ✓ feito --- inconsistência real encontrada: a tese deriva um passo de Euler explícito $V_{mem}(t+\Delta t) = V_{mem}(t) + \frac{\Delta t}{\tau}(R.I_{in}(t) - V_{mem}(t))$ (\autoref{eq:membraneDerivative}, `lst:membranepotentialfull`), mas `LifImpl`/`LifBPTTImpl`/`LifIntegrator` (`include/layers/spiking/*.hpp`) na verdade usam a recorrência exata $V_{mem}(t+\Delta t) = \beta \cdot V_{mem}(t) + I_{in}(t)$, $\beta=e^{-\Delta t/\tau}$ --- sem escalar $I_{in}$ por $R$. Adicionado parágrafo "Nota de implementação" em `chapters/07-bibliographicRevision.tex` (após a Equação 2.31, antes de "Treinamento") explicando a diferença e por que é uma simplificação deliberada (desacopla o ganho de entrada de $\tau$, facilitando treinar $R,C,V_{th}$). Tese recompila limpo (111 pág.).
- [x] 25. Explicar, na figura dos pulsos do neurônio LIF, que os pulsos apresentados são resultado de uma simulação de neurônios de pulso/RNP. ✓ feito --- legenda da `fig:neuronspike` (`chapters/07-bibliographicRevision.tex`, "Sinal ruidoso e os respectivos pulsos produzidos") agora explicita que os pulsos vêm de uma simulação de neurônio RNP/LIF em resposta ao sinal ruidoso, ilustrando o disparo por limiar discutido no texto. Tese recompila limpo.
- [ ] 30. Seção BPTT: ESTUDAR!!!!!
- [x] 42. Criar tabela comparativa entre métodos manuais, automatizados, escalas e wavelets. ✓ feito --- `chapters/09-testsAndResults.tex` (`\Phasezerotable`), 4 longtables geradas automaticamente a partir de `results/phase00/*_summary.json` via `e05_build_phase00_paraconsistent_tables.py`: handcrafted (wavelet × escala × categoria) e autoencoder (ANN-AE/SNN-AE × codificação), para EEG e voz.
- [ ] 47. Criar seção específica para Threshold-Dependent Batch Normalization (TDBN). ✓ seção criada e verificada (`chapters/07-bibliographicRevision.tex:912`, "Normalização em Lote Dependente do Limiar (tdBN)") --- conteúdo substancial (motivação, derivação, exemplo numérico, figura, vantagens/limitações) e, ao contrário do item 24, **consistente** com o código: `include/layers/spiking/ThresholdDependentBatchNorm.hpp` implementa exatamente $Y=\gamma(\alpha V_{th}\hat X)+\beta$ com estatísticas agrupadas sobre lote+tempo ($N=T\cdot B$), $\alpha V_{th}$ escalando apenas o termo normalizado, e os mesmos defaults ($\alpha=1$, $\varepsilon=10^{-5}$). Falta ainda a parte "ESTUDAR!!!" (aprofundamento pessoal), por isso mantido em aberto.
- [x] 48. Revisar toda a monografia/Wiki para garantir que toda variável seja explicitamente definida. ✓ feito --- **Tese**: auditados os ~55 equações do cap. 07 e as 30 equações "vivas" do cap. 11, mais a matemática inline dos caps. 08/09; único gap real encontrado foi deriva notacional entre capítulos para a mesma grandeza ($D_{1,0}$ no cap. 07, $D_{\text{verdade}}$ no cap. 08, $d$/$\bar d$/$\sigma_d$ no cap. 09) --- corrigido em `chapters/09-testsAndResults.tex` linkando explicitamente as três notações. **Wiki** (`software/nn/.wiki/`, 38 arquivos com matemática): (1) `Concepts/Membrane-Dynamics.md` tinha o mesmo bug do item 24 ($v[t]=\beta v[t-1]+R\cdot I[t]$, contradizendo o próprio trecho de código exibido na mesma página, que não escala por $R$) --- corrigido, com nota explicando a simplificação deliberada, igual à tese; (2) `Core/Initializers.md` descrevia Kaiming/He como Gaussiana com um $\alpha$ (leak rate) nunca definido, contradizendo o código `kaimingSNNInitializer` mostrado na mesma página (que é He-uniform) e a página irmã `Concepts/Weight-Initialisation.md` (já correta) --- corrigido para bater com o código; (3) `Experiments/Experiment05.md` (tabela "Measured separability") afirmava a direção invertida de $\alpha$/$D_{\text{truth}}$ (menor $\alpha$ e maior $D_{\text{truth}}$ = melhor), contradizendo a própria definição dada mais acima no mesmo arquivo e em `Core/Paraconsistent.md` (maior $\alpha$, menor $D_{\text{truth}}$ = melhor) --- corrigido o texto da direção; por pedido do usuário, o ranking/valores da tabela em si **não** foram re-derivados por mim, apenas sinalizados para reverificação manual contra `results/phase00/*_paraconsistent.csv`. Demais ~35 arquivos da Wiki e o restante da tese: sem inconsistências encontradas.
- [ ] 49. Avaliar arquiteturas compactas de autoencoders para Raspberry Pi.
- [ ] 51. Avaliar opcionalmente diferentes algoritmos de otimização.
- [x] 56. Avaliar comparativamente diferentes arquiteturas de autoencoders utilizando Engenharia Paraconsistente de Características. ✓ feito --- mesmas tabelas do item 42 (`tab:phase00ae_eeg`/`tab:phase00ae_voice`), ANN-AE vs SNN-AE (direct/latency/poisson) rankeados por $\bar d$ via EPC, com as 300 execuções da Phase 00 completas (`results/run_profiles_phase00.state`: 300/300 PASS).
- [x] 57. Verificar e fundamentar a afirmação sobre taxas de aprendizado para parâmetros biofísicos em SNNs. ✓ feito --- a afirmação ("R, C, V_th precisam de lr ~10× menor que os pesos", `snn_lr_scale=0.1`) não existe na tese, só na Wiki/código (`Core/Optimizers.md`, `Core/Training.md`, `include/optimizers/Adam.hpp`). Encontrada citação fabricada/incorreta sustentando a afirmação: "[37] Y. Cao et al., 'Direct training of spiking neural networks: Challenges and insights,' Frontiers in Neuroscience, 2025" --- busca na web não encontrou esse artigo (provável alucinação de sessão anterior). Além disso, `References.md` tinha um "[37]" *diferente* (mesmo número, dessincronizado) atribuído a "H. Fang et al." para um artigo real (DOI 10.3389/fnins.2026.1795946) cujos autores verdadeiros são Hou, Wu e Zhou (não Fang) e cujo tema (surrogate gradients) nada tem a ver com taxas de aprendizado. Verifiquei também o artigo real de Fang et al. 2021 (PLIF, já citado corretamente em `Membrane-Dynamics.md`) --- não especifica redução de 10× no lr. Resolução (confirmada com o usuário): removida a citação falsa de `Core/Optimizers.md`, `Core/Training.md` e `Adam.hpp`, reformulada a afirmação como escolha de engenharia empírica deste projeto (não amparada por literatura específica), e corrigida a autoria em `References.md`. Valor numérico (`snn_lr_scale=0.1`) mantido, pois é o default real do código, não uma alegação factual sobre terceiros.
- [ ] 58. Fundamentar a tabela de associação entre codificações e funções de perda.

---

# Adiadas / rejeitadas

7. Incluir passo a passo do cálculo das transformadas wavelet packet. (nem fodendo)
8. Listar outras técnicas além da engenharia paraconsistente para consistência de características. (por enquanto não)

---

# Resolvidas

## C12 --- fusão voice+EEG não implementada (wiki x código)

Wiki (`Experiment05.md`:134,390) diz que `modality=fused` concatena os vetores de características de voz+EEG. Código (`E05FeatureExtraction.cpp::signal_for_modality`, ramo `else // "fused"`) na verdade escolhia áudio se presente, senão EEG — um único sinal, sem concatenação. Não era fusão precoce (fundir sinal bruto antes do autoencoder/handcrafted) nem fusão tardia (concatenar vetores de características depois) — nenhuma fusão ocorria.

Ação:
(a) implementar fusão tardia real
(b) implementar fusão precoce
(c) a distinção fusão-precoce-vs-tardia deve ser discutida na tese/wiki como eixo experimental

**RESOLVIDO (2026-07-03):**
(a) Fusão tardia implementada. `extract_features` com `modality=fused, fusion_mode=late` (padrão) extrai voz e EEG independentemente (cada um na taxa nativa, 44100/1024 Hz) e concatena os vetores por amostra `[voz ‖ eeg]`. `E05FeatureExtraction.cpp`.
(b) Fusão precoce implementada. `fusion_mode=early` concatena os sinais brutos (voz seguida de EEG) e extrai numa única passagem, usando a taxa da voz. Novo campo `E05Config::Dataset::fusion_mode` (validado early/late).
(c) Eixo discutido na tese (cap. 08, itemize fusão precoce/tardia) e na wiki (`Experiment05.md`, tabela fusion_mode). README + schema do perfil atualizados; perfis `*-fused.json` agora declaram `fusion_mode: late` explicitamente.
    Testes: 3 novos (E05Fusion.*) — dimensão tardia = voz+eeg, precoce difere e rotula distinto, modo inválido lança. Suítes e05 verdes (feat 26, profile 56, classifiers 16). Tese compila (101 pág).
    Bug original corrigido de passagem: `signal_for_modality` (fallback áudio-senão-EEG, sem fusão) foi substituído; `modality_sample_rate` removido em favor de constantes por sinal `kVoiceSampleRate`/`kEegSampleRate`.

## Falhas do lote da phase00 (4 perfis daub32)

`lfcc_c2`×{eeg,voice}, `mel_c1`×{eeg,voice} falharam na execução original em lote paralelo dos 300 perfis. Reexecutados individualmente em 2026-07-16 e passaram sem problemas — contenção transitória de recursos por causa dos jobs paralelos, não um defeito de código. `results/run_profiles_phase00.state` agora mostra 300/300 PASS.

## tempStrategy.tex incluído por acidente na tese compilada

`documentation/00-thesis/monography/tempStrategy.tex` (a origem das seções "Log de status do Experiment05" / "Referência --- config e grade do pipeline do Experiment05" abaixo) estava sendo incluído via `\include` como a primeiríssima coisa na seção pré-textual da tese (`monografia.tex:12`), apesar do próprio comentário de cabeçalho do arquivo afirmar "scratch, not \input anywhere". Linha `\include` removida em 2026-07-16 (contagem de páginas 118 → 111); conteúdo preservado aqui.

---

# Referência --- config e grade do pipeline do Experiment05

_Material de referência/contexto não-acionável, migrado de `tempStrategy.tex` em 2026-07-16 quando esse arquivo foi retirado da tese compilada. Ver "TODO — ativas" acima e "Log de status do Experiment05" abaixo para o que de fato está pendente._

Modalidade (`dataset.modality`):
- `voice` --- somente áudio
- `eeg` --- somente EEG
- `fused` --- voz+EEG combinados (não é um terceiro sinal gravado); sub-eixo `fusion_mode` = `early` (funde os sinais brutos antes da extração) ou `late` (concatena os vetores de características depois da extração por sinal, padrão)

Estratégia de extração de características (`feature_extraction.strategy`):
- `handcrafted` --- DTWPT + descritores
- `autoencoder` --- `snn-ae` (pulsante) e `ann-ae` (denso) implementados e integrados na Phase 00; `lstm-ae` permanece no código (artigo de Guayaquil) mas nenhum perfil da tese o utiliza

Escala handcrafted (`handcrafted.scale`, apenas se strategy=handcrafted):
- `bark`
- `mel`
- `lfcc`

Wavelet-mãe handcrafted (`handcrafted.wavelet`, apenas se strategy=handcrafted) --- varrida na Phase 00; 23 opções com propriedades dos coeficientes em `include/wavelet/Types.hpp`:
- `haar`
- `daub4`, `daub6`, `daub8`, ..., `daub46` (N par)

Descritores handcrafted (`handcrafted.descriptors`, lista, qualquer subconjunto): energy, zcr, entropy, teager, jitter, shimmer

Classificador:
- RNN (Rede Neural Residual, não-pulsante) --- somente artigo de Guayaquil
- DSNN (Rede Neural de Pulso Residual Profunda) --- somente tese

Modo de avaliação:
- text-dependent
- text-independent (modo de contribuição primário)

Esquema de CV (`training.nested_cv`):
- `true` → nested 5-fold (teste externo / seleção de modelo interna)
- `false` → flat grouped 5-fold

Chaves opcionais de regularização/normalização (independentes, todas desligadas por padrão):
- weight_decay (L2, estilo AdamW)
- firing_rate_reg_lambda (somente DSNN)
- batch_normalization = "threshold-dependent" (somente DSNN)

Perfis já publicados (modalidade × estratégia, 6 no total): handcrafted-eeg, handcrafted-voice, handcrafted-fused, autoencoder-eeg, autoencoder-voice, autoencoder-fused (+ debug, article-full)

Espaço combinatório completo (handcrafted): 3 modalidade × 3 escala = 9 combos base × classificador(2) × modo de avaliação(2) × CV(2) = 72.
Ramo autoencoder: 3 modalidade × classificador(2) × avaliação(2) × CV(2) = 24 (SNN-AE e ANN-AE implementados e integrados; lstm-ae presente no código mas não usado pelos perfis da tese).

## Grade experimental da tese --- o que de fato testar (2 fases)

_Fundamentado em chapters/08-proposedApproach.tex (seções estruturaDaEstrategiaProposta, Métricas) e chapters/06-Introduction.tex (Questão 4 --- text-dependent/independent)._

### Phase 00 --- Construção do vetor de características (melhor vetor por sinal, via EPC paraconsistente)

Objetivo: para **voz** e para **EEG** separadamente, encontrar o método de extração de características que maximiza a separabilidade paraconsistente (D_truth mínimo), antes de qualquer classificador ver os dados (ch08 §estruturaDaEstrategiaProposta).

Candidatos por sinal (voz e EEG recebem ranking próprio cada --- os vetores fundidos são construídos *depois* dessa fase, a partir do vetor vencedor de cada lado):
- **Handcrafted**, varrida sobre *wavelet-mãe* × *escala*:
  - wavelet: 23 opções (`haar` + `daub4`...`daub46`)
  - escala: bark, mel, lfcc; e `cepstral` (booleano): false = Categoria 1 (energias de banda), true = Categoria 2 (log+DCT-II → LFCC/MFCC/BFCC)

  cada uma carregando o conjunto de descritores: energy, ZCR, entropy, Teager-Kaiser, jitter, shimmer. Logo 23 × 3 = 69 combos handcrafted (wavelet × escala) por sinal, ×2 pelo `cepstral` (Categoria 1 / Categoria 2) = 138 variantes handcrafted por sinal.
- **Autoencoder** --- 12 AEs compactos por sinal: 9 SNN-AE (pulsante; 3 tamanhos × 3 codificações temporais --- poisson/latency/direct) e 3 ANN-AE (denso; 3 tamanhos), latente 8/16/32, 2:1 na camada oculta.

Logo a grade da Phase 00 = 2 sinais × (138 handcrafted + 12 autoencoder) = **300 rankings**, cada um pontuado por α, β, G1, G2, D_truth (sec:conceitos). Saída desta fase: um vetor de características vencedor para voz, um para EEG.

Os 12 autoencoders por sinal são SNN-AEs compactos de camada única (poisson/latency/direct × 3 tamanhos) e ANN-AEs (3 tamanhos), ambas as famílias integradas; as variantes handcrafted cepstrais da Categoria 2 também estão implementadas e publicadas. Os quadros de pulso do SNN-AE usam integração temporal (estado reiniciado uma vez por amostra, depois integrado ao longo de `time_steps`=16 quadros, leitura pela média do latente) e um limiar de disparo do codificador reduzido (`voltage_threshold`=0.2 para poisson/latency; o padrão LIF de 1.0 para direct, que reproduz a linha de base sem codificação). Medido em tiny/EEG: poisson α=0.069, latency α=0.258, direct α=0.875 (perto de cara-ou-coroa) --- confirmando que a codificação temporal é o que torna o SNN-AE separável.

### Phase 01 --- Autenticação (classificação DSNN, ablação biométrica)

Objetivo: alimentar **apenas a combinação vencedora única da Phase 00** (o wavelet×escala ou autoencoder escolhido pelo ranking paraconsistente) no **DSNN** (classificador residual pulsante profundo, exclusivo da tese --- o classificador RNN/ResNet convencional é exclusivo do artigo de Guayaquil, ch08 §82) e medir o desempenho de verificação biométrica sob 4 modos de fonte de sinal, para isolar a contribuição de cada fonte (ch08 §32):

- [00a] **voz + EEG, fusão precoce** (`fused-early`) --- voz+EEG brutos concatenados antes de uma única passagem de extração
- [00b] **voz + EEG, fusão tardia** (`fused-late`) --- extração por sinal, vetores de características concatenados depois (auditoria C12)
- [01] **apenas voz (fonada)** --- ablação, somente vetor de voz
- [02] **apenas EEG (imaginada)** --- ablação, somente vetor de EEG

Cada um desses 4 modos de fonte é adicionalmente cruzado com 2 eixos exigidos explicitamente pelas próprias questões de pesquisa/seção de métricas da tese:
- **text-dependent vs. text-independent** (ch06, Questão 4): mesma frase falada+imaginada vs. frase arbitrária/não coincidente entre treino e teste. Testa se o desempenho do método de extração é sensível ao texto.
- **Esquema de CV**: nested 5-fold (seleção de hiperparâmetro sem viés) vs. flat grouped 5-fold. Ambos são disjuntos por locutor (protocolo de verificação, ch08 §71) --- locutores de teste nunca aparecem no treino.

A grade publicada também cruza um terceiro eixo fora do plano original, `training.standardize_features` (bruto vs. z-score por fold, auditoria G1), logo a grade da Phase 01 = 4 modos de fonte × 2 (dependent/independent) × 2 (nested/flat) × 2 (raw/std) = **32 perfis DSNN**, todos publicados em `profiles/phase01/` (`classifier.type=dsnn`), usando a combinação vencedora única da Phase 00 (o eixo do extrator *não* é varrido na Phase 01 --- o ranking paraconsistente já o escolheu). Os perfis publicados ainda carregam um bloco `feature_extraction` provisório (`daub4/lfcc`) a ser substituído pelo vencedor real da Phase 00 via `02_e05_apply_winner.py` antes de rodar.

**Métricas primárias** (ch08 §71, protocolo de verificação): EER e AUC. Métricas de conjunto fechado (acurácia/F1/precisão/recall/especificidade) só reportadas se uma avaliação de conjunto fechado também for rodada junto com a verificação --- não é a alegação principal. MSE é apenas de reconstrução (treino do autoencoder), não é métrica de classificação.

### Observações / coisas fáceis de esquecer

- **Invariante de carregamento pareado**: o carregador sempre puxa áudio+EEG juntos por trial (via `eeg_index`); o campo `modality` só seleciona o que alimenta a extração. Isso garante que as execuções voice-only / EEG-only / fused compartilhem o *mesmo exato* conjunto de sujeitos/trials --- caso contrário a ablação de 3 modos na Phase 01 não seria uma comparação justa.
- **Regularização é um eixo de configuração de treino, não um eixo de combinação de método** --- mas precisa ficar fixa (ou ela mesma ser varrida e reportada) entre as execuções da Phase 01 para manter as comparações justas: weight decay L2 desacoplado (estilo AdamW, poupa os parâmetros biofísicos R, C, V_th) e regularização de taxa de disparo (exclusiva do DSNN, mantém a taxa de disparo em [0.05, 0.80], evita neurônios mortos/saturados).
- **Threshold-Dependent Batch Normalization (tdBN)** --- exclusiva do DSNN, estabiliza o treino de redes pulsantes profundas; também é uma configuração de treino fixa/varrida, não uma combinação de extração de características.
- **Vazamento de dados / normalização de entrada** (auditoria G1, implementada): padronização z-score por característica da entrada do classificador, estatísticas ajustadas apenas nas linhas de treino de cada fold e aplicadas a treino+teste (`E05Classifiers.cpp`, `training.standardize_features`, ligado por padrão). Evita vazamento do conjunto de teste. Corresponde ao ch07 §normalizacaoEntrada.
- **Pré-ênfase** (α=0.97) aplica-se apenas a vetores derivados de áudio, antes do cálculo de energia/espectral --- relevante para toda combinação envolvendo voz em ambas as fases (fused incluso).
- **LSTM-AE e o classificador RNN/ResNet convencional estão fora do escopo da tese** --- existem no código do Experiment05 porque os dois artigos (congresso de Guayaquil) compartilham o mesmo pipeline, mas não fazem parte da grade de combinações de nenhuma das fases acima.

---

# Log de status do Experiment05 (reverificado contra código + estado de execução, 2026-07-16)

_Status extraído de: `src/experiments/05/lib/{include,src}/E05Config.{hpp,cpp}`, `E05FeatureExtraction.cpp`, `E05Classifiers.cpp`, `profiles/*.json`, `results/run_profiles_phase00.state`, `results/phase00/`, `results/phase01/`._
`[x]` = implementado, tem perfil publicado, E foi executado (arquivos de resultado em disco). `[~]` = implementado + perfil publicado existe, mas nunca executado. `[ ]` = não implementado / rejeitado por validate().

## Phase 00 --- extração de características (300 perfis: 2 sinais × (69 handcrafted × 2 categorias + 12 autoencoder))

**Categoria 1 vs 2** (auditoria G2, implementada): `handcrafted.scale` agrupa as sub-bandas da DTWPT por frequência (linear/Bark/Mel). Com `cepstral=false` as energias por banda são usadas diretamente (Categoria 1); com `cepstral=true` um estágio log+DCT-II sobre essas energias produz os coeficientes cepstrais LFCC/MFCC/BFCC (Categoria 2). Ambas as categorias são selecionáveis e publicadas.

Voz e EEG (mesmo status, mesmo caminho de código, agnóstico ao sinal):
- [x] Energia de banda Linear/Mel/Bark (Categoria 1) --- `cepstral=false`, 23 wavelets × 3 escalas × 2 sinais = 138 perfis, todos executados
- [x] LFCC/MFCC/BFCC (Categoria 2) --- `cepstral=true`, 138 perfis, todos executados
- [x] SNN-AE --- `ProtocolSpikingAutoencoder`; 3 tamanhos × 3 codificações temporais (poisson/latency/direct) × 2 sinais = 18 perfis, todos executados
- [x] ANN-AE --- `ProtocolAutoencoder`; 3 tamanhos × 2 sinais = 6 perfis, todos executados

O próprio ranking paraconsistente (α, β, G1, G2, D_truth): **[x] implementado**, `E05Paraconsistent.cpp`, exercitado por `e05_profile_audit_gtest`.

**Phase 00 está totalmente executada**: `results/run_profiles_phase00.state` mostra **300/300 PASS** (ver "Resolvidas" acima sobre os 4 perfis daub32 que precisaram de reexecução). Todo perfil tem seu CSV/JSON de ranking paraconsistente de 3 repetições em `results/phase00/`.

**Lacuna restante**: `scripts/pipeline/e05/01_e05_phase00_rank.py` (lê os 300 resultados, escolhe o vencedor por sinal, grava `winners.json`) ainda não foi rodado contra o conjunto de resultados agora completo --- `winners.json` não existe. Os perfis da Phase 01 ainda carregam um extrator provisório `daub4/lfcc` em vez do vencedor real da Phase 00. → item 1 do TODO.

## Phase 01 --- autenticação DSNN (32 perfis: 4 modos de fonte × 2 text_mode × 2 CV × 2 standardize_features)

Todo `profiles/phase01/*.json` define `classifier.type = "dsnn"` (32 perfis; os antigos perfis `"rnn"` agora vivem apenas em `debug.json`/`smoke/debug.json` e são exclusivos do artigo de Guayaquil, fora do escopo da tese). `E05DsnnClassifier` está implementado e testado (`e05_classifiers_gtest`, 16/16 passando), e todo eixo está publicado como perfil real:

- [~] fused-early, fused-late, voice-only, eeg-only × classifier=dsnn --- 8 perfis por combinação de modo, publicados, **ainda não executados**
- [~] `text_mode`: `dependent` e `independent`, cada um pareado com dados reais (não-debug) --- publicados, **ainda não executados**
- [~] `nested_cv`: `true` (nested 5-fold) e `false` (flat grouped 5-fold) --- publicados, **ainda não executados**
- [~] `training.standardize_features`: `true` e `false` --- publicados, **ainda não executados**

**Maior lacuna restante**: `results/phase01/` não existe --- **nenhum dos 32 perfis DSNN foi executado**. O classificador primário da tese tem cobertura completa de código + perfis mas nenhum resultado. Bloqueado pela lacuna do `winners.json` acima (substituir o extrator provisório via `02_e05_apply_winner.py` antes de rodar). → itens 2--3 do TODO.

Chaves de regularização/normalização: ver item 4 do TODO acima (weight_decay / firing_rate_reg_lambda / tdBN, nenhum exercitado fora do debug/smoke).
