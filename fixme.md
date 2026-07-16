# TODO — ativas

## ⚠️ Decisões pendentes (2026-07-16) --- só o autor decide

| # | Assunto | Gravidade | Bloqueia |
|---|---|---|---|
| D1 | Critério da Fase 00 premia AE morto | 🔴 crítico | item 1 do pipeline |
| D2 | Conclusão do §08 confundida por D1 | 🔴 crítico | fechar §08 |
| D3 | `snn_lr_scale` quebrado | 🟡 médio | D4, reprodutibilidade |
| D4 | Seção da tese sobre lr por grupo | 🟢 baixo | --- (depende de D3) |
| D5 | Item 51: otimizadores | 🟢 baixo | --- |
| D6 | Eixo `scale` do EEG é inócuo | 🟡 médio | contagem da grade na tese |

---

### D1 --- O critério da Fase 00 premia autoencoder morto 🔴

Ao escolher o vencedor pelo menor $D_{\text{verdade}}$, a Fase 00 acaba selecionando, no ramo dos autoencoders, justamente a rede que menos aprendeu.

A demonstração é direta. Quando o `direct_tiny_eeg` é treinado com taxa de aprendizado efetiva de 1e-5, ele pontua α=1,0000 e β=1,0000 --- exatamente o vértice *Ambiguidade*, como definido em `Core/Paraconsistent.md:67`. Esse par de valores descreve um latente morto: α=1 significa dispersão intraclasse zero, e β=1 significa que todas as classes ocupam o mesmo ponto. Em outras palavras, toda amostra produz o mesmo vetor de características, independentemente da classe. Ainda assim, essa configuração pontua d=1,4142, o que lhe daria o primeiro lugar do EEG --- o melhor resultado real hoje é o do `ann_tiny`, com d=1,4861.

A causa está na saturação do β. Nos autoencoders ele fica preso em torno de 0,99 e, com isso, o $d$ deixa de depender de duas grandezas e passa a ser função apenas de α. Como α é máximo quando a saída é constante, minimizar $d$ equivale a premiar o colapso do latente.

O problema não atinge o ramo handcrafted, onde o β gira em torno de 0,78 e portanto ainda carrega informação. Os 276 perfis handcrafted continuam válidos.

Há quatro caminhos possíveis:

- (a) rejeitar explicitamente as configurações com α≈1 e β≈1;
- (b) redefinir α a partir da variância, em vez da amplitude (mínimo--máximo);
- (c) acrescentar uma guarda de colapso que exija variância mínima do latente;
- (d) aceitar a limitação e documentá-la.

---

### D2 --- A conclusão do §08 está confundida por D1 🔴

O item 59 corrigiu a ordem das codificações (`direct` > `latency` > `poisson`), e essa ordem descreve fielmente os dados armazenados. O problema está na interpretação que escrevi junto: atribuí o resultado ao ruído estocástico da codificação derrubando o α, o que é verdade, mas é apenas parte da história.

Como mostra D1, o ranking também mede o quanto cada codificador deixou de aprender. Já que o critério pune qualquer latente com variância, a codificação `poisson` pode simplesmente nunca ter tido um teste justo --- ela é penalizada por produzir variação, que é exatamente o que se espera de uma codificação estocástica.

A consequência é que o "resultado negativo para codificação temporal" que registrei **não está estabelecido**. O §08 precisa ser revisto, mas só depois que D1 for decidido, já que a redação correta depende de qual critério passará a valer.

---

### D3 --- `snn_lr_scale` está quebrado 🟡

O parâmetro promete uma taxa de aprendizado por grupo (menor para R, C e V_th do que para os pesos), mas na prática é um multiplicador global:

```cpp
// Trainer.hpp:100 --- preenche TODOS os params, não só R/C/V_th
std::vector<float> scales(params.size(), cfg_.snn_lr_scale);
```

Como o default é `0.1F` (`TrainerConfig.hpp:45`) e o E05 nunca o sobrescreve, as 300 execuções rodaram com taxa efetiva de **1e-4**, embora os perfis declarem **1e-3**. Ou seja, o que está publicado não é o que foi executado.

O alcance do problema é menor do que parece: só os **24 perfis de autoencoder** (6 ANN + 18 SNN) são afetados, porque o ramo handcrafted não treina rede nenhuma. O Experimento 04 escapa por passar `1.0F` explicitamente (`E04Training.cpp:148`).

São três as opções:

- (a) **Relabel honesto.** Colocar `snn_lr_scale=1.0` e `learning_rate=0.0001` nos perfis de AE. A taxa efetiva continua a mesma, então os resultados ficam idênticos e nenhuma re-execução é necessária --- muda apenas que o perfil passa a declarar a verdade. O custo é abandonar de vez a ideia de taxa por grupo.
- (b) **Implementar a taxa por grupo de verdade.** Exige mudar o contrato do `Module` para etiquetar quais parâmetros são biofísicos dentro do `params()` achatado, e depois re-rodar os 24 perfis.
- (c) **Manter 1e-3 e re-rodar.** A varredura mostra que isso destrói o `direct`, que colapsa para α=0.

---

### D4 --- Seção da tese sobre taxa de aprendizado por grupo 🟢

Essa seção foi pedida durante a sessão, mas **deixei de escrevê-la de propósito**.

O motivo é que o código não implementa taxa por grupo --- implementa uma escala global. Escrever a seção agora documentaria uma intenção que o código não honra, que é exatamente a falha corrigida nos itens 24, 57 e 59.

A decisão depende de D3. Se a escolha for a opção (a), a seção honesta passa a ser sobre *por que não* existe diferenciação entre os grupos de parâmetros.

---

### D5 --- Item 51 (otimizadores) está bloqueado 🟢

Avaliar outros otimizadores não é uma questão de gerar perfis: hoje não existe seleção de otimizador em lugar nenhum do E05. Concretamente:

- o `Trainer` fixa `Adam optimizer_` como membro concreto (`Trainer.hpp:151`);
- nem `TrainerConfig` nem `E05Config` têm campo de otimizador;
- a `OptimizerFactory` só conhece `adam` e `sgd`, e apenas o Exp03 a utiliza;
- Schedule-Free AdamW não existe no código;
- `attach_with_scales` é exclusivo do Adam --- não é virtual na classe base e o SGD não o possui.

A decisão, portanto, é se vale o trabalho de framework (tornar o otimizador polimórfico e estender a interface base para escalas por grupo) antes de qualquer ablação.

---

### D6 --- O eixo `scale` do EEG é inócuo 🟡

No EEG, as três escalas (bark, mel e lfcc) produzem d_truth **idêntico** em 44 dos 46 grupos. Na voz, todos os 46 diferem entre si.

A causa é simples: Bark e Mel são escalas perceptuais de *áudio*, e o conteúdo do EEG fica abaixo da estrutura de bandas delas, de modo que o agrupamento espectral colapsa no mesmo particionamento.

Na prática, dos 138 perfis handcrafted de EEG apenas 46 são realmente distintos --- cerca de 92 execuções são duplicatas. Isso também torna enganosa, para o EEG, a afirmação de que há "23 × 3 × 2 = 138 combinações manuais por sinal".

Duas opções: remover o eixo `scale` do EEG e corrigir a contagem na tese, ou mantê-lo e documentar a redundância.

---

## Correções mecânicas (decorrem das decisões acima)

- [ ] **F1.** O comentário em `TrainerConfig.hpp:10` afirma que `snn_lr_scale` "is ignored for pure ANN models". Isso é falso: em modelos ANN puros ele escala todos os pesos por 0,1 --- o efeito medido no ANN-AE foi de +4,6σ. Corrigir junto com D3.
- [ ] **F2.** Nenhum teste cobre o caminho em que `snn_lr_scale` é diferente de 1.0. Todos os testes de `trainer_genericity_gtest.cpp` (linhas 158, 198, 223, 243 e 276) o desligam com `= 1.0F`, e foi por isso que o bug passou despercebido. Vale acrescentar um teste que fixe o comportamento escolhido em D3.
- [ ] **F3.** O diretório `results/phase00/` não é reproduzível a partir dos perfis: quem rodá-los como estão (com 1e-3 declarado) obtém resultados diferentes dos publicados (que usaram 1e-4 efetivo). Some assim que D3 for decidido.
- [ ] **F4.** A tabela de `.wiki/Experiments/Experiment05.md` já foi corrigida no item 59, mas ainda não menciona D1 nem o fato de os números virem de uma taxa efetiva de 1e-4. Atualizar depois de D1 e D3.
- [ ] **F5.** A contagem "138 variantes handcrafted por sinal", na seção "Referência" mais abaixo, continua enganosa para o EEG. Ver D6.

---

### Dados de apoio --- varredura de lr (tiny/EEG, `d_truth`/`α`)

**1e-4 = o que de fato rodou. 1e-3 = o que os perfis declaram.**

| config | 1e-5 | 3e-5 | **1e-4** | 3e-4 | 1e-3 |
|---|---|---|---|---|---|
| `ann_tiny` | 1,540/0,532 | 1,545/0,525 | **1,486/0,643** | 1,477/0,663 | 1,571/0,498 |
| `direct_tiny` | 1,414/1,000 ☠️ | 1,509/0,625 | **1,493/0,667** | 1,668/0,375 | 1,999/0,000 |
| `latency_tiny` | 1,640/0,401 | 1,862/0,131 | **1,875/0,116** | 1,871/0,121 | 1,885/0,107 |
| `poisson_tiny` | 1,839/0,162 | 1,996/0,000 | **1,997/0,000** | 1,999/0,000 | 1,998/0,000 |

☠️ = latente morto (α=1 ∧ β=1), e mesmo assim o melhor `d` da tabela --- é a prova de D1.

**Leitura:** toda codificação SNN melhora conforme o lr cai, ou seja, conforme o latente colapsa. **Não existe lr "correto" que salve o ramo AE.** Só o `ann_tiny` tem ótimo interior (3e-4), e a ~0,5σ do valor em 1e-4.

## Pipeline Experiment05 (crítico para a tese, ordem de prioridade)

1. **[BLOQUEADO POR D1]** Rodar `01_e05_phase00_rank.py` sobre o conjunto de 300 perfis agora completo para produzir `winners.json`. --- Rodar agora escolheria o vencedor por `D_truth` mínimo, critério que (D1) premia latente colapsado no ramo AE; para o EEG o vencedor atual (`ann_tiny`, d=1,4861) é um AE, logo é justamente o ramo afetado. Decidir D1 antes.
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

- [ ] **51. Avaliar diferentes algoritmos de otimização.** → bloqueado, ver **D5**.

**Feitos:**

- [x] **24. Consistência entre as equações do LIF e o código.** Havia uma divergência real: a tese derivava um passo de Euler explícito, em que a corrente de entrada é escalada por $R$, mas o código (`Lif`, `LifBPTT` e `LifIntegrator`) usa a recorrência exata $\beta V_{mem} + I_{in}$, sem escalar a entrada por $R$. Acrescentei uma "Nota de implementação" em `07-bibliographicRevision.tex` explicando que se trata de uma simplificação deliberada, e não de um erro de discretização: ela desacopla o ganho da entrada de $\tau$, o que facilita treinar $R$, $C$ e $V_{th}$.
- [x] **25. Figura dos pulsos do neurônio LIF.** A legenda da `fig:neuronspike` agora deixa explícito que os pulsos vêm de uma simulação de neurônio RNP/LIF em resposta ao sinal ruidoso, ilustrando o disparo por limiar discutido no texto.
- [x] **42. Tabela comparativa entre métodos manuais, automatizados, escalas e wavelets.** São quatro longtables em `09-testsAndResults.tex`, geradas automaticamente a partir de `results/phase00/*_summary.json` pelo script `e05_build_phase00_paraconsistent_tables.py`.
- [x] **47. Seção específica sobre tdBN.** Criada em `07-bibliographicRevision.tex:912` e verificada contra o código. Ao contrário do item 24, aqui texto e implementação batem: `ThresholdDependentBatchNorm.hpp` implementa exatamente $Y=\gamma(\alpha V_{th}\hat X)+\beta$, com estatísticas agrupadas sobre lote e tempo e os mesmos defaults. Falta apenas a parte "ESTUDAR!!!", que é aprofundamento pessoal --- ver a seção dedicada mais abaixo.
- [x] **48. Garantir que toda variável esteja explicitamente definida (tese e Wiki).** Auditei cerca de 85 equações da tese e os 38 arquivos da Wiki que contêm matemática. Apareceram quatro problemas:
  - Na tese, a mesma grandeza aparecia com três notações diferentes ($D_{1,0}$ no cap. 07, $D_{\text{verdade}}$ no cap. 08 e $d$ no cap. 09); as três foram explicitamente ligadas em `09-testsAndResults.tex`.
  - `Membrane-Dynamics.md` repetia o mesmo erro do item 24 e contradizia o próprio trecho de código exibido na página.
  - `Core/Initializers.md` descrevia Kaiming/He como Gaussiana, usando um α que nunca era definido, quando o código na verdade é He-uniform.
  - `Experiment05.md` tinha a direção de α/D_truth invertida. Corrigi o texto na hora, mas o ranking em si só foi re-derivado depois, no item 59.
- [x] **56. Comparar arquiteturas de autoencoder via EPC.** Feito com as mesmas tabelas do item 42 (`tab:phase00ae_eeg` e `tab:phase00ae_voice`), sobre as 300 execuções completas da Fase 00. ⚠️ A leitura dessas tabelas está confundida por **D1**.
- [x] **57. Fundamentar a afirmação sobre taxas de aprendizado para parâmetros biofísicos.** Encontrei uma **citação fabricada**: "[37] Y. Cao et al., *Direct training of SNNs: Challenges and insights*, Frontiers 2025" --- o artigo simplesmente não existe, provavelmente uma alucinação de sessão anterior. Junto com isso, o `[37]` do `References.md` apontava para outro artigo, esse real, mas atribuído ao autor errado (são Hou, Wu e Zhou, não "Fang") e sobre um tema sem relação com taxas de aprendizado. Removi a citação falsa de `Core/Optimizers.md`, `Core/Training.md` e `Adam.hpp`, reformulei a afirmação como escolha empírica do projeto e corrigi a autoria. O valor `0.1` foi mantido por ser o default real do código. ⚠️ Ver **D3**: o código nem sequer implementa o que a afirmação descreve.
- [x] **58. Fundamentar a tabela de codificações × funções de perda.** Ao contrário do item 57, aqui as citações eram reais e corretas (Comşa 2021 e Manna 2024, ambas verificadas). Acrescentei em `Spike-Encoding.md` a explicação mecanicista de por que cada combinação errada quebra o treino, derivada direto do código: o `SpikeCountLoss` soma disparos ao longo do tempo, e o `SpikeTimeLoss::backward` escreve gradiente em um único índice temporal. Também alinhei uma inconsistência entre as duas tabelas da Wiki.
  Como seguimento na tese, confirmei que `SpikeCountLoss` e `SpikeTimeLoss` **não são usadas em nenhum experimento** --- aparecem só em testes unitários. O SNN-AE treina as três codificações sempre com EQM, porque o alvo da reconstrução é o sinal contínuo original, e não um trem de pulsos. Isso virou um parágrafo em `08-proposedApproach.tex`, com `comsa2021spiking` e `manna2024time` adicionadas à `bibliography.bib`.
- [x] **59. Corrigir a afirmação invertida sobre as codificações temporais (§08).** Resolve a pendência deixada no item 48. A tese se contradizia a duas linhas de distância: dizia que vence a combinação de *menor* $D_{\text{verdade}}$ e, logo acima, que a `poisson` (D=1,93) "supera" a `direct` (D=1,43). Os números vinham de uma execução preliminar de 20 épocas e 2 folds, não da grade final.
  Re-derivada contra `results/phase00/` (300 configs × 3 repetições), a ordem real é `direct` (α=0,667, D=1,493, **2ª de 150**), depois `latency` (α=0,116, D=1,875, 145ª) e por fim `poisson` (α=0,000, D=1,997, **150ª e última**) --- sem exceções nos dois sinais e nos três tamanhos. Reescrevi o §08 com a ordem real, o mecanismo e uma ressalva, registrando o achado como resultado negativo explícito, e removi a expressão "linha de base degenerada" (a `direct` não é degenerada: é a melhor). A Wiki foi sincronizada. ⚠️ A interpretação foi depois confundida por **D1** --- ver **D2**.

## ESTUDAR (aprofundamento pessoal --- não delegável, não marcar como feito por terceiros)

- [ ] 30. Seção BPTT: ESTUDAR!!!!!
- [ ] 47b. Threshold-Dependent Batch Normalization (TDBN): parte "ESTUDAR!!!" pendente --- a seção/conteúdo objetivamente verificável do item 47 já está feita e consistente com o código (ver item 47 acima), falta apenas o aprofundamento pessoal.

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

Os 12 autoencoders por sinal são SNN-AEs compactos de camada única (poisson/latency/direct × 3 tamanhos) e ANN-AEs (3 tamanhos), ambas as famílias integradas; as variantes handcrafted cepstrais da Categoria 2 também estão implementadas e publicadas. Os quadros de pulso do SNN-AE usam integração temporal (estado reiniciado uma vez por amostra, depois integrado ao longo de `time_steps`=16 quadros, leitura pela média do latente) e um limiar de disparo do codificador reduzido (`voltage_threshold`=0.2 para poisson/latency; o padrão LIF de 1.0 para direct, que reproduz a linha de base sem codificação). 
> 🚫 **OBSOLETO** (ver itens 59, D1, D2). O texto abaixo, antes aqui, estava **invertido**:
> ~~"Medido em tiny/EEG: poisson α=0.069, latency α=0.258, direct α=0.875 (perto de cara-ou-coroa) --- confirmando que a codificação temporal é o que torna o SNN-AE separável."~~
> Números de execução preliminar (20 épocas, 2 folds). Contra a grade final: `direct` (d=1,493) > `latency` (d=1,875) > `poisson` (d=1,997, último de 150) --- ou seja, o oposto. E a comparação em si é confundida por **D1**.

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
