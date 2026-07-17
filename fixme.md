# TODO — ativas

## ⚠️ Decisões pendentes (2026-07-16) --- só o autor decide

| # | Assunto | Gravidade | Bloqueia |
|---|---|---|---|
| ~~D1~~ | ~~Critério da Fase 00 premia AE morto~~ → **RESOLVIDO** (métrica `d_penalized`) | ✅ | --- |
| ~~D2~~ | ~~Conclusão do §08 confundida por D1~~ → **RESOLVIDO** (afirmação retirada; mecanismo quantificado) | ✅ | --- |
| D3 | ~~`snn_lr_scale` quebrado~~ → **CORRIGIDO no código**, também afeta o artigo E04 (Guaiaquil, rascunho) | 🟡 re-execução pendente | D4, reprodutibilidade, artigo E04 |
| ~~D4~~ | ~~Seção da tese sobre lr por grupo~~ → **ESCRITA** (§2.1.10.10) | ✅ | --- |
| D5 | ~~Item 51: otimizadores~~ → **framework FEITO** (otimizador polimórfico + escalas por grupo na base); ablação pendente | 🟡 ablação pendente | item 51 |
| ~~D6~~ | ~~Eixo `scale` do EEG é inócuo~~ → **RESOLVIDO** (removido para EEG; grade 300→208) | ✅ | --- |

---

### D1 --- O critério da Fase 00 premia autoencoder morto ✅ RESOLVIDO

**Problema (mantido para registro):** ao escolher o vencedor pelo menor $D_{\text{verdade}}$, a Fase 00 selecionava, no ramo dos autoencoders, justamente a rede que menos aprendeu. Um latente morto (α=1, β=1, o vértice *Ambiguidade*) pontuava d=1,4142 e ganharia o EEG, batendo o melhor real (`ann_tiny`, d=1,4861). A causa é a saturação do β (~0,99 nos AEs), que reduz o $d$ a função só de α, e α é máximo quando a saída é constante.

**Solução implementada (2026-07-16):** nova métrica de seleção `d_penalized`, seguindo a orientação do autor (manter o $D_{\text{verdade}}$, penalizar ambiguidade e indefinição):

$$D_{\text{penalizado}} = D_{\text{verdade}} + \lambda \cdot |G_2|, \qquad \lambda = 2 - \sqrt{2} \approx 0{,}586$$

$|G_2| = |\alpha+\beta-1|$ é o grau de contradição, cujos dois polos são exatamente os vértices *Ambiguidade* e *Indefinição*. O λ foi escolhido de forma principiada: com $\lambda = 2-\sqrt2$, os três vértices não-Verdade (*Falsidade*, *Ambiguidade*, *Indefinição*) pontuam todos exatamente 2,0, e *Verdade* pontua 0 --- ou seja, penaliza as três formas de degeneração igualmente, sem viés direcional.

**Validado** contra as 300 execuções: o latente morto cai para o último lugar (#151/151) nos dois sinais; o vencedor passa a ser handcrafted (`haar_bark_c1` no EEG, `haar_lfcc_c1` na voz). Foi testada também a fórmula literal $D_A + D_I - D_T$ (recompensar Verdade, penalizar só Ambiguidade+Indefinição) e ela **abre um exploit novo**: sem penalizar Falsidade, o otimizador foge para lá e a `poisson` (a pior, colada na Falsidade) passa a "vencer". Por isso a fórmula adotada mantém o $D_{\text{verdade}}$, que já penaliza a Falsidade.

**Importante:** `d_penalized` é função pura de (α, β), ambos já gravados em todo summary. Logo a re-classificação das 300 execuções **não exige re-execução** --- o script de ranking recalcula a métrica a partir do `g2` já gravado nos CSVs antigos.

**Arquivos alterados:** `E05Paraconsistent.{hpp,cpp}` (campo + cálculo + ordenação), `E05Output.cpp` (coluna CSV `d_penalized` + `best_d_penalized` no JSON), `01_e05_phase00_rank.py` (seleciona por `d_penalized`, com backfill para CSVs antigos), `02_e05_apply_winner.py`, `.wiki/Core/Paraconsistent.md`. Testes verdes (e05_output 5/5, e05_profile_audit 2335/2335).

**Tese (2026-07-16):** adicionada nova subseção "Vulnerabilidade de $D_{1,0}$ isolado e métrica penalizada por contradição" (`chapters/07-bibliographicRevision.tex`, §2.1.5.1, logo após o exemplo numérico original de $D_{1,0}$/plano paraconsistente). Conteúdo: (1) por que $D_{1,0}$ isolado é vulnerável --- uma saída constante do autoencoder força $\alpha=1$ e, pela definição de $\beta$, também $\beta=1$, caindo no vértice Ambiguidade com $D_{1,0}=\sqrt2\approx1{,}4142$, valor baixo o bastante para vencer extratores genuinamente separáveis; (2) o caso real observado durante os experimentos (variante do SNN-AE em lr baixo atingindo exatamente $\alpha=\beta=1$); (3) a generalização --- $\alpha=\beta$ desliza sobre o eixo que liga Ambiguidade e Indefinição, ao qual $D_{1,0}$ é cego; (4) a tentativa descartada ($D_{0,1}+D_{0,-1}-D_{1,0}$, maximizar) e por que ela abre um exploit novo em direção à Falsidade; (5) a métrica adotada, $D_{\text{penalizado}}=D_{1,0}+\lambda|G_2|$; (6) a dedução de $\lambda=2-\sqrt2$ exigindo que os três vértices não-Verdade sejam igualmente penalizados; (7) exemplos numéricos (caso degenerado e caso já-bom); (8) nota de que a métrica é recalculável sobre os 300 resultados existentes sem reexecução. Tese recompila limpo (115 pág.).

**Ainda pendente (propagação de narrativa, atada a D2):** o gerador de tabelas da tese (`e05_build_phase00_paraconsistent_tables.py`) e o texto de §08/§09 ainda refletem o `d_truth` sem `d_penalized`, e a nova subseção do §07 aponta para o cap. 4 (Testes e Resultados) para a discussão do impacto no vencedor --- discussão essa ainda não escrita, pois depende de D2. A **decisão do autor foi "implementar S2 ajustando λ primeiro"** --- λ ajustado e implementado, com fundamentação agora também na tese; falta a decisão de narrativa em D2 antes de reescrever §08/§09.

**Causa-raiz do latente morto (investigada 2026-07-16):** por que o SNN-AE morre. A arquitetura do encoder (`linear:16:leaky, linear:8:identity`) resolve para `Linear(256,16) → Lif(V_th) → Linear(16,8) → identity` --- o token `identity` é um no-op literal (`return;`), então o latente de 8 dims é a saída crua da segunda `Linear`, sem não-linearidade alguma. Com codificação `direct` (T=1) a recorrência do LIF perde o estado prévio e vira um teste de limiar puro sobre a saída da primeira `Linear`. Um teste diagnóstico temporário (revertido depois, `fundamental_mechanisms_gtest.cpp`) mediu a taxa de disparo dessa camada na inicialização He-uniform padrão: **8,3%** dos neurônios disparam (85/1024), subindo monotonicamente com a magnitude dos pesos (2×→24,5%, 5×→35,5%, 10×→37,1%, 20×→39,1%). Isso é exatamente o "No-Spike Problem" já documentado em `.wiki/Concepts/Threshold-Dependent-Batch-Normalization.md`: pré-ativação abaixo de V_th → sem disparo → gradiente substituto ≈0 → neurônio permanentemente morto. O framework já tem duas defesas contra isso (`firing_rate_reg_lambda` e tdBN), mas **nenhuma das duas estava conectada ao autoencoder** --- só ao classificador DSNN (`E05DsnnClassifier`). Um MSE puro sobre um canal de informação faminto degenera no "aprender a média": o decodificador passa a ignorar o latente e emitir a média do dataset, tornando qualquer amostra funcionalmente irrelevante para o encoder.

**Regularização de taxa de disparo no AE (implementada 2026-07-16):** por decisão do autor ("Also add spike-rate regularization to the AE now"), portei o mecanismo do classificador para o SNN-AE. Mesma matemática do `E05DsnnClassifier::add_firing_rate_grad`: penalidade de banda `reg = λ(max(0,fr_min−r)² + max(0,r−fr_max)²)`, gradiente `d_reg/d_spike = 2λ(r−clamp(r,fr_min,fr_max))/n` somado como escalar uniforme ao gradiente que chega em cada camada Lif do encoder, durante o backward. Diferença de implementação: o encoder do AE é um `Sequential` genérico (montado dinamicamente a partir de `encoder_layer_spec`), não membros nomeados como no classificador --- então as camadas Lif são localizadas uma vez, na construção, via `dynamic_cast<nn::Lif*>` sobre `Sequential::layers`, e a injeção do gradiente é feita por um laço de backward manual que replica `Sequential::backward` (`ProtocolSpikingAutoencoder.cpp`, função `backward_with_firing_rate_reg`), evitando tocar a classe `Sequential` compartilhada. Cobre também o ramo dual-branch (EEG+áudio+fusão).

Novos campos, com o mesmo default "desligado" (`0.0`) do classificador, para não alterar nenhum perfil existente: `AutoencoderConfig::firing_rate_reg_lambda/min/max` (`experiment03`), `E05Config::AutoencoderConfig::firing_rate_reg_lambda/min/max`, expostos no JSON como `feature_extraction.autoencoder.firing_rate_reg_lambda/min/max`. Validação simétrica à do `training.*` já existente.

**Validação empírica:** rodei o perfil `p00_ae_snn_direct_tiny_eeg` a uma taxa de aprendizado baixa (declarada 1e-4, efetiva 1e-5 dado o bug de D3) com e sem a regularização. Sem ela: α=0,875, β≈0,9999 (quase no vértice Ambiguidade). Com λ=5,0 e banda [0,10; 0,60]: α cai para 0,25 --- ou seja, o latente deixa de ser quase-constante e passa a variar de fato entre classes, confirmando que o gradiente de taxa de disparo está de fato alcançando e alterando o treino do encoder. `β` continua alto nesse perfil específico, o que é esperado: β mede sobreposição entre classes no sinal EEG bruto, um problema de separabilidade diferente do colapso do latente, e a regularização não pretende (nem deveria) resolvê-lo. Suite de testes inalterada com λ=0 (default): `experiment_03_autoencoder_redesign_gtest` 47/47 e `e05_profile_audit_gtest` 2335/2335 verdes.

**Arquivos alterados:** `AutoencoderConfig.hpp` (exp03), `ProtocolSpikingAutoencoder.{hpp,cpp}`, `E05Config.{hpp,cpp}`, `E05FeatureExtraction.cpp`.

**Valor recomendado aplicado a todos os perfis (2026-07-16):** por decisão do autor, `firing_rate_reg_lambda=0,5`, `firing_rate_min=0,10`, `firing_rate_max=0,80` foram escritos nos 18 perfis reais `snn-ae` da Fase 00 (e espelhados nos 18 perfis `smoke` correspondentes --- 36 arquivos ao todo). A escolha do valor: `firing_rate_max=0,80` mantém o padrão já usado pelo classificador DSNN; `firing_rate_min=0,10` fica acima da taxa de disparo nativa medida na inicialização He/Kaiming (8,3\%, ver causa-raiz acima), garantindo que a penalidade realmente entre em ação em vez de ficar inerte; `λ=0,5` foi validado empiricamente na codificação `direct` (a única testada com o tempo de treino real do perfil, ~poucos minutos): sem regularização, α=0,5; com `λ=0,5` na mesma configuração, α cai para 0,375, confirmando engajamento sem precisar do valor extremo (`λ=5`) usado só no teste de colapso forçado a lr artificialmente baixo. **Não foi possível validar o mesmo valor nas codificações `poisson`/`latency`** (T=16 passos): um teste de verificação nessas duas encodings ficou rodando por quase 2h sem terminar (nested CV de 5 folds sobre 16× mais amostras temporais que `direct`) e foi interrompido --- ficou claro que qualquer bateria de comparação nessas encodings é, ela própria, um experimento caro (na escala do guard do CLAUDE.md) e exige confirmação explícita antes de ser reexecutada.

**Suite de testes com λ>0 (2026-07-16):** adicionados dois testes a `AutoencoderRedesign_gtest.cpp`: `ProtocolSnnFiringRateRegularizationInjectsGradientWhenEnabled` (dois modelos com a mesma semente de inicialização --- logo, mesmo forward --- comparados no backward: com `λ=0` vs `λ=5` e uma faixa-alvo deliberadamente acima de qualquer taxa de disparo plausível, o gradiente de entrada diverge, confirmando que a injeção realmente acontece) e `ProtocolSnnFiringRateRegularizationInertWhenLambdaZero` (confirma que o campo tem default `0,0` e que nesse caso o modelo segue funcionando normalmente). Suite completa: 49/49 verdes (47 anteriores + 2 novos).

**Ainda pendente:** os 300 resultados armazenados da Fase 00 foram gerados **antes** desta mudança --- os CSVs/summaries dos 18 `snn-ae` já não refletem mais a configuração atual dos perfis (que agora tem `λ=0,5`). Reexecutar esses 18 perfis (3 repetições cada) para obter resultados consistentes com o `λ` recomendado é, pela evidência coletada acima, uma operação cara (as variantes `poisson`/`latency` sozinhas passam de 2h cada) --- **requer confirmação explícita do autor antes de ser disparada**, por força do guard de experimento caro do `CLAUDE.md`.

---

### D2 --- A conclusão do §08 sobre codificação temporal ✅ RESOLVIDO (2026-07-16)

**Decisão do autor:** adiar todas as reexecuções → resta a opção (a): reescrever o §08 reconhecendo o viés, sem testar $T$ maior. Feito.

**Reavaliação (registro):** D1 e D2 são **defeitos independentes do mesmo critério geométrico**, não duas faces do mesmo. D1: o critério **recompensa** indevidamente variância nula (saída constante → vértice Ambiguidade → pontua bem). D2: o critério pode **punir** indevidamente variância *real* que não decorre de pior aprendizado. Recomputei `d_penalized` sobre os três encodings a partir dos CSVs armazenados: a ordem **não muda** (`direct` 1,8764 < `latency` 1,9352 < `poisson` 1,9988) --- esperado, já que a penalidade em |G₂| é quase nula longe dos vértices degenerados. **Corrigir D1 não corrige D2.**

**O mecanismo (teoria, não medição --- é o que sobrevive à remoção dos dados):** cada codificação injeta, **antes de qualquer aprendizado**, uma quantidade de ruído calculável em fechado.
- `poisson`: sorteio de Bernoulli **novo e independente a cada passo** → a média sobre $T$ quadros tem variância $v(1-v)/T$. Com $T=16$, $v=0{,}5$: $\sigma = \sqrt{0{,}25/16} = \mathbf{0{,}125}$, ou seja **12,5% da faixa total [0,1]** da característica, por amostra, presente ainda que o codificador fosse perfeito.
- `latency`: **determinística** --- o mesmo $v$ sempre dá o mesmo instante de disparo; erro só de quantização, $\le \tfrac{1}{2}\cdot\tfrac{1}{T-1} \approx 0{,}033$ (≈4× menor) e idêntico entre amostras de mesmo valor.
- `direct`: ruído zero.

Como α é definido pela **amplitude** (mín--máx) intraclasse, ele é maximamente sensível a extremos: esse piso de ruído mapeia-se monotonicamente em α e **prevê a ordem observada** como artefato conjunto da métrica e do piso do codificador --- não como evidência de que códigos temporais carreguem menos informação de locutor.

**Conclusão:** a frase "resultado negativo para codificação temporal" **não está estabelecida**, e não por incerteza: α é *estruturalmente incapaz* de distinguir "o codificador não aprendeu" de "esta codificação tem piso estatístico irredutível em T=16". Como o piso cai com $1/T$, é testável (T=64 → variância ÷4 → σ=0,0625). **Esse teste não foi executado** (reexecuções adiadas).

**§08 reescrito:** as medições foram **retiradas, não corrigidas** --- vinham das execuções sob o bug de D3 (lr efetivo 1e-4 nos pesos) e os dados-fonte foram apagados, logo não há número a reportar; reapresentá-los com ressalva preservaria 3 casas decimais sobre execução inexistente (foi assim que uma tabela preliminar de 20 épocas/2 folds chegou ao documento com a ordem invertida). O texto agora traz: a retirada e seus dois motivos, o mecanismo quantificado (nova `eq:ruidoPoisson`), a incapacidade estrutural de α, a testabilidade via $T$, e a **independência explícita** entre D1 e D2. Também corrigido "150 configurações" (o EEG tem 58 após D6). Tese compila limpo (121 pág.).

---

### D3 --- `snn_lr_scale` está quebrado 🟡 --- **CORRIGIDO (2026-07-16), re-execução pendente**

O parâmetro promete uma taxa de aprendizado por grupo (menor para R, C e V_th do que para os pesos), mas na prática era um multiplicador global:

```cpp
// Trainer.hpp:100 (antes da correção) --- preenche TODOS os params, não só R/C/V_th
std::vector<float> scales(params.size(), cfg_.snn_lr_scale);
```

**Escopo real, corrigido:** o texto anterior desta seção dizia que só os 24 perfis de autoencoder do E05 eram afetados e que "o Experimento 04 escapa". **A segunda afirmação está errada.** `make_trainer_config` (E04Training.cpp) tem dois pontos de entrada: o ramo LSTM chama com o default `1.0F` (inofensivo), mas o ramo SNN chama com `0.1F`, e esse valor é **sobrescrito** sempre que o perfil define `learning_rate_biophysical`:

```cpp
tcfg.snn_lr_scale = (cfg.training.learning_rate_biophysical > 0.0f)
    ? cfg.training.learning_rate_biophysical / cfg.training.learning_rate
    : snn_lr_scale;
```

E os três perfis usados no **artigo da conferência de Guaiaquil** (`article-snn-dense.json`, `article-snn-conv1d.json`, `article-snn-recurrent.json`) **definem** `learning_rate_biophysical=0.0001` com `learning_rate=0.001` --- exatamente a mesma razão 0,1 do bug do E05. Ou seja, o artigo (ainda rascunho, confirmado com o autor) comparou LSTM-AE treinado a 1e-3 contra SNN-dense/conv1d/recurrent com **todos os pesos** treinados a 1e-4, não só R/C/V_th --- uma comparação estruturalmente injusta contra a própria contribuição do artigo. Isso é mais sério do que o problema original do E05: ali são só 24 perfis de uma tese ainda em andamento; aqui é uma tabela de comparação de um artigo já escrito.

**Escopo completo, agora confirmado:** os 24 perfis de autoencoder do E05 (6 ANN + 18 SNN --- o ramo ANN também sofre, pois usa o mesmo `Trainer` com o mesmo default; só o handcrafted escapa por não treinar rede nenhuma) **e** os 3 perfis `article-snn-*` do E04. O DSNN do E05 (`E05Classifiers.cpp`, Fase 01) usa a mesma `TrainerConfig` default (nunca sobrescreve `snn_lr_scale`) e portanto está igualmente exposto --- mas a Fase 01 ainda não produziu nenhum resultado real (perfis são placeholder, bloqueados até a Fase 00 escolher o vencedor), então não há números publicados para corrigir ali, só o comportamento futuro.

**Correção implementada:** em vez de etiquetar parâmetros biofísicos no contrato do `Module` (opção (b) do texto anterior, que exigiria mudar a assinatura de `params()` em toda a base de código), a correção aproveita um fato já verdadeiro hoje: R, C e V_th são **sempre** tensores escalares 1×1 (`LifImpl`/`LifBPTTImpl`), enquanto pesos e vieses nunca são --- o mesmo critério de tamanho que `Adam`'s `weight_decay` já usa para excluir vieses/biofísicos do decaimento (`param.rows() > 1 && param.cols() > 1`). `Trainer.hpp`'s construtor agora monta o vetor de escalas por parâmetro:

```cpp
std::vector<float> scales(params.size(), 1.0F);
for (std::size_t i = 0; i < params.size(); ++i)
    if (params[i]->size() == 1) scales[i] = cfg_.snn_lr_scale;
```

Sem mudança de interface, sem tocar `Module`. **Teste de regressão** adicionado (`trainer_genericity_gtest.cpp`, `SnnLrScaleOnlyAppliesToSizeOneParams`): um modelo com um parâmetro 1×1 e um parâmetro 2×2, ambos recebendo o mesmo gradiente fixo, confirma que após um passo do Adam o parâmetro 1×1 se move por `lr·snn_lr_scale` e o 2×2 se move pelo `lr` completo --- exatamente o que o bug violava. Suites verificadas sem regressão: `trainer_genericity_gtest` (6/6), `trainer_gtest` (13/13), `optimizers_gtest` (27/27), `experiment_03_autoencoder_redesign_gtest` (49/49), `e05_profile_audit_gtest` (2335/2335).

**Pendente --- decisão de re-execução (dois lotes independentes, ambos caros):**

- **E04 (prioridade alta --- artigo ainda em rascunho):** re-rodar `article-snn-dense.json`, `article-snn-conv1d.json`, `article-snn-recurrent.json` (`01_e04_run_article_profiles.sh`, ~2,5h documentado no CLAUDE.md) e reconstruir `paper_*.csv`/as tabelas do artigo antes de considerar o rascunho pronto para submissão. Sem isso, a tabela atual do artigo compara SNN artificialmente lento contra LSTM em velocidade plena.
- **E05 (prioridade mais baixa --- tese ainda em Fase 00):** re-rodar os 24 perfis de autoencoder da Fase 00 (6 ANN + 18 SNN) e re-rankear com `d_penalized`. Mesma ressalva de custo já registrada em D1/D2 (`poisson`/`latency` a T=16 já passam de 2h cada nos testes desta sessão).

Nenhum dos dois lotes foi disparado --- ambos exigem confirmação explícita do autor antes de rodar, por serem execuções longas (guard do `CLAUDE.md`).

---

### D4 --- Seção da tese sobre taxa de aprendizado por grupo ✅ ESCRITA (2026-07-16)

Essa seção tinha sido pedida durante a sessão, mas eu tinha **deixado de escrevê-la de propósito**: o código, até D3 ser corrigido, não implementava taxa por grupo --- implementava uma escala global disfarçada. Escrever a seção antes disso documentaria uma intenção que o código não honrava.

**Texto escrito** em `chapters/07-bibliographicRevision.tex`, nova subseção "Taxa de aprendizado por grupo de parâmetros" (§2.1.10.10, `\label{sec:snnLrScale}`, logo após "Inicialização de Pesos" e antes de "Autoencoders"). Conteúdo: (1) motivação --- por que R/C/V_th precisam de uma taxa menor que pesos (τ=R·C e o limiar de disparo são sensíveis a atualizações grandes); (2) registro explícito de honestidade de que o fator 0,1 é escolha de engenharia do projeto, não valor de literatura, e que uma citação anterior para essa afirmação foi removida por não ser verificável (mesmo padrão do item 57); (3) o mecanismo real do Adam (`attach_with_scales`, $\eta_i=\eta_{\text{global}}\times s_i$); (4) o defeito original (preenchimento uniforme do vetor de escalas) e como foi descoberto --- durante a investigação do latente colapsado (cross-referenciado com `\autoref{sec:dPenalizado}`); (5) o alcance real do defeito, incluindo o artigo de Guaiaquil (SNN treinado a 1/10 da taxa da LSTM de comparação, uma desvantagem não intencional); (6) a correção pelo critério de tamanho (parâmetro 1×1 = biofísico), com a mesma fórmula usada no código; (7) o teste de regressão adicionado; (8) situação em aberto --- nem a Fase 00 nem o artigo foram reexecutados sob a correção. A referência cruzada em `sec:dPenalizado` ("a causa raiz... é discutida à parte") foi convertida num `\autoref` real para esta seção. Tese recompila limpa (120 pág., zero refs indefinidas), verificada visualmente.

---

### D5 --- Item 51 (otimizadores) 🟢 --- **TRABALHO DE FRAMEWORK FEITO (2026-07-16)**, ablação ainda pendente

O bloqueio original: avaliar outros otimizadores não era uma questão de gerar perfis --- não existia seleção de otimizador em lugar nenhum do E05. O `Trainer` fixava `Adam optimizer_` como membro concreto, nem `TrainerConfig` nem `E05Config` tinham campo de otimizador, e `attach_with_scales` era exclusivo do Adam (não-virtual, ausente do SGD).

**Decisão do autor:** "sim, tornar o otimizador polimórfico e estender a interface base para escalas por grupo". Implementado:

**1. Escalas por grupo na classe base.** `attach_with_scales()` passou a ser `virtual` em `Optimizer`, com implementação padrão que chama o `attach()` virtual (preservando a alocação de estado por parâmetro de cada otimizador --- momentos do Adam, velocidade do SGD) e depois guarda as escalas. O armazenamento `lr_scales_` e o campo `weight_decay` subiram para a base, pois um chamador que segura um `Optimizer&` precisa configurá-los sem conhecer o tipo concreto. **Todos os três otimizadores** (`Adam`, `SGD`, `SGDMinimal`) agora leem `lr_scales_` no `step()` (`lr_i = learning_rate * lr_scales_[i]`, 1,0 além do fim do vetor) e aplicam `weight_decay` desacoplado com a mesma restrição a matrizes 2-D. Antes, só o Adam honrava escalas: `SGD` e `SGDMinimal` ignorariam silenciosamente qualquer escala passada --- exatamente a classe de bug que era D3.

**2. `Trainer` polimórfico.** O membro virou `std::unique_ptr<::Optimizer> optimizer_`, construído por `OptimizerFactory` a partir de dois campos novos em `TrainerConfig`: `optimizer_type` (`"adam"` default --- comportamento idêntico ao anterior para todo chamador/perfil existente --- ou `"sgd"`) e `optimizer_momentum`. Tipo desconhecido lança em vez de cair silenciosamente no Adam.

**3. Bug latente encontrado e corrigido de passagem.** O `Optimizer::attach()` da base era um no-op literal, embora o próprio comentário dissesse "concrete optimizers should call `Optimizer::attach(params)` to preserve this storage for no-arg convenience methods" --- não preservava nada. O Adam contornava atribuindo `attached_params_` à mão; o **`SGD` confiava no contrato e portanto nunca populava `attached_params_`, de modo que `sgd.step()` sem argumentos sempre lançava** mesmo após um `attach()` correto. A base agora honra o contrato que ela mesma documenta. O teste `OptimizerBaseTest.ConvenienceMethodsAndDefaults` afirmava o comportamento antigo (`EXPECT_TRUE(opt.attached_params_.empty())` sob o comentário "Base attach() is a no-op") e foi atualizado para o contrato novo.

**Testes** (3 novos + 1 atualizado; 53/53 verdes no `ctest -R "Optim|Trainer|Adam|SGD|StateIO"`, incluindo os dois testes reais de convergência):
- `OptimizerBaseTest.AttachWithScalesIsPolymorphic` --- dois parâmetros com gradiente fixo idêntico e escalas 1,0 vs 0,1; verifica que os deslocamentos diferem pela razão das escalas para SGD, para Adam, e através de um `Optimizer&` vindo da factory (o caso de uso do `Trainer`).
- `SGDTest.DecoupledWeightDecayOnly2DWeights` --- com gradiente zero, só o parâmetro 2×2 decai; o 1×1 (biofísico) fica intacto.
- `TrainerGenericity.OptimizerTypeSelectsImplementation` --- default é `"adam"`, ambos os tipos treinam, tipo inexistente lança.
- `OptimizerBaseTest.ConvenienceMethodsAndDefaults` --- atualizado ao novo contrato do `attach()`.

**Arquivos:** `Optimizer.hpp`, `Adam.hpp` (removidos os membros agora duplicados e o `attach_with_scales` próprio --- a base faz o mesmo), `SGD.hpp`, `SGDMinimal.hpp`, `Trainer.hpp`, `TrainerConfig.hpp`, `optimizers_gtest.cpp`, `trainer_genericity_gtest.cpp`. Docs: `.wiki/Core/Optimizers.md`, `.wiki/Core/Training.md`, `CLAUDE.md` (invariante 4 do SNN), + 4 páginas da wiki que citavam `Adam::attach_with_scales()`.

**Otimizadores SOTA + plumbing + ground truth (2026-07-16):** por decisão do autor ("expose optimizer_type to profile JSON; implement SOTA optimizers; create ground truth comparing with pytorch/snntorch"), com o objetivo declarado de **ablação da tese**.

**1. `optimizer_type` exposto no JSON.** `E05Config::Training::{optimizer_type, optimizer_momentum}` → validados em `validate()` → encaminhados a `TrainerConfig` nos **três** pontos onde o E05 constrói um `Trainer` (classificador em `E05Classifiers.cpp`; os dois AEs em `E05FeatureExtraction.cpp`). Chave JSON: `training.optimizer_type`. Default `"adam"` reproduz tudo que já foi publicado. Verificado ponta a ponta: `"lion"` parseia, `"nope"` lança. Novo teste `E05ProfileAuditTest.OptimizerTypeIsSupported` roda sobre os 333 perfis (suite: 2335 → 2668 testes).

**2. Implementados: Lion e Schedule-Free AdamW.** Todos portados **lendo o código-fonte das implementações de referência** (baixadas via pip), não de memória --- decisão deliberada dada a precedência do item 57 (citação fabricada). Isso pegou detalhes que uma porta "de cabeça" erraria: a Lion aplica o decay **antes** do update e avança o momentum **depois** dele; a Schedule-Free mantém três sequências acopladas (x/y/z) e **avalia num ponto diferente do que treina**. Registrados na `OptimizerFactory` (tokens: `adam`, `sgd`, `lion`, `schedule-free-adamw`); `Optimizer::train_mode()` + o RAII `OptimizerEvalScope` foram adicionados para que o `Trainer` valide no iterado médio `x` da Schedule-Free (no-op para os demais).

**3. Descartados, com motivo técnico concreto:**
- **Muon** --- foi implementada e validada contra o `muon-optimizer` (parity verde nas duas orientações de matriz), e depois **removida a pedido do autor**. Motivo prático: ela ortogonaliza só matrizes 2-D de verdade (`rows>1 && cols>1`) e cai num fallback Adam para o resto --- como R, C e V_th são tensores 1×1 aqui, a Muon **nunca os tocaria**, afetando apenas as matrizes das Lineares. Somado ao alvo de projeto dela (pré-treino em escala de LLM), não havia papel plausível na ablação da tese. Recuperável no histórico do git se um dia fizer falta.
- **Sophia** --- incompatível com o contrato atual `Optimizer::step(span<Tensor*>)`. Li a referência (`sophia-opt`): o Hessiano vem de um `update_hessian()` separado que precisa dos **gradientes de um segundo backward com rótulos reamostrados da distribuição de saída do modelo** (estimador Gauss-Newton-Bartlett). Nosso `step()` recebe só params + `.grad()` --- sem modelo, sem loss, sem como disparar esse passe. Implementá-la sem isso faria o "hessian" virar o gradiente² da loss real, que é ≈ o segundo momento do Adam com um rótulo "Sophia" --- mentira silenciosa. Exigiria mudança arquitetural no `Trainer`/loss.
- **SOAP** --- exige decomposição em autovalores (`eigh` de `GGᵀ`/`GᵀG`), que não existe na interface `Tensor` agnóstica de backend. Adicioná-la significaria estender o `TensorBackendParityContract` nos **quatro** backends (XTensor, OpenCL, SYCL, Device). O `xtensor-blas` cobriria só o XTensor.

**4. Ground truth contra as referências reais.** `scripts/testing/gen_optimizer_refs.py` (segue o padrão já existente do `gen_pytorch_refs.py`: gera `.npz` commitado, CI não precisa de torch) dirige **a implementação de referência** por uma sequência fixa de params/grads e grava o parâmetro após cada passo; `optimizer_parity_gtest` (13 testes, todos verdes) replica os mesmos dados pela nossa porta. Referências: `torch.optim.{Adam,AdamW,SGD}` + `lion-pytorch`, `schedulefree` (ambos no pypi). Cobre também o swap train/eval da Schedule-Free. **10 testes, todos verdes.**

> **⚠️ Aviso de escala, para a ablação.** Os defaults de lr das referências divergem (Adam 1e-3, Lion 1e-4, Schedule-Free 2.5e-3), e o update da Lion é `±lr` em **toda** coordenada, independente da magnitude do gradiente --- por isso o lr utilizável dela é bem menor. **Um lr único não é comparável entre otimizadores**: a ablação precisa tunar lr por otimizador, senão mede a escolha de lr e não o otimizador.

**Dois bugs reais encontrados pelo ground truth (o motivo de ele existir):**
- **Ordem do weight decay do Adam estava errada.** Aplicávamos o decay *depois* do passo de gradiente; Loshchilov & Hutter (ICLR 2019) o definem sobre θ_{t-1}, e o `torch.optim.AdamW` faz nessa ordem. O nosso deixava um erro sistemático de `lr²·wd·u`. Confirmado numericamente: decay-antes reproduz o torch com erro **0.0**; decay-depois erra 1e-5. Corrigido no Adam e, por consistência, no SGD e no SGDMinimal (nesses dois não havia ground truth: o `weight_decay` do `torch.optim.SGD` é L2 acoplado, não desacoplado).
- **Armadilha de valor-semântica do `nn::Tensor`.** Ao mover o decay para antes do update, o Adam parou de atualizar: atribuir a `param` **substitui o storage e descarta o buffer de gradiente** (armadilha já documentada no `SGDMinimal.hpp`), então `param.grad()` depois do decay lia gradiente vazio. Corrigido salvando/restaurando o gradiente em volta do decay.
- (E um bug no próprio gerador de fixtures: `np.asarray()` sobre um tensor torch compartilha storage, e os otimizadores mutam in-place --- sem `copy=True` todos os passos gravados viravam alias do último. Pego porque Adam/SGD, que são corretos e antigos, falharam junto com os novos: 13/13 falhando é sintoma de harness, não de 13 algoritmos errados.)

**Arquivos:** `Lion.hpp`, `ScheduleFreeAdamW.hpp` (novos), `Optimizer.hpp` (`train_mode`, `OptimizerEvalScope`), `Adam.hpp`, `SGD.hpp`, `SGDMinimal.hpp`, `OptimizerFactory.hpp`, `Trainer.hpp`, `TrainerConfig.hpp`, `E05Config.{hpp,cpp}`, `E05Classifiers.cpp`, `E05FeatureExtraction.cpp`, `scripts/testing/gen_optimizer_refs.py` (novo), `src/core/optimizers/tests/{optimizer_parity_gtest.cpp,fixtures/optimizer_refs.npz}` (novos), `.gitignore` (whitelist da fixture).

**5. Cada perfil recebe o lr do SEU otimizador (2026-07-16).** Pedido do autor ("make sure that every profile gets its respective lr"). Diagnóstico: os 333 perfis hoje são todos `adam` + `lr=0.001` --- que **é** o default de referência do Adam, logo nada está errado *agora*. O risco é estrutural: no momento em que um perfil de ablação puser `optimizer_type: lion` e deixar `lr: 0.001`, a Lion treina 10× quente e a conclusão seria "Lion é pior" --- um achado falso do mesmo tipo de D3. Implementado para que a garantia valha **por construção**:
- `nn::optimizers::reference_learning_rate(token)` --- fonte única de verdade com o default publicado de cada otimizador (adam 1e-3, sgd 1e-2, lion 1e-4, schedule-free-adamw 2.5e-3).
- `training.learning_rate` virou **opcional** (`std::optional`) no perfil: omitido → `Training::effective_learning_rate()` resolve pelo otimizador escolhido; declarado → vence (varredura de lr continua possível). Os três pontos que constroem `TrainerConfig` passaram a usar o accessor, então ninguém lê um `nullopt` nem cravaria 1e-3 na mão.
- **O summary agora grava o que de fato rodou**: bloco `training` com `optimizer_type`, o `learning_rate` **resolvido** e um `learning_rate_source` (`profile` | `optimizer_default`), além de epochs/batch/weight_decay. Antes o summary **não gravava nenhum parâmetro de treino** --- exatamente a lacuna que tornou D3 possível (o publicado não era o executado, e nada em disco registrava a diferença).
- Testes: `E05OptimizerLearningRate.EachOptimizerResolvesToItsOwnReferenceLr` (cada otimizador resolve ao seu próprio lr, e os quatro defaults são distintos entre si --- senão o mecanismo seria inócuo), `...ExplicitProfileValueOverridesTheDefault`, e uma asserção por perfil no `e05_profile_audit_gtest` (2668 → 2670 testes): quem declara lr resolve exatamente para o declarado (**guarda de regressão: tornar o campo opcional não pode ter mudado nenhuma execução já publicada**), quem omite resolve para o default do próprio otimizador. Verificado ponta a ponta: perfil `lion` sem lr → summary grava `1e-4` / `optimizer_default`; perfil existente → `0.001` / `profile`, inalterado.

**Ainda pendente (o item 51 em si):** **nenhuma ablação foi rodada** --- nenhum perfil real declara `optimizer_type` (todos usam o default `adam`). Rodar a ablação exige (a) decidir se basta o lr de referência por otimizador (agora automático) ou se cada um terá varredura própria de lr, e (b) tempo de máquina: é experimento caro, mesma ressalva de D1/D2/D3.

---

### D6 --- O eixo `scale` do EEG é inócuo ✅ RESOLVIDO (2026-07-16, opção A)

**Fato confirmado, e mais forte do que o texto original dizia.** Verifiquei contra os 300 resultados armazenados em vez de confiar na anotação: no EEG, as três escalas dão `d_truth` idêntico **bit a bit** em **46/46** grupos wavelet×categoria --- não 44/46. As 2 exceções aparentes eram todas `daub32`, com diferença de ~1,5e-6, isto é **ruído de ponto flutuante** da reexecução individual daqueles perfis (documentada na seção "Resolvidas" acima), não um efeito de escala. Na voz, os 138 grupos diferem, como dizia.

**A causa registrada estava ERRADA.** O texto anterior dizia que "o conteúdo do EEG fica abaixo da estrutura de bandas delas, de modo que o agrupamento espectral colapsa no mesmo particionamento". Duas coisas erradas:
1. Não há estrutura absoluta para ficar "abaixo": `group_by_scale()` normaliza pelo Nyquist **do próprio sinal** (`max_sv = hz_to_bark(nyquist)`), então a curva é esticada para caber em qualquer taxa de amostragem.
2. Nada "colapsa" no EEG --- ocorre o **oposto**. As 16 sub-bandas do EEG caem em **16 bins distintos** (nada funde), e é exatamente por isso que o agrupamento degenera no `lfcc` (um grupo por sub-banda). Quem funde é a **voz** (bark→9 grupos, mel→11).

**A causa real:** Bark é uma escala **absoluta** --- ~24 Barks cobrem a faixa audível, e `n_bands=24`. Para a voz o fator de normalização é **0,97 ≈ no-op** (o bin *é* o número de Bark). Para o EEG (Nyquist 512 Hz) o fator é **4,96**: a curva é esticada 5× para preencher os 24 bins. Como Bark/Mel são ~lineares nessa faixa, o mapeamento vira injetor → 1:1 → idêntico ao linear. **"Bark" no EEG nunca foi Bark**: era uma pseudo-escala reescalada linearmente. Ou seja, a normalização foi desenhada para áudio (onde é inócua) e se comporta mal em qualquer outra taxa de amostragem.

**Isso já contaminava um resultado relatado.** O vencedor da Fase 00 no EEG era `haar_bark_c1` (d_penalized=1,58826668) --- mas `haar/lfcc/c1` e `haar/mel/c1` têm **exatamente o mesmo valor**. Os três primeiros eram um empate perfeito, e "bark venceu" era puro artefato da ordem de desempate. Escrever "a escala Bark venceu para o EEG" seria uma afirmação falsa: a Bark não fez efeito nenhum e o vetor vencedor é literalmente o linear.

**Decisão do autor: opção A --- remover o eixo `scale` do EEG**, justificada por princípio (escalas cocleares não têm base fisiológica para EEG) e não apenas por redundância. Implementado:
- **Perfis:** removidos os 92 perfis `p00_hc_*_{bark,mel}_*_eeg.json` da phase00 + os 92 espelhos de smoke (184 arquivos, **rastreados no git** --- diferente dos perfis de AE, que são untracked por `*.json` no .gitignore). Grade: phase00 **300 → 208** (46 hc eeg + 138 hc voz + 24 AE). **Nenhuma reexecução necessária**: os resultados `lfcc` do EEG já *são* a resposta; os 92 resultados bark/mel em `results/phase00/` ficam órfãos mas **não foram apagados** (são dados de execução longa).
- **Guarda estrutural:** `E05Config::validate()` rejeita `handcrafted.scale != "lfcc"` quando `modality=eeg`, com a justificativa no comentário. `fused` é deliberadamente não restrito (a metade de voz usa bark/mel legitimamente). Teste: `E05EegScaleAxis.BarkAndMelAreRejectedForEeg`.
- **Ranking expõe empates** (decisão do autor): `01_e05_phase00_rank.py` detecta empates exatos (tolerância 1e-5, que absorve o ruído do daub32) e grava `tie_count`/`tied_with` no `winners.json`, além de imprimir um aviso `[TIE]` explícito. Validado reconstruindo o estado pré-D6 a partir do git: o detector marca corretamente o empate bark/lfcc e avisa para não atribuir o resultado à "bark". Depois da remoção dos perfis degenerados, **o empate do EEG deixa de existir** (vencedor único `handcrafted/haar/lfcc/c1`).
- **Rótulo do extrator agora é único:** `extractor_label()` omitia a categoria cepstral, então `haar/lfcc/c1` e `haar/lfcc/c2` imprimiam ambos como `handcrafted/haar/lfcc` --- duas linhas do ranking pareciam duplicadas com notas diferentes. Passou a incluir `c1`/`c2` e, nos autoencoders, a codificação e o tamanho latente.
- **Tese:** 4 afirmações de contagem corrigidas no §08 (138/sinal → 138 voz + 46 EEG; 300 → 208 execuções; 300 → 208 perfis) e nova subseção §2.1.3.x "Aplicabilidade das escalas Bark e Mel ao EEG" (`\label{sec:escalaEeg}`) com o argumento fisiológico, o mecanismo da normalização (fatores 0,97 vs 4,96), a verificação empírica 46/46 e a nota sobre o falso "Bark venceu". Compila limpo (121 pág.).
- **Wiki:** `Experiment05.md` --- contagens e a nota sobre o eixo ser voice-only.
- **Testes:** lista fixa de perfis do `e05_profile_audit_gtest` limpa (92 entradas removidas; 0 referências pendentes a arquivos apagados); suite 1935 verdes.

- **Gerador de tabelas da tese:** `e05_build_phase00_paraconsistent_tables.py` monta as tabelas a partir de `results/`, **não** do conjunto de perfis --- então os 276 arquivos de resultado órfãos (92 perfis × 3 reps) do EEG bark/mel continuariam emitindo linhas duplicadas na tese mesmo com os perfis apagados. Adicionado filtro explícito (imprime `[info] skipped 276 retired EEG bark/mel run(s)`, não silencioso). Verificado: tabela hc do EEG 138 → **46 linhas, só LFCC**; a da voz permanece **138 (46 bark + 46 mel + 46 lfcc)**.

**Resultados da Fase 00 apagados (2026-07-16, decisão do autor: "remove All phase00 results and postpone the rerun for now").** Removidos os 1800 arquivos (900 CSVs + 900 summaries, 7,1 MB) e zerado o `results/run_profiles_phase00.state` (dizia 300/300 PASS, desatualizado desde que a grade virou 208). Deixado um `results/phase00/README.md` explicando por que estão vazios e como regenerar.

Não é trabalho perdido: os resultados já estavam invalidados por três correções do mesmo dia --- **D3** (os 24 perfis de AE treinaram todos os pesos a 1e-4 declarando 1e-3), **D1** (os 18 `snn-ae` agora declaram `firing_rate_reg_lambda=0,5`, que não existia quando rodaram) e **D6** (os 276 arquivos do EEG bark/mel ficaram sem perfil que os gerasse). Some-se a correção da ordem do weight decay do Adam, que afeta qualquer modelo treinado.

Registro honesto: os **184 perfis handcrafted não estavam numericamente obsoletos** --- extração manual não treina nada, então uma reexecução os reproduz bit a bit. Foram apagados junto assim mesmo, para que a fase inteira seja regenerada como um conjunto único e coerente cujos summaries carreguem o novo bloco `training` (otimizador + lr resolvido + origem do lr), que os antigos não tinham --- exatamente a lacuna de proveniência que tornou D3 possível.

**Ordem importava:** as tabelas da tese são geradas a partir de `results/`, não dos perfis, e as versões commitadas ainda tinham **136 linhas com bark/mel no EEG** --- contradizendo diretamente o §08 recém-corrigido (46, só lfcc). Regenerei as 4 tabelas **antes** de apagar os dados: `phase00_hc_eeg` 136 → **46 linhas, só LFCC**; `phase00_hc_voice` **138** (46 bark + 46 mel + 46 lfcc). A tese continua compilando e seus números handcrafted são válidos; as linhas de AE são provisórias até a reexecução.

Recuperabilidade: os 900 CSVs eram **rastreados** e estão no histórico do git; os 900 `*_summary.json` eram gitignored e só voltam com a reexecução (que está adiada por decisão do autor).

**Pendente:** o F5 (contagem "138 variantes handcrafted por sinal" na seção "Referência" abaixo) --- corrigido junto, ver F5.

---

## Testes de paridade de micro-redes vs PyTorch/snnTorch (2026-07-16)

Pedido do autor: "create tests that implement micro ann, snn and lstm networks and compare it with the ground truth pytorch/snntorch versions so its guaranteed that ours networks works the way it should" --- antes das reexecuções.

Já existia paridade **por camada** (`pytorch_parity_gtest` + `gen_pytorch_refs.py`). O que faltava era paridade da **rede inteira**: toda camada pode estar certa e a rede montada com elas estar errada (encadeamento de gradiente, estado vazando entre sequências, ordem de gates só visível na composição). Novos: `scripts/testing/gen_micro_network_refs.py` → `micro_network_refs.npz` (commitado, CI não precisa de torch) e `micro_network_parity_gtest` (**8 testes, todos verdes**), seguindo o padrão já estabelecido.

**Três achados reais --- o teste pagou por si antes de rodar experimento nenhum:**

1. 🔴 **`MSELoss`/`MAELoss` grampeavam o próprio gradiente, sempre, em norma 1,0** (`kMaxGradientNorm = 1.0F`, fixo, não-configurável). Confirmado exatamente: a norma verdadeira na fixture era `2,132514` e a nossa saía menor por exatamente esse fator. Gravidade: (a) `MSELossImpl` é o **LossType default do `Trainer`** --- atingia **todo autoencoder treinado no projeto**, incluindo os 24 perfis AE da Fase 00 e os modelos do artigo de Guaiaquil; (b) **contradizia a configuração**: `TrainerConfig::grad_clip_norm` tem default `0.0F` = "sem grampeamento", o `Trainer` respeita isso, e o `MSELoss` grampeava assim mesmo por baixo --- o chamador pedia "sem clip" e recebia clip; (c) não era um reescalonamento constante: só dispara quando a norma passa de 1, então a taxa de aprendizado efetiva virava função da magnitude do gradiente (distorção não-linear, não um "lr 2× menor"); (d) `CrossEntropyLoss`/`SpikeCountLoss`/`SpikeTimeLoss` **não** fazem isso --- as perdas eram mutuamente inconsistentes. Mesma espécie de D3: o declarado não era o executado. **Decisão do autor: tornar configurável, default OFF.** Agora `max_gradient_norm = 0.0F` (desligado) → `backward()` devolve o gradiente exato e bate com o `torch` na casa decimal; o clip continua disponível como escape hatch. O teste `MSELossTest.GradientIsClipped`, que fixava o comportamento antigo, foi substituído por `GradientIsExactByDefault` + `GradientIsClippedWhenExplicitlyEnabled`.

2. 🟡 **O LIF não é "exatamente o `snn.Leaky`" do snnTorch --- só no modo que usamos.** `gen_pytorch_refs.py` afirmava "This is exactly snnTorch's snn.Leaky(beta, threshold, reset_mechanism)". Falso para `subtract`. Nosso reset é aplicado **imediatamente** (`v -= V_th`, e portanto decai no passo seguinte); o do snnTorch é subtraído **não-decaído** no passo seguinte:
   - nós:      `v[t] = beta*v[t-1] + I[t] - V_th*spk[t]`
   - snnTorch: `mem[t] = beta*mem[t-1] + I[t] - V_th*spk[t-1]`
   
   Ou seja, nosso termo de reset acaba multiplicado por `beta`. Medido: **desacordo de 1,9--3,0% dos disparos** no modo `subtract`, e **0,0% no modo `zero`**. **Impacto real: nenhum** --- `Lif`/`LifBPTT` têm `reset_zero = true` por default e **nenhum código de produção seleciona `subtract`**, então a tese usa exatamente o modo que casa perfeitamente. O nosso é a forma de livro-texto (soft-reset padrão); o do snnTorch é convenção do snnTorch. Nenhum está errado --- a **afirmação de equivalência** é que estava. Por que passou despercebido: a fixture por-camada existente dispara só **3 de 36 vezes (8%)**, fraca demais para exercitar o caminho de reset. Os novos testes fixam ambos os modos sob drive forte: `zero` bate exato, `subtract` diverge dentro de banda asseverada.

3. 🟡 **Nosso LSTM não usa sigmoid/tanh.** Usa as aproximações racionais de `FastActivations.hpp` (`rat_sig(x)=0.5+x/(2(1+|x|))`, `rat_tanh(x)=x/(1+|x|)` --- gates *softsign*, escolha de velocidade). Não são próximas: `|tanh - rat_tanh|` chega a **0,306** em [-4,4] (em x=2: tanh=0,964 vs nosso=0,667), e a divergência medida nos estados ocultos vs `torch.nn.LSTM` é **0,1626**. Logo o nosso é um **LSTM de gates softsign** e não pode casar com o `torch` --- comparar contra ele exigiria tolerância tão frouxa que não provaria nada. O teste fixa contra um modelo NumPy da **nossa própria** recorrência (ainda pega ordem de gates i,f,o,g vs i,f,g,o do torch, ordem de atualização c/h, bias fundido, composição) e **assevera que a divergência do torch real fica na banda esperada**, para que a aproximação não cresça em silêncio nem alguém leia "LSTM" e assuma semântica padrão. Relevante para o artigo de Guaiaquil, que compara "LSTM-AE" vs SNN.

**Limites deliberados, codificados e não escondidos:** o backward *spiking* não é comparável (nosso surrogate é exponencial, o do snnTorch é arctan --- funções diferentes), então o backward é fixado em **modo readout**, onde não há surrogate e ambos os lados têm de bater exato; o forward *com* disparos é comparado normalmente. Há também um teste de que `reset_state()` de fato isola sequências (a invariante #4 do `CLAUDE.md`), que nenhuma checagem de disparo único pegaria.

---

## Referência = PyTorch/snnTorch: fidelidade configurável por perfil (2026-07-16)

Decisão do autor, em quatro partes: (1) os perfis devem poder escolher se usam ou não as aproximações rápidas; (2) o grampeamento de gradiente deve ser configurável no perfil, **default OFF**; (3) nosso código deve casar também com o modo `subtract` do SNN; (4) **como o comportamento do snnTorch/PyTorch é a referência, o nosso deve se comportar igual**. O item (4) decidiu os outros: onde havia escolha, o default passou a ser o que casa com a referência.

**1. `training.gradient_clip_norm` (default `0.0` = OFF).** Plumbado nos três pontos que constroem `TrainerConfig` → `grad_clip_norm`, validado (`>= 0`). Junto com a correção do `MSELoss`/`MAELoss` (que grampeava sozinho, sempre, em norma 1,0), agora existe **um único** knob de clipping, honesto e visível no perfil, em vez de um escondido sobrepondo o configurado.

**2. `numerics.exact_activations` (default `true` = exato).** Como o PyTorch é a referência, o **exato virou o default** e a aproximação virou opt-in explícito. Adicionados `sigmoid_exact_block`/`tanh_exact_block`/`tanh_exact_tensor` + dispatchers em `FastActivations.hpp`; `LSTMLayer` ganhou o flag `exact_activations` (default `true`), usado no forward **e** no backward.

> **Resultado: nosso LSTM agora casa com o `torch.nn.LSTM` exatamente**, elemento a elemento, em todos os backends. O teste `PyTorchParityTyped.LSTMLayerForward` usava um **bound frouxo de 0,25** justamente porque a camada usava softsign incondicionalmente; agora é comparação apertada contra o PyTorch. Os dois modos ficam fixados: exato ≡ torch, rápido ≡ aproximação --- então o trade velocidade/fidelidade continua existindo, exercitado, e não pode virar mentira em silêncio.

**3. 🔴 Segundo bug real, achado ao fazer o (2): o backward do LSTM não correspondia ao seu forward.** O backward calculava `y(1-y)` e `1-y²` --- as derivadas do sigmoid/tanh **exatos** --- enquanto o forward rodava as aproximações racionais. Ou seja, no modo rápido o gradiente era a derivada de uma função que o forward nunca avaliou, **errado por até 5×** (em x=2: `1-y²` dá 0,556 onde `rat_tanh'(2) = 0,111`). O exato-por-default corrige isso de graça (aí `y(1-y)` e `1-y²` *são* as derivadas certas). Para o modo rápido, derivei as formas fechadas a partir da saída em cache (o backward não tem a pré-ativação): com `s = y - 0,5`, `rat_sig' = (1-2|s|)²/2` e `rat_tanh' = (1-|y|)²` --- verificadas contra as derivadas analíticas com erro ~1e-16. Agora o modo rápido é uma aproximação de verdade, não um gradiente errado.

**4. Modo `subtract` do LIF: divergência mantida, documentada e fixada (decisão do autor).** Determinei a semântica exata do snnTorch por tracing (não por leitura de código):
   - `subtract`: `mem[t] = beta*mem[t-1] + I[t] - V_th*spk[t-1]` --- reset **não-decaído**, aplicado no passo seguinte; o `mem` fica **sem reset** (no trace, `mem_out = 1,5 > V_th`).
   - `zero`: `mem[t] = beta*mem[t-1]*(1-spk[t-1]) + I[t]` --- o reset mata o `beta*mem` anterior mas **mantém a entrada nova** (t=1: `mem_out = 0,9 = I` exatamente).

   Nosso `zero` **já é equivalente** (guardar 0 pós-reset ≡ zerar `beta*mem_prev`), o que explica os 0,0% de desacordo --- e é o modo que a tese usa. Só o `subtract` diverge (~2--3% dos disparos), porque aplicamos o reset imediatamente e ele acaba multiplicado por `beta`.

   **Decisão do autor: deixar fixado como divergência documentada de caminho não usado.** Alinhá-lo exigiria inverter o estado guardado de pós-reset para sem-reset no forward **e** no backward BPTT do `LifBPTT` (que encadeia `v_post_history` em `dL/dR`, `dL/dC` e no termo de reset de `dL/dVth`) --- cirurgia no neurônio de que toda a tese depende, para consertar um caminho que **nenhum código de produção seleciona** (`reset_zero = true` é o default). Fixado por `MicroNetworkParity.LifSubtractResetDivergesFromSnntorchAsDocumented`, que assevera que a diferença fica dentro da banda medida --- não pode crescer despercebida enquanto espera. Documentado no cabeçalho do `Lif.hpp` **e** do `LifBPTT.hpp` (ambos afirmavam equivalência com o snnTorch sem ressalva) e no `gen_pytorch_refs.py`.

**Arquivos:** `FastActivations.hpp`, `LSTMLayer.hpp`, `MSELoss.hpp`, `MAELoss.hpp`, `Lif.hpp`, `LifBPTT.hpp`, `E05Config.{hpp,cpp}`, `E05Classifiers.cpp`, `E05FeatureExtraction.cpp`, `pytorch_parity_gtest.cpp`, `micro_network_parity_gtest.cpp`, `layers_gtest.cpp`, `gen_pytorch_refs.py`, `gen_micro_network_refs.py`.

---

## Auditoria de comentários não-confiáveis (2026-07-16)

Varredura pedida pelo autor ("scan the code looking for untrustworthy comments"). Priorizei a classe de defeito que já mordeu este projeto duas vezes: **comentário que afirma um contrato que o código não cumpre** (D3, D5) e **citação vaga/errada** (item 57). Não é uma varredura exaustiva de todo comentário do repo.

**Corrigidos nesta sessão:**

- [x] **C-1. `Optimizer::attach()` mentia sobre si mesmo.** O comentário mandava os otimizadores concretos chamarem `Optimizer::attach(params)` "to preserve this storage for no-arg convenience methods" --- mas o corpo era `{}`, um no-op literal, e não preservava nada. Consequência real: o **SGD confiava no contrato e nunca populava `attached_params_`**, então `sgd.step()` sem argumentos sempre lançava mesmo após um `attach()` correto. A base agora honra o que documenta (ver D5).
- [x] **C-2. `TrainerConfig.hpp`: "SNN-specific fields are ignored for pure ANN models".** Era falso antes de D3 (escalava todos os pesos de qualquer modelo). Voltou a ser verdadeiro *por causa* da correção de D3, não por edição do texto --- ver F1.
- [x] **C-3. `Trainer.hpp`, lista "Bugs fixed", item 4.** Dizia "snn_lr_scale wired via attach_with_scales (was silently ignored)" --- apresentava como *bug corrigido* justamente a linha que **era** o bug de D3 (o wiring existia, mas estava errado). Reescrito com o comportamento real e nota explícita de que a redação anterior era enganosa.
- [x] **C-4. `Lif.hpp`: citação mal-atribuída.** Dizia "Reference: [34-35] MPD-ATP (IEEE Xplore 2025); AR-LIF (arXiv 2025)". `[35]` de fato é MPD-ATP (Wang et al., IEEE Xplore 2025) ✅, mas `[34]` é **Lv et al., PMC 2025**, sobre adaptação espaço-temporal --- não um arXiv chamado "AR-LIF". Corrigido contra `References.md`. Mesmo padrão do item 57, embora aqui as referências existam de verdade (só o rótulo estava errado).
- [x] **C-5. `Adam.hpp`: comentário do weight decay descrevia a ordem errada.** Dizia que o decay é aplicado após o passo; a definição de AdamW (e o `torch.optim.AdamW`) o aplicam sobre θ_{t-1}. O comentário *descrevia fielmente o código* --- e o código é que estava errado. Ver D5.

**Encontrados, não corrigidos (baixa severidade, registro para decisão):**

- [x] **C-6.** ✅ Comentários `@file` apontando para caminhos inexistentes. **Eram 106, não 41** --- minha contagem anterior só pegou o padrão `include/nn/`; havia também `src/core/dataLoaders/` (real: `data_loaders`) e 15 basenames errados (ex.: `paraconsistent.h` para um `.hpp`). Dos 350 arquivos com `@file`: 206 usam só o basename (convenção dominante), 53 usam caminho completo. Corrigi os 106 obsoletos **cada um no estilo que ele mesmo pretendia** (quem reivindicava caminho → caminho real; quem reivindicava basename → basename real), em vez de uniformizar 350 arquivos e inflar o diff. Verificado: **0 obsoletos** em 350.
- [x] **C-7.** ✅ `LifBPTT::spike_history` removido. Verificado antes: **uma única referência em todo o repo --- a própria declaração**; não aparece em `state_dict`/`load_state_dict` (logo não quebra checkpoints), nem em `reset_state()`, e nunca era lido nem escrito. Compila sem ele.

---

## Correções mecânicas (decorrem das decisões acima)

- [x] **F1.** O comentário em `TrainerConfig.hpp:10` afirma que `snn_lr_scale` "is ignored for pure ANN models". Era falso antes da correção de D3 (escalava todos os pesos por 0,1 mesmo em ANN puro --- o efeito medido no ANN-AE foi de +4,6σ). **Resolvido pela própria correção de D3**: com a escala agora restrita a parâmetros 1×1, e nenhum modelo ANN puro (ex. `ProtocolAutoencoder`) tendo parâmetro 1×1 algum (pesos e vieses são sempre >1 elemento), `snn_lr_scale` de fato não afeta mais modelos ANN --- o comentário voltou a ser verdadeiro sem precisar editá-lo.
- [x] **F2.** Nenhum teste cobria o caminho em que `snn_lr_scale` é diferente de 1.0. Adicionado `TrainerGenericity.SnnLrScaleOnlyAppliesToSizeOneParams` (`trainer_genericity_gtest.cpp`): modelo com um parâmetro 1×1 e um 2×2, mesmo gradiente fixo nos dois, confirma que só o 1×1 recebe a escala reduzida. Verde.
- [x] **F3.** ~~`results/phase00/` não era reproduzível a partir dos perfis~~ --- **superado**: os resultados foram apagados (ver D3), então a inconsistência deixou de existir. A reexecução dos 208 perfis, quando acontecer, produzirá um conjunto coerente com os perfis atuais e com summaries que gravam o lr efetivo.
- [x] **F4.** ✅ A tabela "Measured separability" de `.wiki/Experiments/Experiment05.md` foi **retirada, não corrigida** --- não havia número a corrigir: saíam de execuções sob o bug de D3, o "rank (of 150, eeg)" ficou aritmeticamente impossível após D6 (o EEG tem 58), e os dados-fonte foram apagados. Substituída por um bloco que explica os três motivos e **preserva o mecanismo** (que é teoria e sobrevive), apontando para D2.
- [x] **F5.** A contagem "138 variantes handcrafted por sinal" era enganosa para o EEG. Corrigida na tese (§08: 138 voz + 46 EEG; grade 300 → 208) e na wiki junto com **D6**. A seção "Referência" abaixo foi atualizada.

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

1. **[BLOQUEADO PELA REEXECUÇÃO]** Rodar `01_e05_phase00_rank.py` para produzir `winners.json`. D1 já foi decidido e implementado (o script seleciona por `d_penalized` e agora também detecta/reporta empates exatos --- ver D6), então o bloqueio *conceitual* caiu. Mas os resultados da Fase 00 foram **apagados** (ver D3) e a reexecução está **adiada por decisão do autor**, logo não há dado de entrada: rodar agora não produz nada. Ordem correta: reexecutar os 208 perfis → rodar este script → item 2.
2. Rodar `02_e05_apply_winner.py` para injetar o vencedor real nos 32 perfis da phase01, substituindo o bloco `feature_extraction` provisório.
3. Executar os 32 perfis `classifier.type=dsnn` da phase01 (`run_e05_profiles.sh phase01`) --- este é o experimento de autenticação real da tese e atualmente não tem nenhum resultado.
4. Com resultados reais do DSNN em mãos, considerar uma rodada explícita de ablação para `weight_decay`, `firing_rate_reg_lambda` e tdBN --- nenhum dos três jamais foi exercitado fora do perfil debug/smoke:
   - [ ] `training.weight_decay` > 0 --- só configurado em `debug.json`/`smoke/debug.json`; não está em nenhum perfil real da phase01
   - [ ] `training.firing_rate_reg_lambda` > 0 --- mesma situação: apenas debug/smoke
   - [ ] `training.batch_normalization = "threshold-dependent"` (tdBN) --- caminho de código implementado (`E05Config.hpp:135`, `ThresholdDependentBatchNorm`), mas nunca configurado em nenhum perfil publicado, debug incluso --- ainda genuinamente não testado de ponta a ponta

Status completo verificado: ver "Log de status do Experiment05" abaixo.

## Fazer a tese gerar as mesmas informações que o paper de Guayaquil
- Guayaquil também armazena informações e estatísticas muito boas sobre as execuções; faça com que a tese gere os mesmos dados.

## Questões em aberto

- Estou vendo que você vai testar apenas o perfil de autoencoder com 4 camadas, mas acho bem provável que o estado da arte para dispositivos de baixo poder computacional (tipo Raspberry Pi B) use mais camadas. Estou errado?

## Aprofundar (revisão de texto/tese)

- [ ] **51. Avaliar diferentes algoritmos de otimização.** → **infraestrutura pronta, ablação não rodada** (ver **D5**). Feito: otimizador polimórfico via `OptimizerFactory`; `attach_with_scales` virtual na base (todos honram escalas por grupo); `training.optimizer_type` exposto no JSON dos perfis; **Lion** e **Schedule-Free AdamW** implementados a partir do código-fonte das referências e validados por ground truth (`optimizer_parity_gtest`, 10/10). Sophia, SOAP e Muon descartados com motivo técnico registrado em D5. Falta apenas: (a) definir a grade de lr **por otimizador** (um lr único não é comparável --- ver aviso em D5) e (b) rodar a ablação, que é experimento caro (mesma ressalva de D1/D2/D3).

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

`lfcc_c2`×{eeg,voice}, `mel_c1`×{eeg,voice} falharam na execução original em lote paralelo dos 300 perfis. Reexecutados individualmente em 2026-07-16 e passaram sem problemas — contenção transitória de recursos por causa dos jobs paralelos, não um defeito de código. `results/run_profiles_phase00.state` mostrava 300/300 PASS. ⚠️ Obsoleto: os resultados da Fase 00 foram apagados e o state zerado em 2026-07-16 (ver D3); a grade agora é 208, não 300.

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

  cada uma carregando o conjunto de descritores: energy, ZCR, entropy, Teager-Kaiser, jitter, shimmer. Para a **voz**: 23 × 3 = 69 combos (wavelet × escala), ×2 pelo `cepstral` = **138** variantes. Para o **EEG**: a escala não se aplica (ver **D6** --- bark/mel são escalas cocleares sem base fisiológica para EEG, e degeneram no linear), logo 23 × 1 × 2 = **46** variantes.
- **Autoencoder** --- 12 AEs compactos por sinal: 9 SNN-AE (pulsante; 3 tamanhos × 3 codificações temporais --- poisson/latency/direct) e 3 ANN-AE (denso; 3 tamanhos), latente 8/16/32, 2:1 na camada oculta.

Logo a grade da Phase 00 = (138 + 12) voz + (46 + 12) EEG = **208 rankings**, cada um pontuado por α, β, G1, G2, D_truth (sec:conceitos). Saída desta fase: um vetor de características vencedor para voz, um para EEG.

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

## Phase 00 --- extração de características (208 perfis: voz 138 handcrafted + EEG 46 handcrafted + 24 autoencoder)

**Categoria 1 vs 2** (auditoria G2, implementada): `handcrafted.scale` agrupa as sub-bandas da DTWPT por frequência (linear/Bark/Mel). Com `cepstral=false` as energias por banda são usadas diretamente (Categoria 1); com `cepstral=true` um estágio log+DCT-II sobre essas energias produz os coeficientes cepstrais LFCC/MFCC/BFCC (Categoria 2). Ambas as categorias são selecionáveis e publicadas.

Voz e EEG (mesmo status, mesmo caminho de código, agnóstico ao sinal):
- [x] Energia de banda Linear/Mel/Bark (Categoria 1) --- `cepstral=false`, 23 wavelets × 3 escalas × 2 sinais = 138 perfis, todos executados
- [x] LFCC/MFCC/BFCC (Categoria 2) --- `cepstral=true`, 138 perfis, todos executados
- [x] SNN-AE --- `ProtocolSpikingAutoencoder`; 3 tamanhos × 3 codificações temporais (poisson/latency/direct) × 2 sinais = 18 perfis, todos executados
- [x] ANN-AE --- `ProtocolAutoencoder`; 3 tamanhos × 2 sinais = 6 perfis, todos executados

O próprio ranking paraconsistente (α, β, G1, G2, D_truth): **[x] implementado**, `E05Paraconsistent.cpp`, exercitado por `e05_profile_audit_gtest`.

**⚠️ Phase 00 NÃO tem resultados (2026-07-16).** Todos os 1800 arquivos de `results/phase00/` foram apagados e o `run_profiles_phase00.state` zerado, porque estavam invalidados por D1/D3/D6 (ver D3 para o detalhamento). A reexecução dos 208 perfis está **adiada por decisão do autor**. Os `[x]` abaixo indicam apenas que o perfil existe e o caminho de código está implementado --- **não** que haja resultado em disco. As tabelas da tese foram regeneradas a partir dos dados antigos antes da remoção, então a tese ainda compila (números handcrafted válidos; linhas de AE provisórias).

**Lacuna restante**: `scripts/pipeline/e05/01_e05_phase00_rank.py` (lê os 300 resultados, escolhe o vencedor por sinal, grava `winners.json`) ainda não foi rodado contra o conjunto de resultados agora completo --- `winners.json` não existe. Os perfis da Phase 01 ainda carregam um extrator provisório `daub4/lfcc` em vez do vencedor real da Phase 00. → item 1 do TODO.

## Phase 01 --- autenticação DSNN (32 perfis: 4 modos de fonte × 2 text_mode × 2 CV × 2 standardize_features)

Todo `profiles/phase01/*.json` define `classifier.type = "dsnn"` (32 perfis; os antigos perfis `"rnn"` agora vivem apenas em `debug.json`/`smoke/debug.json` e são exclusivos do artigo de Guayaquil, fora do escopo da tese). `E05DsnnClassifier` está implementado e testado (`e05_classifiers_gtest`, 16/16 passando), e todo eixo está publicado como perfil real:

- [~] fused-early, fused-late, voice-only, eeg-only × classifier=dsnn --- 8 perfis por combinação de modo, publicados, **ainda não executados**
- [~] `text_mode`: `dependent` e `independent`, cada um pareado com dados reais (não-debug) --- publicados, **ainda não executados**
- [~] `nested_cv`: `true` (nested 5-fold) e `false` (flat grouped 5-fold) --- publicados, **ainda não executados**
- [~] `training.standardize_features`: `true` e `false` --- publicados, **ainda não executados**

**Maior lacuna restante**: `results/phase01/` não existe --- **nenhum dos 32 perfis DSNN foi executado**. O classificador primário da tese tem cobertura completa de código + perfis mas nenhum resultado. Bloqueado pela lacuna do `winners.json` acima (substituir o extrator provisório via `02_e05_apply_winner.py` antes de rodar). → itens 2--3 do TODO.

Chaves de regularização/normalização: ver item 4 do TODO acima (weight_decay / firing_rate_reg_lambda / tdBN, nenhum exercitado fora do debug/smoke).
