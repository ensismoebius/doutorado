# Adiadas / rejeitadas / Aprofundar
7. Página 17: Incluir passo a passo do cálculo das transformadas wavelet packet. (nem fodendo)
8. Página 18: Listar outras técnicas além da engenharia paraconsistente para consistência de características. (por enquanto não)
24. Página 31: Verificar consistência entre as equações do LIF e o código utilizado.
25. Figura 19: Explicar que os pulsos apresentados são resultado de uma simulação de neurônios de pulso/RNP.
30. Seção BPTT: ESTUDAR!!!!!
40. Revisar todo o trabalho para garantir que LFCC seja a representação espectral principal.
42. Criar tabela comparativa entre métodos manuais, automatizados, escalas e wavelets.
47. Criar seção específica para Threshold-Dependent Batch Normalization (TDBN). ok (ESTUDAR!!!)
48. Revisar toda a monografia/Wiki para garantir que toda variável seja explicitamente definida.
49. Avaliar arquiteturas compactas de autoencoders para Raspberry Pi.
51. Avaliar opcionalmente diferentes algoritmos de otimização.
53. Revisar sistematicamente e enriquecer a Wiki do projeto.
56. Avaliar comparativamente diferentes arquiteturas de autoencoders utilizando Engenharia Paraconsistente de Características.
57. Verificar e fundamentar a afirmação sobre taxas de aprendizado para parâmetros biofísicos em SNNs.
58. Fundamentar a tabela de associação entre codificações e funções de perda.

# Observações e Correções Fornecidas pelo Usuário

1. Página 9: Incluir setas no fluxo da Figura 1 e citar o teorema de Nyquist. ok
2. Página 11: Corrigir o uso da escala Bark com wavelet Haar, em vez de escala Bark com Mel. ok
3. Página 11: Na legenda da figura, adicionar a referência original de onde a figura foi adaptada. ok
4. Página 13: Na figura da escala Mel, adicionar a referência original. ok
5. Página 15: Definir o que é uma matriz ortogonal. ok
6. Página 16: Esclarecer que filtros normalizados são uma estratégia auxiliar e não uma condição fixa. ok
9. Página 18: Fazer revisão gramatical completa do documento. ok
10. Página 19: Explicar o plano paraconsistente antes de utilizá-lo. ok
11. Página 20: Corrigir o vetor da classe C1 para (-4, -9, -7, -2). ok
12. Página 21: Remover a listagem redundante do plano paraconsistente. ok
13. Página 21: Adicionar uma label ao ponto P na Figura 8. ok
14. Página 23: Inserir a palavra “geralmente” ao falar do intervalo de frequência cerebral.ok
15. Página 23: Registrar que a amplitude da onda delta não foi fornecida pelas fontes.ok
16. Página 23: Explicitar a ausência da amplitude da onda delta. ok
17. Página 24: Verificar a referência “2023a”, possivelmente incorreta.ok
18. Páginas 25 e 26: Redimensionar figuras que ocupam espaço excessivo.ok
19. Página 28 (Figura 14): Trocar “área perisilviana” por “fissura de Silvio”.ok
20. Página 29 (Figura 15): Definir claramente o que significa fala fluente e não  fluente.ok
21. Página 30 (Figura 16): Adicionar descrição explicando o que a figura representa.ok
22. Página 31 (Figura 17 – Modelo RC): Corrigir símbolo que parece letra “V”.ok
23. Página 31: Explicar que a redução do custo computacional do neurônio LIF também vale para hardware convencional, não apenas neuromórfico.ok
26. Figura 19: Manter a figura antes da formalização matemática para fornecer intuição visual inicial.ok
27. Página 34: Ao explicar o decaimento da voltagem/potencial da membrana, justificar por que essa explicação está sendo apresentada e qual sua relevância. ok
28. Página 35 — Corrigir “equação ordinária de primeira ordem” para “equação diferencial ordinária (EDO) de primeira ordem” e verificar se é apropriado especificar que se trata de uma EDO linear de primeira ordem. ok
29. Página 38 — Definir feedforward, backpropagation, Reservoir Computing, ESN e LSM e relacionar ESN/LSM com SNNs.ok
30. Seção BPTT: Conferir referências e verificar cobertura de unrolling, gradientes, gradiente desaparecendo/explodindo e surrogate gradients. ok
31. Seção BPTT: Incluir figura ilustrando o desenrolamento temporal.ok
32. Página 38: Remover o último parágrafo por redundância.ok
33. Seção BPTT: Reescrever e expandir significativamente a explicação.ok
35. Figura 24: Trocar círculos por blocos/retângulos para representar camadas.ok
34. Função de Resposta Exponencial: Revisar e expandir a seção.
37. Figura 27: Trocar círculos por retângulos para representar camadas. ok (não precisa)
36. Apêndices — Técnicas de Regularização: Incluir L1, L2 e Ω. (apenas L2). ok
38. Seção “Por que LFCC para Biometria?”: Mover para posição mais inicial e justificar a escolha da LFCC. ok
39. Seção “Por que LFCC para Biometria?”: Adicionar figura mostrando o pipeline completo do cálculo da LFCC. ok
41. Avaliar inclusão de comparação entre LFCC, Mel e Bark. ok
43. Inserir especificação da LFCC em paralelo às descrições de Mel e Bark. ok
44. Expandir regularização para incluir L1, L2 e Ω. (Apenas L2) ok
45. Decidir se os modelos usarão regularização e justificar a escolha. ok
46. Criar subseção específica para BPTT aplicado a SNNs. ok
50. Criar seção sobre inicialização de pesos em redes neurais.  ok
52. Padronizar a apresentação de variáveis antes das equações. ok
55. Demonstrar passo a passo a derivação das expressões matemáticas.ok
54. Incluir exemplos numéricos completos.ok
62. Expandir e fundamentar a seção “Profile Guided Optimization (PGO)”.ok
61. Fundamentar e referenciar adequadamente a seção “Why this range”. ok
60. Expandir e fundamentar a seção “The no-spike problem”.ok
59. Fundamentar teoricamente as estratégias de normalização utilizadas no projeto.ok

--- Add contraditions bellow ---

# Contradições e incoerências Wiki × Tese (levantadas 2026-07-01)

Fonte da verdade quando há conflito de números da base: código
`include/data_loaders/10.1117/schema/Metadata.hpp` (EEG 1024 Hz, 6 canais,
4096 amostras/canal; áudio 44100 Hz, 176400 amostras). A tese bate com o código.

C1. Taxa de amostragem e canais da base divergem entre tese, wiki e README.
   - Tese (08-proposedApproach.tex:9): áudio 44100 Hz mono; EEG 1024 Hz, 6 eletrodos (F3,F4,C3,C4,P3,P4); 4096 amostras EEG / 176400 áudio por trial de 4 s. → CORRETO (bate com Metadata.hpp).
   - Wiki (Experiments/Experiment05.md:127-129, 242-248, 373, 389 e Research-Context.md:63-68): áudio 22050 Hz, EEG 800 Hz. → ERRADO.
   - Código README (src/experiments/05/README.md:290, 368): EEG 800 Hz, 14 canais, Emotiv EPOC. → ERRADO em taxa E em número de canais.
   Ação: corrigir wiki Experiment05.md, Research-Context.md e o README do exp05 para 44100/1024 Hz, 6 canais.

C2. Passa-banda de EEG fisicamente impossível na wiki. Wiki (Experiment05.md:129,248) afirma "bandpass 1–800 Hz" com EEG a 800 Hz; e mesmo com o valor correto de 1024 Hz, a frequência de Nyquist é 512 Hz, logo um passa-banda até 800 Hz não existe. A tese não menciona esse pré-processamento. Ação: corrigir a wiki; se houver pré-processamento real, fundamentá-lo na tese.

C3. Número de modalidades. Tese (08:7): 2 modalidades de fala (fonatória e imaginada). Wiki (Experiment05.md:126): "Three modalities: phonated speech, imagined speech, mixed" — mistura modalidade de fala com fusão de sinais (voice/eeg/fused). Ação: uniformizar terminologia. E assegurar que a tese cubra o uso de fala fonada + fala imaginada, pois afinal de contas, é o tema da pesquisa. Na tese deve haver as estratégias de fala fonada+imaginada, apenas fonada, apenas imaginada.

C4. Lista de comandos direcionais. Tese (08:7): 6 comandos (arriba, abajo, adelante, atras, derecha, izquierda). Wiki (Experiment05.md:125): 5 comandos (omite "atras"). Ação: corrigir wiki para 6.

C5. Métricas de avaliação são incoerentes entre tese e wiki.
   - Tese (08:64-73) lista como métricas: acurácia, F1, precisão, recall, MSE, AUC, sensitividade, especificidade. Não lista EER.
   - Wiki (Experiment05.md:351-364) declara protocolo verification-only (folds speaker-disjoint): acurácia/F1/precisão/recall/especificidade NÃO são reportadas (NaN); EER e AUC são as métricas primárias.
   Conflitos: (a) a tese lista métricas que a wiki diz não reportar; (b) EER (métrica primária na wiki) está ausente da tese; (c) MSE (métrica de reconstrução) aparece na lista de métricas de classificação da tese. Ação: alinhar a lista de métricas da tese ao protocolo verification-only e incluir EER, ou justificar a divergência. Ação: Sincronizar Tese e Wiki de forma que ambos tenham a lista completa.

C6. Número de estratégias de normalização. Tese (07, seção "Normalização de Características de Entrada") afirma "duas estratégias" (áudio por característica; EEG por janela). O projeto usa três esquemas: soma a normalização min-max [0,1] exigida pela engenharia paraconsistente (wiki Data-Normalisation e Experiment05.md:32,387; tese seção paraconsistente também usa [0,1]). A seção dedicada de normalização não menciona esse terceiro esquema. Ação: incluir o min-max paraconsistente na seção de normalização da tese.

C7. Pré-ênfase do áudio não explicada na tese. Wiki (Experiment05.md:243-244) lista o pré-processamento de áudio como "normalization, pre-emphasis, windowing"; demos usam pré-ênfase α=0.97 (snn-speaker-demo.md:45). A tese não descreve nem fundamenta a pré-ênfase. Ação: descrever/fundamentar na tese.

C8. Arquitetura do classificador. Tese (08:30,77) descreve um único classificador: "rede neural residual profunda de pulsos" (residual + spiking). Wiki (Experiment05.md:80-89) descreve DOIS classificadores selecionáveis: RNN (residual NÃO-spiking, com BatchNorm+ReLU) e DSNN (spiking). Além disso, a wiki descreve a conversão de vetor estático em pulsos por injeção de corrente constante ao longo de kSnnTimeSteps (default 16) — etapa não descrita na tese. Ação: O classificador residual não-spiking foi feito para confecção do paper de Guaiquil, a tese usará como classificador apenas a "rede neural residual profunda de pulsos".

C9. Autoencoder implementado. Wiki (Experiment05.md:74-76, 393): o único AE implementado é LSTM-AE; SNN-AE é "planned". Tese (07 seção Autoencoders; 08:28) trata autoencoders de forma genérica, nunca menciona LSTM-AE, e o texto sugere autoencoders de pulso. Ação: A rede LSTM-AE; foi feita apenas para o paper do congresso de Guayaquil, ignore, quanto aos auto-encoders usados na Tese a intenção é comparar handcrafted, SNN-AE e ANN-AE via engenharia paraconsistente de características.

C10. Comparações sem citação na wiki. Experiment05.md:24 ("Standard MFCC+GMM or x-vector systems degrade sharply") e :26 ("a severely dysphonic speaker produces identical imagined-speech EEG to a healthy speaker") são afirmações comparativas/fortes sem referência. Ação: citar fontes CONFIÁVEIS ou excluir.

C11. Bloco residual: estrutura interna divergente. Wiki (Experiment05.md:87) define o bloco residual do RNN como F(x)=2 Linear + BatchNorm + ReLU. A tese (07, "Aprendizado residual") descreve o resíduo de forma genérica (r=h−x, h=r+x) sem BatchNorm/ReLU nem a composição usada no código. Ação: alinhar a descrição do bloco residual da tese à implementação.