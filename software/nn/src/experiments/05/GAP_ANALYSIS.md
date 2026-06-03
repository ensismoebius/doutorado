# Experiment05 Gap Analysis

Data: 2026-06-03
Escopo: comparar implementacao atual com wiki de Experiment05, contexto de pesquisa e monografia.

## Matriz de Conformidade

| Requisito documental | Status | Evidencia no codigo | Evidencia documental | Acao objetiva |
|---|---|---|---|---|
| E3 automatico com autoencoders (LSTM-AE/SNN-AE) operacional | Parcial | [E05FeatureExtraction.cpp](lib/src/E05FeatureExtraction.cpp) implementa treino/inferencia para `lstm-ae`; `snn-ae` nao suportado | [Experiment05.md](../../../.wiki/Experiments/Experiment05.md#L67), [08-proposedApproach.tex](../../../../../documentation/00-thesis/monography/chapters/08-proposedApproach.tex#L146) | Manter `lstm-ae` como baseline executavel e declarar `snn-ae` como trabalho futuro |
| E4 com RNN e DSNN | Conforme | [E05Classifiers.cpp](lib/src/E05Classifiers.cpp) implementa ambos os caminhos (`rnn` e `dsnn`) com nested/flat CV | [Experiment05.md](../../../.wiki/Experiments/Experiment05.md#L16), [Research-Context.md](../../../.wiki/Research-Context.md#L38) | Coberto por testes de classificador e perfis |
| Modos text-dependent e text-independent afetam split/treino/teste | Conforme | [E05Classifiers.cpp](lib/src/E05Classifiers.cpp) aplica `make_text_split` e intersecoes nos indices de treino/teste | [Experiment05.md](../../../.wiki/Experiments/Experiment05.md#L86), [Research-Context.md](../../../.wiki/Research-Context.md#L40) | Coberto por testes funcionais de classificador |
| Base 10.1117 carregada de forma reproduzivel | Conforme (baseline atual) | Perfis usam sqlite em [article-full.json](profiles/article-full.json#L9) e discovery suporta raiz sqlite/diretorio em [E05Dataset.cpp](lib/src/E05Dataset.cpp) | [README.md](README.md#L281) descreve base 10.1117 | Manter contrato documentado e mensagens de erro explicitas |
| Nested CV completo (outer + inner para selecao) | Conforme | [E05Classifiers.cpp](lib/src/E05Classifiers.cpp) percorre todos os inner folds e seleciona melhor modelo por validacao | [Experiment05.md](../../../.wiki/Experiments/Experiment05.md#L95) descreve nested para evitar vies | Coberto no pipeline atual |
| transform configuravel (dtwpt/lfcc/mfcc) | Conforme ao baseline | [E05Config.cpp](lib/src/E05Config.cpp) restringe `transform=dtwpt`; [E05FeatureExtraction.cpp](lib/src/E05FeatureExtraction.cpp) implementa Bark/MEL/LFCC via agrupamento de bandas | [README.md](README.md#L327) | Manter contrato explicito: DTWPT unico transform no binario |
| classificador por layer_spec completo | Parcial | Parser simplifica para hidden_dim/depth em [E05Classifiers.cpp](lib/src/E05Classifiers.cpp#L133) | [README.md](README.md#L343) sugere especificacao de arquitetura por layer_spec | Alinhar parser ao contrato ou simplificar documento |
| EPC/paraconsistente com alpha beta d_truth | Conforme | [E05Paraconsistent.cpp](lib/src/E05Paraconsistent.cpp) | [README.md](README.md#L205), [Research-Context.md](../../../.wiki/Research-Context.md#L105) | Manter; adicionar teste de regressao de ranking |
| Saidas de resultados (CSV/JSON/DAT/modelos) | Conforme | [E05Output.cpp](lib/src/E05Output.cpp), salvamento em [E05Classifiers.cpp](lib/src/E05Classifiers.cpp#L357) | [README.md](README.md#L407) | Manter; incluir metadados de split e seed por fold |

## Impacto na Tese

1. Risco de claim parcial: SNN-AE ainda nao esta executavel no binario baseline.
2. Risco de escopo: manuscrito precisa deixar explicito baseline final (`lstm-ae + rnn/dsnn`).

## Ordem Minima de Correcao

1. Congelar escopo final: implementar SNN-AE ou remover claim correspondente do texto final.
2. Alinhar contrato de `layer_spec` no texto com parser/arquitetura efetivamente usada.
3. Reexecutar matriz experimental final e regenerar tabelas do capitulo 9.

## Criterio de Pronto para Capitulo 9

1. Cada claim metodologico do texto tem caminho executavel no codigo.
2. Perfis oficiais rodam fim-a-fim sem fallback silencioso.
3. Modo dependent e independent geram resultados diferentes por desenho experimental.
4. CSV, JSON e DAT batem numericamente.
5. Texto da tese nao promete componente ausente no binario final.
