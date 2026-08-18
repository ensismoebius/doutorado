# For software/efficient_nn_lab

Todos os itens abaixo estão resolvidos (2026-08-18). Cada um traz onde a
correção mora e qual teste impede a volta do problema.

- In \frametitle{A camada BitLinear}
  -- In \framesubtitle{Da camada Linear densa à ternária} - need to explain what are each item of the equation \gamma = \text{mean}(|W|), \qquad
			\widetilde{W} = \text{round}\big(\text{clip}(W/\gamma,\,-1,\,1)\big)

     **RESOLVIDO** — `slides/bitnetCamada.tex`, novo quadro 2 ("Lendo a
     equação termo a termo"): $W$, $|W|$, `mean`, $\gamma$, $W/\gamma$,
     `clip`, `round`, $\widetilde{W}$, um item cada, na ordem em que o
     dado passa por eles (não na ordem de leitura da fórmula), com a
     metáfora da **régua** para $\gamma$.
     De quebra: o slide chamava a quantização dos pesos de *absmax* sobre
     a fórmula `mean(|W|)`, que é *absmean*. Rótulo corrigido —
     `presentation.md` §1.5 já dizia "absmean real".

  -- In \framesubtitle{Exemplo numérico} - It is hard to understand what "\par As ativações $X$ recebem seu próprio tratamento, quantização \textbf{absmax} para INT8: $\widetilde{X} = \text{round}(X \cdot 127/\max(|X|))$, com \textit{clip} em $[-128,127]$." means

     **RESOLVIDO** — aquela frase comprimia três ideias numa linha (que há
     *duas* quantizações diferentes, o que `absmax` faz, e de onde vem o
     127). Virou o quadro 4 inteiro: por que as ativações **não** podem
     ser ternárias, o que é $\max(|X|)$, por que 127, exemplo numérico
     ($X=[0{,}5,-2,1] \Rightarrow \widetilde{X}=[32,-127,64]$, conferido
     em Python) e uma tabela contrastando as duas réguas — pesos
     *absmean* vs. ativações *absmax*, que é o par confundível aqui.

- \frametitle{Hardware e limitações}
  Define ASICS

  **RESOLVIDO** — `slides/bitnetHardwareLimitacoes.tex`: caixa no pé do
  slide, na mesma tela em que a sigla aparece, definindo ASIC
  (*Application-Specific Integrated Circuit*) **contra** a GPU (o par
  confundível) e nomeando o custo da troca: a função não muda depois de
  fabricada. O quadro passou a usar `[shrink]` (6,1%) para caber.

- In software:
  -- The SNN and ANN comparison fonts are too small.

     **RESOLVIDO** — `widgets/neuron_view.py`, `_cmp_geometry` +
     `_render_comparison_pipeline`. A causa não era só um número baixo: o
     canvas do Qt/matplotlib mantém o dpi e cresce em polegadas, então
     `fontsize=8` calibrado a 900x400 encolhe *relativamente* justo no
     projetor. Agora o corpo vem da geometria — o menor entre o que a
     altura permite e o que a largura de uma coluna permite (célula
     quebrada em no máximo 2 linhas) — e a altura sobrando vai para os
     espaços, para a tabela preencher a caixa. Medido no tamanho real da
     janela: ~22pt por célula (~3x). O rótulo da linha de saída, que era
     só "O", agora é "Saída".
     Testes: `tests/test_comparison_table_layout.py` (cresce com o canvas,
     nada se sobrepõe, preenche a caixa, toda célula cabe em 2 linhas).

  -- In "Modo palestra" there must have a button that bring the user to another item in the current section or, if the current is the last in section, bring the user to the first one of the next section.

     **RESOLVIDO** — botão **Próxima demo ▸** (atalho `N`) no alto da
     janela, visível nos dois modos: `app/main_window.py`
     (`_demo_order`, `_on_next_demo`, `_go_to_demo`). Próximo item da
     seção; do último item da seção, o primeiro da seguinte; do último de
     todos, volta ao primeiro (botão morto no fim do roteiro só apareceria
     ao vivo). A ordem deriva do mesmo dicionário que constrói a árvore, e
     a seleção da árvore acompanha. `N` e não `->`/`PageDown`: `->` já é
     "próximo passo dentro da demo" e apresentadores remotos mandam
     `PageDown` para passar slide.
     Testes: 11 novos em `tests/test_main_window.py` (ordem = árvore,
     travessia de seção, volta ao início, uma volta completa visita cada
     demo uma vez, sobrevive ao modo palestra, move a árvore, `N` funciona,
     `->` continua sendo passo).
