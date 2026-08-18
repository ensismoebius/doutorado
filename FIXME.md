# For software/efficient_nn_lab

- In \frametitle{A camada BitLinear}
  In \framesubtitle{Da camada Linear densa à ternária} - need to explain what are each item of the equation \gamma = \text{mean}(|W|), \qquad
			\widetilde{W} = \text{round}\big(\text{clip}(W/\gamma,\,-1,\,1)\big)

  In \framesubtitle{Exemplo numérico} - It is hard to understand what "\par As ativações $X$ recebem seu próprio tratamento, quantização \textbf{absmax} para INT8: $\widetilde{X} = \text{round}(X \cdot 127/\max(|X|))$, com \textit{clip} em $[-128,127]$." means

- \frametitle{Hardware e limitações}
  Define ASICS

- In software:
  The SNN and ANN comparison fonts are too small
  
