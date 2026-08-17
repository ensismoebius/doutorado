# For software/efficient_nn_lab

(nada pendente)

## Resolvido

- ~~Na seção de backpropagation, acrescentar uma tela demonstrando que redes
  neurais são, em princípio, multiplicação de vetores e matrizes, fazendo o
  mapeamento visual entre o grafo da rede (pesos, entradas, saídas, função de
  ativação, função de erro) e os vetores/matrizes correspondentes, com a
  animação mostrando com valores numéricos como o forward e o backward
  funcionam através desses vetores e matrizes.~~
- ~~E mostrar a regra da cadeia aplicada a essa rede.~~

Entregue como um item novo na seção "Backpropagation":
`src/efficient_nn_lab/backprop/demos/matrix_algebra.py`
(`MatrixAlgebraDemo`, slug `backprop.matrix`), renderizado por
`NeuronView._render_matrix_algebra`. 46 passos, um por operação escalar --
cada termo, cada soma, cada sigmoide, cada derivada local e cada célula de
gradiente têm passo próprio, invariante travada pelos testes
`test_matrix_demo_reveals_at_most_one_new_value_per_step` e
`test_matrix_demo_reveals_every_value_exactly_once` em
`tests/test_backprop.py`.
