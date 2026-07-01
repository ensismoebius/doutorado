# Adiadas / rejeitadas
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

Pag 20 se tornou 24
Figura 14 virou 18

52. Padronizar a apresentação de variáveis antes das equações.
53. Revisar sistematicamente e enriquecer a Wiki do projeto.
54. Incluir exemplos numéricos completos.
55. Demonstrar passo a passo a derivação das expressões matemáticas.
56. Avaliar comparativamente diferentes arquiteturas de autoencoders utilizando Engenharia Paraconsistente de Características.
57. Verificar e fundamentar a afirmação sobre taxas de aprendizado para parâmetros biofísicos em SNNs.
58. Fundamentar a tabela de associação entre codificações e funções de perda.
59. Fundamentar teoricamente as estratégias de normalização utilizadas no projeto.
60. Expandir e fundamentar a seção “The No Spike Problem”.
61. Fundamentar e referenciar adequadamente a seção “Why this range”.
62. Expandir e fundamentar a seção “Profile Guided Optimization (PGO)”.