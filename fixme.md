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

# Contradições wiki x código

C12. Wiki (`Experiment05.md`:134,390) diz que `modality=fused` concatena os vetores de características de voz+EEG. Código (`E05FeatureExtraction.cpp::signal_for_modality`, ramo `else // "fused"`) na verdade escolhe áudio se presente, senão EEG — um único sinal, sem concatenação. Não é fusão precoce (fundir sinal bruto antes do autoencoder/handcrafted) nem fusão tardia (concatenar vetores de características depois) — nenhuma fusão ocorre hoje.
Action: 
(a) implementar fusão tardia real
(b) implementar fusão precoce
(c) a distinção fusão-precoce-vs-tardia deve ser discutida na tese/wiki como eixo experimental