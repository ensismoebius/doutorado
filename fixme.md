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

# Next changes
- If memory is avaiable run at least 2 profiles in paralel.
- Im seeing that you are going to test only ae profile with 4 layers, but im pretty sure that SOTA for low end devices like raspberry pi b can use more layers. m i wrong ?
- Check if snn torch has support for 3d tensors and, if positive, copy the implementation into our own.
- Then update the thesis.


# Contradições wiki x código

C12. Wiki (`Experiment05.md`:134,390) diz que `modality=fused` concatena os vetores de características de voz+EEG. Código (`E05FeatureExtraction.cpp::signal_for_modality`, ramo `else // "fused"`) na verdade escolhe áudio se presente, senão EEG — um único sinal, sem concatenação. Não é fusão precoce (fundir sinal bruto antes do autoencoder/handcrafted) nem fusão tardia (concatenar vetores de características depois) — nenhuma fusão ocorre hoje.
Action: 
(a) implementar fusão tardia real
(b) implementar fusão precoce
(c) a distinção fusão-precoce-vs-tardia deve ser discutida na tese/wiki como eixo experimental

RESOLVIDO (2026-07-03):
(a) Fusão tardia implementada. `extract_features` com `modality=fused, fusion_mode=late` (padrão) extrai voz e EEG independentemente (cada um na taxa nativa, 44100/1024 Hz) e concatena os vetores por amostra `[voz ‖ eeg]`. `E05FeatureExtraction.cpp`.
(b) Fusão precoce implementada. `fusion_mode=early` concatena os sinais brutos (voz seguida de EEG) e extrai numa única passagem, usando a taxa da voz. Novo campo `E05Config::Dataset::fusion_mode` (validado early/late).
(c) Eixo discutido na tese (cap. 08, itemize fusão precoce/tardia) e na wiki (`Experiment05.md`, tabela fusion_mode). README + schema do perfil atualizados; perfis `*-fused.json` agora declaram `fusion_mode: late` explicitamente.
    Testes: 3 novos (E05Fusion.*) — dimensão tardia = voz+eeg, precoce difere e rotula distinto, modo inválido lança. Suítes e05 verdes (feat 26, profile 56, classifiers 16). Tese compila (101 pág).
    Bug original corrigido de passagem: `signal_for_modality` (fallback áudio-senão-EEG, sem fusão) foi substituído; `modality_sample_rate` removido em favor de constantes por sinal `kVoiceSampleRate`/`kEegSampleRate`.