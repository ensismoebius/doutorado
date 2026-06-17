
# Biometria por voz (demo) — WPT → codificação em spikes → SNN

## Como rodar este demo

> Note: runtime ingestion of `.npz` weight or shard files is disabled in this build. Python demos that produce `.npz` outputs (for offline analysis) still work, but any runtime loading of `.npz` artifacts is disabled — use `scripts/mat_to_npz.py` or other offline converters and treat `.npz` files as offline artifacts.


Entre no diretório `pydemos` e execute:

```bash
python -m app.main <comando> [opções]
```

Exemplo para rodar o demo visual:

```bash
python -m app.main demo --duracao 1.0 --saida-plot demo.png
```

Todos os comandos abaixo são subcomandos do `app.main`.

Para ver todas as opções e ajuda de cada comando, use:

```bash
python -m app.main -h
python -m app.main <comando> -h
```

### Subcomandos disponíveis

| Comando     | Descrição                                                     |
| ----------- | ------------------------------------------------------------- |
| demo        | Executa pipeline e gera plots didáticos                       |
| capturar    | Captura áudio e salva WAV para uma pessoa (cadastro)          |
| enrolar     | Alias de capturar (cadastrar amostras por pessoa)             |
| treinar     | Treina a SNN para classificar locutores                       |
| identificar | Identifica a pessoa por voz (microfone)                       |
| verificar   | Identifica e pode retornar 'desconhecido' (com limiar)        |
| avaliar     | Avalia o modelo em WAVs gravados e imprime matriz de confusão |

Cada comando aceita diversas opções (veja -h). Exemplos de uso estão abaixo e nas seções específicas.

> **Nota:** A profundidade da SNN (número de blocos residuais) pode ser ajustada no código ao criar o modelo, veja `rede_snn.py` e a seção "Arquitetura da SNN" neste README.

Este diretório contém um demo didático de **identificação de locutor** (biometria por voz) usando:

- Extração de características por **Wavelet Packet Transform (WPT)** (energia por banda)
- **Pré-processamento** (compressão log + normalização) para estabilidade
- **Codificação em spikes** (Poisson / rate coding) com taxa adaptativa
- **Spiking Neural Network (SNN)** com dinâmica temporal (vários passos por janela)
- Fluxo completo: **enrolar (capturar)** → **treinar** → **identificar/verificar** → **avaliar**

> Observação: é um demo “end-to-end” focado em clareza. Para biometria real em produção, normalmente há mais dados, controle de canal, VAD, normalizações globais e avaliação mais rigorosa.

---

## Estrutura esperada de dados

O treino e a avaliação esperam WAVs organizados assim:

```
<diretorio_base>/
  alice/
    amostra_*.wav
  bob/
    amostra_*.wav
  carol/
    amostra_*.wav
```

Por padrão, o diretório base é `dados/vozes`.

---

## Dependências

Este demo usa:

- Python 3.10+ (deve funcionar em versões recentes)
- `numpy`
- `torch`
- `snntorch`
- `pywavelets`
- `matplotlib`
- `sounddevice` (captura de áudio)
- `scipy` (opcional, melhora reamostragem; se não existir, cai para interpolação linear)

### Instalação via pip (exemplo)

```bash
pip install numpy torch snntorch pywavelets matplotlib sounddevice scipy
```

### Instalação via conda/mamba (exemplo)

```bash
conda install -c conda-forge numpy scipy matplotlib pywavelets sounddevice
pip install torch snntorch
```

> Em Linux, `sounddevice` depende de PortAudio/ALSA. Se der erro de dispositivo, veja a seção de troubleshooting.

---

## Comandos (CLI)

Todos os comandos abaixo são executados **dentro deste diretório**.


### Receita rápida (3 passos)

1) Enrole (cadastre) 2–5 amostras por pessoa:

```bash
python -m app.main enrolar --pessoa alice --duracao 3
python -m app.main enrolar --pessoa bob --duracao 3
```

2) Treine o classificador:

```bash
python -m app.main treinar --diretorio-dados dados/vozes --epocas 10
```

3) Identifique (microfone):

```bash
python -m app.main identificar --duracao 2
```

### Ajuda do CLI

Para ver todas as opções disponíveis:

```bash
python -m app.main -h
python -m app.main treinar -h
python -m app.main verificar -h
```

### Resumo dos subcomandos

| Subcomando    | Para que serve                                       | Saídas típicas                                    |
| ------------- | ---------------------------------------------------- | ------------------------------------------------- |
| `demo`        | roda pipeline e gera plots didáticos                 | `result_pipeline_wpt_snn.png` (ou `--saida-plot`) |
| `capturar`    | grava uma amostra WAV para uma pessoa                | `<base>/<pessoa>/amostra_*.wav`                   |
| `enrolar`     | alias de `capturar` (linguagem biométrica)           | `<base>/<pessoa>/amostra_*.wav`                   |
| `treinar`     | treina o classificador multiclasse                   | `modelo_snn_locutor.pt`, `rotulos_locutor.json`   |
| `identificar` | identifica a pessoa via microfone                    | imprime predição/confiança                        |
| `verificar`   | identifica e pode retornar `desconhecido` (limiar)   | imprime predição/confiança                        |
| `avaliar`     | avalia em WAVs gravados e imprime matriz de confusão | imprime acurácia + matriz                         |

### 1) Demo visual (pipeline + plot)

Captura áudio do microfone, extrai WPT, codifica em spikes e gera um PNG com os gráficos.

```bash
python -m app.main demo --duracao 1.0 --passos-por-janela 10 --saida-plot result_demo_biometria.png
```

Parâmetros úteis:
- `--duracao`: duração da gravação (segundos)
- `--taxa-amostragem`: taxa alvo (Hz)
- `--tamanho-janela` / `--tamanho-passo`: janelamento
- `--num-bandas`: dimensão do vetor por janela
- `--passos-por-janela`: passos internos (dinâmica da SNN dentro de cada janela)

### 2) Enrolar (cadastrar) amostras por pessoa

Captura áudio e salva WAV no diretório de dados.

```bash
python -m app.main enrolar --pessoa alice --duracao 3
python -m app.main enrolar --pessoa bob --duracao 3
```

O comando `capturar` é equivalente (alias):

```bash
python -m app.main capturar --pessoa alice --duracao 3
```

Recomendação prática:
- Grave **2 a 5 amostras** por pessoa (ou mais), em dias/ambientes diferentes se possível.

### 3) Treinar o classificador (multiclasse)

Treina uma SNN para classificar **janelas** por pessoa.

```bash
python -m app.main treinar --diretorio-dados dados/vozes --epocas 10 --passos-por-janela 10
```

Saídas geradas:
- `modelo_snn_locutor.pt`
- `rotulos_locutor.json`

Parâmetros úteis:
- `--epocas`: número de épocas
- `--lr`: taxa de aprendizado
- `--alvo-spikes-por-passo`: controla a densidade de spikes (autoajuste do codificador)

### 4) Identificar (microfone)

Carrega o modelo treinado e identifica quem está falando.

```bash
python -m app.main identificar --duracao 2
```

### 5) Verificar (com “desconhecido” via limiar)

Útil quando você quer rejeitar vozes fora do conjunto de pessoas treinadas.

```bash
python -m app.main verificar --duracao 2 --limiar 0.55
```

Interpretação:
- Se `confiança < limiar`, retorna `desconhecido`.

### 6) Avaliar offline em WAVs gravados

Roda o classificador em cada WAV em `<base>/<pessoa>/*.wav` e imprime acurácia e matriz de confusão.

```bash
python -m app.main avaliar --diretorio-dados dados/vozes
```

Para ver o resultado arquivo-a-arquivo:

```bash
python -m app.main avaliar --diretorio-dados dados/vozes --verbose
```

---

## Como o “spike coding” foi escolhido (resumo)

A entrada WPT (energia por banda) tem grande faixa dinâmica. O pipeline faz:

1) **Pré-processamento**: `log(1+E)` + normalização para `[0,1]`
2) **Codificação Poisson (rate coding)** com **taxa máxima adaptativa**
3) **Vários passos por janela** (`--passos-por-janela`), dando dinâmica temporal interna para a SNN

Esse combo tende a ser uma escolha robusta para **classificação** (locutor), mantendo o demo simples.

---

## Troubleshooting

### Erro de captura / sample rate
Alguns dispositivos não aceitam certas taxas. A captura tenta taxas comuns (48k/44.1k) e reamostra para a taxa desejada.

### Sem SciPy
Se `scipy` não estiver instalado, a reamostragem cai para interpolação linear (funciona, mas com qualidade inferior).

### Sem microfone (ou rodando em servidor)
- Use o comando `avaliar` com WAVs gravados.
- Para gerar WAVs, use `enrolar` em uma máquina com microfone.

---

## Arquivos principais

- `app/main.py`: CLI do demo
- `conjunto_dados.py`: leitura do conjunto de dados por pastas + extração de janelas
- `caracteristicas.py`: extração de energia WPT
- `preprocessamento.py`: compressão/normalização das características
- `codificacao.py`: codificador Poisson e taxa adaptativa
- `rede_snn.py`: modelo SNN profundo com blocos residuais (ResNet-style), suporta entrada em sequência `[T,B,F]` e configuração de profundidade
## Arquitetura da SNN (atualizado)

O modelo SNN deste demo agora suporta **profundidade arbitrária** e **blocos residuais** (estilo ResNet) usando snnTorch.

- Por padrão, a rede é composta por:
  - Camada de entrada (Linear + LIF)
  - Vários blocos residuais profundos (cada bloco: Linear → LIF → Linear → LIF + skip connection)
  - Camada de saída (Linear + LIF)

- O número de blocos residuais pode ser ajustado ao criar o modelo:

```python
from rede_snn import criar_modelo_snn
modelo = criar_modelo_snn(num_inputs=100, num_outputs=3, profundidade=5)  # 5 blocos residuais
```

- O forward aceita entrada 2D (um passo) ou 3D (sequência temporal), e o estado é explícito para permitir continuidade entre janelas.

Veja o código em `rede_snn.py` para detalhes e exemplos.

### Exemplos de `--profundidade`

Você pode controlar o número de blocos residuais (profundidade) via CLI usando `--profundidade`. Exemplos:

```bash
# Demo com profundidade 5
python -m app.main demo --profundidade 5 --duracao 1.0 --saida-plot demo.png

# Treino com profundidade 4 (salva profundidade em rotulos.json)
python -m app.main treinar --profundidade 4 --diretorio-dados dados/vozes --epocas 10

# Inferência forçando profundidade ao carregar modelo (override)
python -m app.main identificar --modelo modelo_snn_locutor.pt --rotulos rotulos_locutor.json --profundidade 3 --duracao 2
```

Nota: ao treinar, se você não passar `--profundidade`, a profundidade padrão do código será usada; a profundidade utilizada é salva no arquivo de rótulos (`rotulos.json`) e será lida automaticamente ao recarregar o modelo. Você também pode forçar um override com `--profundidade` na inferência.
- `identificacao_locutor.py`: treino e inferência (microfone e WAV)
- `visualizacao.py`: plots didáticos
- `captura.py`: captura e reamostragem
- `arquivo_audio.py`: leitura/escrita WAV (stdlib)
- `cadastro.py`: captura e salvamento de amostras WAV

---

## Próximos passos (sugestões)

Se você quiser evoluir isso para uma biometria mais séria:
- Agregação por áudio (voto/média por janela) e métricas EER/DET
- VAD (detecção de fala) para remover silêncio
- Normalização global por corpus (em vez de por janela)
- Mais dados por pessoa e divisão treino/validação/teste
