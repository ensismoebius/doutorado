# Doutorado - PhD Research Repository

Este repositório contém o trabalho de pesquisa de doutorado focado em identificação de locutor utilizando redes neurais artificiais espicantes (Spiking Neural Networks - SNNs) com dados sincronizados de EEG e áudio.

*This repository contains PhD research work focused on speaker identification using Spiking Neural Networks (SNNs) with synchronized EEG and audio data.*

## 📋 Visão Geral / Overview

O projeto desenvolve um pipeline abrangente para identificação de locutor que integra:
- **Redes Neurais Espicantes (SNNs)** para classificação
- **Sincronização EEG-Áudio** para análise multimodal
- **Análise Paraconsistente** para avaliação de características
- **Framework de Deep Learning em C++20** otimizado para performance

*The project develops a comprehensive pipeline for speaker identification that integrates:*
- *Spiking Neural Networks (SNNs) for classification*
- *EEG-Audio synchronization for multimodal analysis*
- *Paraconsistent analysis for feature evaluation*
- *C++20 Deep Learning framework optimized for performance*

## 🗂️ Estrutura do Repositório / Repository Structure

```
doutorado/
├── documentation/           # Documentação da pesquisa / Research documentation
│   ├── 00-dissertation/    # Tese e monografia / Dissertation and monograph
│   ├── 01-researchNotesAndFiles/  # Notas de pesquisa / Research notes
│   ├── 03-articlesToRead/  # Artigos para leitura / Articles to read
│   ├── 06-BooksToRead/     # Livros para leitura / Books to read
│   ├── 07-articlesProduced/ # Artigos produzidos / Produced articles
│   └── projeto_de_pesquisa.pdf  # Proposta de pesquisa / Research proposal
│
├── software/               # Implementações de software / Software implementations
│   ├── nn/                # Framework de redes neurais em C++20 / C++20 neural network framework
│   ├── signalAquirer/     # Sistema de aquisição de sinais / Signal acquisition system
│   └── sampleRateMeasurer/ # Medidor de taxa de amostragem / Sample rate measurer
│
├── hardware/              # Projetos de hardware / Hardware projects
│   ├── INA128/           # Circuito amplificador / Amplifier circuit
│   └── kicadParts/       # Componentes KiCad / KiCad components
│
└── notebooks/            # Jupyter notebooks para análise / Jupyter notebooks for analysis
    ├── Phd_PreprocesingPipeline.ipynb  # Pipeline de pré-processamento
    ├── lfcc_mgdf.ipynb                 # Análise LFCC
    └── simpleNN.ipynb                  # Testes de redes neurais
```

## 🚀 Componentes Principais / Main Components

### 1. Framework de Redes Neurais (C++20) / Neural Network Framework

Localização: `software/nn/`

Um framework de alto desempenho para redes neurais com suporte a:
- Spiking Neural Networks (SNNs) com neurônios Leaky Integrate-and-Fire
- Autoencoders esparsos para redução de dimensionalidade
- Sincronização EEG/Áudio com transformadas wavelet
- Otimização SIMD e paralelização OpenMP

*A high-performance neural network framework with support for:*
- *Spiking Neural Networks (SNNs) with Leaky Integrate-and-Fire neurons*
- *Sparse autoencoders for dimensionality reduction*
- *EEG/Audio synchronization with wavelet transforms*
- *SIMD optimization and OpenMP parallelization*

**Documentação completa:** Ver [software/nn/README.md](software/nn/README.md)

**Requisitos:**
- C++20 compatible compiler (Clang 15+ recomendado)
- CMake 3.20+
- OpenMP 5.1+
- Python 3.8+ (para utilitários)

**Build rápido:**
```bash
cd software/nn
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
ctest --test-dir build --output-on-failure
```

### 2. Pipeline de Identificação de Locutor / Speaker Identification Pipeline

O pipeline experimental está organizado em fases sistemáticas (ver `software/nn/TODO.md`):

**Fase 0:** Congelamento metodológico e infraestrutura
- Janela fixa: 1.5s com 50% de sobreposição
- Normalização obrigatória: [0,1]
- Classificador: Residual SNN

**Fase 1:** Engenharia clássica de características (Wavelets)
- Wavelet-Packet Transform (WPT)
- Energia de sub-bandas

**Fase 2:** Escalas espectrais (FASE CENTRAL)
- Comparação: LFCC × MEL × BARK
- Modalidades: Voz, EEG, Voz + EEG

**Fase 3:** Aprendizado de características (Autoencoders Espicantes)
- AE sub-completo, supra-completo, denoising

**Fase 4:** Modalidades (Unimodal × Multimodal)
- Avaliação da fusão EEG + voz

**Fase 5:** Fala imaginada (Diferencial da tese)
- Viabilidade biométrica de fala imaginada

**Fase 6:** Robustez a ruído
- Validação estrutural dos modelos baseados em SNN

**Fase 7:** Consolidação final
- Tabelas comparativas e plots paraconsistentes
- Comparação com estado da arte

### 3. Sistema de Aquisição de Sinais / Signal Acquisition System

Localização: `software/signalAquirer/`

Sistema para aquisição sincronizada de dados EEG e áudio.

### 4. Hardware de Aquisição / Acquisition Hardware

Localização: `hardware/`

Projetos de circuitos para amplificação e condicionamento de sinais EEG.

## 📊 Análise Paraconsistente / Paraconsistent Analysis

O projeto utiliza métricas paraconsistentes para avaliar a qualidade das características:
- **α** (similaridade intra-classe)
- **β** (sobreposição inter-classe)
- **G1, G2** (medidas de certeza e contradição)

*The project uses paraconsistent metrics to evaluate feature quality:*
- *α (intra-class similarity)*
- *β (inter-class overlap)*
- *G1, G2 (certainty and contradiction measures)*

## 🔬 Metodologia / Methodology

1. **Coleta de dados:** EEG + áudio sincronizados
2. **Pré-processamento:** Janelamento, normalização
3. **Extração de características:** Wavelets, LFCC, MEL, BARK, Autoencoders
4. **Classificação:** Residual SNN
5. **Análise:** Métricas paraconsistentes + métricas clássicas
6. **Comparação:** Estado da arte

## 📚 Documentação / Documentation

- **Proposta de pesquisa:** `documentation/projeto_de_pesquisa.pdf`
- **Dissertação:** `documentation/00-dissertation/`
- **Framework NN:** `software/nn/README.md`
- **TODO da pesquisa:** `software/nn/TODO.md`

## 🛠️ Requisitos do Sistema / System Requirements

### Para o framework C++:
- Linux (Arch Linux é o alvo primário)
- Clang 15+ ou GCC 11+
- CMake 3.20+
- OpenMP 5.1+
- Bibliotecas: Eigen, FFTW, NFFT, MatIO, YAML-CPP

### Para análise em Python:
- Python 3.8+
- Jupyter Notebook
- NumPy, SciPy, Matplotlib
- PyTorch (para alguns experimentos)

## 🚦 Início Rápido / Quick Start

### 1. Clonar o repositório / Clone the repository
```bash
git clone <repository-url>
cd doutorado
```

### 2. Compilar o framework / Build the framework
```bash
cd software/nn
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
```

### 3. Executar testes / Run tests
```bash
ctest --test-dir software/nn/build --output-on-failure
```

### 4. Explorar notebooks / Explore notebooks
```bash
jupyter notebook notebooks/
```

## 📈 Status do Projeto / Project Status

Este é um repositório ativo de pesquisa de doutorado. Consulte `software/nn/TODO.md` para o status atual das tarefas experimentais.

*This is an active PhD research repository. See `software/nn/TODO.md` for current experimental task status.*

## 🤝 Contribuindo / Contributing

Este é um projeto de pesquisa acadêmica. Para questões ou sugestões, por favor abra uma issue.

*This is an academic research project. For questions or suggestions, please open an issue.*

## 📄 Licença / License

[Especificar licença / Specify license]

## 👤 Autor / Author

André Pacheco Neves (ensismoebius)
Programa de Pós-Graduação em Ciência da Computação

## 📧 Contato / Contact

[Adicionar informações de contato / Add contact information]

---

**Nota:** Este repositório está em desenvolvimento ativo como parte de uma pesquisa de doutorado. A estrutura e o conteúdo podem mudar conforme a pesquisa progride.

*Note: This repository is under active development as part of a PhD research. Structure and content may change as research progresses.*
