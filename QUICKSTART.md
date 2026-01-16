# Guia de Início Rápido / Quick Start Guide

Este guia fornece instruções passo a passo para começar a trabalhar com o repositório de pesquisa de doutorado.

*This guide provides step-by-step instructions to get started with the PhD research repository.*

## 🚀 Instalação Rápida / Quick Installation

### Método 1: Setup Automático / Automatic Setup

O jeito mais fácil de configurar o repositório é usar o script de setup:

*The easiest way to set up the repository is using the setup script:*

```bash
# Clone o repositório / Clone the repository
git clone <repository-url>
cd doutorado

# Execute o script de setup / Run the setup script
./setup.sh
```

O script irá:
- Verificar dependências do sistema
- Inicializar submódulos git (se houver)
- Oferecer compilar o framework C++
- Oferecer criar ambiente virtual Python
- Instalar pacotes Python necessários

*The script will:*
- *Check system dependencies*
- *Initialize git submodules (if any)*
- *Offer to build the C++ framework*
- *Offer to create Python virtual environment*
- *Install necessary Python packages*

### Método 2: Setup Manual / Manual Setup

Se preferir fazer manualmente:

*If you prefer to do it manually:*

#### 1. Clonar o Repositório / Clone the Repository

```bash
git clone <repository-url>
cd doutorado
```

#### 2. Compilar o Framework C++ / Build the C++ Framework

```bash
cd software/nn

# Configurar com CMake / Configure with CMake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Compilar / Build (use número apropriado de jobs / use appropriate number of jobs)
# Linux: nproc, macOS: sysctl -n hw.ncpu
cmake --build build -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)

# Executar testes (opcional) / Run tests (optional)
ctest --test-dir build --output-on-failure

cd ../..
```

#### 3. Configurar Ambiente Python / Setup Python Environment

```bash
# Criar ambiente virtual / Create virtual environment
python3 -m venv venv

# Ativar ambiente virtual / Activate virtual environment
source venv/bin/activate

# Instalar pacotes / Install packages
pip install --upgrade pip
pip install jupyter numpy scipy matplotlib torch
```

#### 4. Explorar o Repositório / Explore the Repository

```bash
# Abrir Jupyter / Open Jupyter
jupyter notebook notebooks/

# Ver documentação / View documentation
cd documentation/
```

## 📋 Pré-requisitos / Prerequisites

### Para o Framework C++:

#### Linux (Recomendado / Recommended)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential cmake clang libomp-dev

# Arch Linux
sudo pacman -S base-devel cmake clang openmp

# Fedora
sudo dnf install gcc-c++ cmake clang libomp-devel
```

#### Bibliotecas requeridas (instaladas automaticamente via CMake FetchContent):
- Eigen 3.4+
- Google Test
- FFTW 3.3+
- NFFT 3.3+
- MatIO
- YAML-CPP

### Para Análise Python:

```bash
# Instalar Python e pip
sudo apt-get install python3 python3-pip python3-venv

# Ou usando pyenv (recomendado para múltiplas versões)
curl https://pyenv.run | bash
pyenv install 3.8.0
pyenv global 3.8.0
```

## 🔍 Estrutura de Diretórios / Directory Structure

```
doutorado/
├── README.md              # Este arquivo / This file
├── CONTRIBUTING.md        # Guia de contribuição / Contributing guide
├── QUICKSTART.md         # Guia de início rápido / Quick start guide
├── setup.sh              # Script de configuração / Setup script
│
├── software/             # Código-fonte / Source code
│   ├── nn/              # Framework de redes neurais / Neural network framework
│   ├── signalAquirer/   # Aquisição de sinais / Signal acquisition
│   └── sampleRateMeasurer/
│
├── documentation/        # Documentação da pesquisa / Research documentation
├── hardware/            # Projetos de hardware / Hardware projects
└── notebooks/           # Jupyter notebooks
```

## 🎯 Primeiros Passos / First Steps

### 1. Explorar a Documentação / Explore Documentation

```bash
# Ver README principal / View main README
cat README.md

# Ver documentação do framework NN / View NN framework docs
cat software/nn/README.md

# Ver TODO da pesquisa / View research TODO
cat software/nn/TODO.md
```

### 2. Executar Exemplos / Run Examples

#### Framework C++:

```bash
cd software/nn/build

# Executar testes de tensor / Run tensor tests
./src/core/tensor/tests/tensor_gtest

# Executar testes de layers / Run layer tests
./src/core/layers/tests/layers_gtest

# Executar experimento / Run experiment
./bin/experiment_02
```

#### Jupyter Notebooks:

```bash
# Ativar ambiente virtual / Activate virtual environment
source venv/bin/activate

# Iniciar Jupyter / Start Jupyter
jupyter notebook notebooks/

# Abrir um notebook específico / Open a specific notebook
jupyter notebook notebooks/Phd_PreprocesingPipeline.ipynb
```

### 3. Verificar o Build / Verify Build

```bash
cd software/nn

# Executar análise estática / Run static analysis
cmake --build build --target analysis-cppcheck
cmake --build build --target analysis-clang-tidy

# Executar todos os testes / Run all tests (cross-platform)
ctest --test-dir build --output-on-failure -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)

# Gerar relatório de cobertura / Generate coverage report
cmake --build build --target coverage
```

## 🛠️ Tarefas Comuns / Common Tasks

### Compilar e Testar / Build and Test

```bash
# Rebuild completo / Complete rebuild
cd software/nn
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
ctest --test-dir build --output-on-failure
```

### Executar Análise de Performance / Run Performance Analysis

```bash
cd software/nn

# Profiling com Callgrind
cmake --build build --target profile_experiment_02

# Análise de memória com Valgrind
valgrind --leak-check=full ./build/bin/experiment_02
```

### Trabalhar com Notebooks / Working with Notebooks

```bash
# Iniciar servidor Jupyter / Start Jupyter server
source venv/bin/activate
jupyter notebook notebooks/

# Converter notebook para Python / Convert notebook to Python
jupyter nbconvert --to python notebooks/simpleNN.ipynb

# Executar notebook na linha de comando / Run notebook from command line
jupyter nbconvert --execute --to html notebooks/simpleNN.ipynb
```

### Atualizar o Repositório / Update Repository

```bash
# Atualizar do remote / Update from remote
git pull

# Atualizar submódulos / Update submodules
git submodule update --remote --recursive

# Recompilar após atualização / Rebuild after update
cd software/nn
cmake --build build
```

## 📚 Recursos Adicionais / Additional Resources

### Documentação / Documentation
- **README Principal:** Visão geral do projeto / Project overview
- **software/nn/README.md:** Framework C++ detalhado / Detailed C++ framework
- **software/nn/TODO.md:** Status das tarefas experimentais / Experimental task status
- **CONTRIBUTING.md:** Como contribuir / How to contribute

### Notebooks de Exemplo / Example Notebooks
- `Phd_PreprocesingPipeline.ipynb` - Pipeline de pré-processamento
- `lfcc_mgdf.ipynb` - Análise LFCC
- `simpleNN.ipynb` - Testes de redes neurais

### Ferramentas Úteis / Useful Tools
- **CMake:** https://cmake.org/
- **Eigen:** https://eigen.tuxfamily.org/
- **Google Test:** https://github.com/google/googletest
- **Jupyter:** https://jupyter.org/

## 🐛 Solução de Problemas / Troubleshooting

### Erro de Compilação / Build Error

```bash
# Limpar cache do CMake / Clear CMake cache
rm -rf software/nn/build
rm -rf software/nn/CMakeCache.txt

# Tentar novamente / Try again
cd software/nn
cmake -S . -B build
```

### Problemas com Python / Python Issues

```bash
# Recriar ambiente virtual / Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install jupyter numpy scipy matplotlib torch
```

### Testes Falhando / Tests Failing

```bash
# Executar teste específico com debug / Run specific test with debug
cd software/nn/build
./src/core/tensor/tests/tensor_gtest --gtest_filter=TensorTest.* --gtest_break_on_failure

# Verificar memória / Check memory
valgrind ./src/core/tensor/tests/tensor_gtest
```

## 💡 Dicas / Tips

1. **Use o script setup.sh** para configuração inicial rápida
   *Use setup.sh script for quick initial setup*

2. **Ative o ambiente virtual** antes de trabalhar com Python
   *Activate virtual environment before working with Python*

3. **Compile em modo Release** para melhor performance
   *Build in Release mode for better performance*

4. **Execute testes frequentemente** durante desenvolvimento
   *Run tests frequently during development*

5. **Consulte o TODO.md** para ver progresso da pesquisa
   *Check TODO.md to see research progress*

## 📞 Suporte / Support

Se você encontrar problemas:

1. Verifique este guia e a documentação
2. Procure em Issues existentes no GitHub
3. Crie uma nova Issue se necessário
4. Veja CONTRIBUTING.md para mais detalhes

*If you encounter problems:*

1. *Check this guide and documentation*
2. *Search existing GitHub Issues*
3. *Create a new Issue if necessary*
4. *See CONTRIBUTING.md for more details*

---

**Boa pesquisa! / Happy researching!**
