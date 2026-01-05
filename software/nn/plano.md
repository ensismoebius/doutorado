# Plano de Refatoração Incremental da Arquitetura do Projeto

Este documento detalha o plano de refatoração para reorganizar a estrutura do projeto `nn`, separando interfaces públicas (`include/`) de implementações (`src/`) e headers privados, visando melhor escalabilidade, tempo de build e análise semântica.

## Objetivos
1.  **Separação de Responsabilidades**: `include/nn/<modulo>` (público) vs `src/core/<modulo>` (privado/implementação).
2.  **Higiene de Includes**: Uso de caminhos qualificados (`#include "nn/tensor/Tensor.hpp"`), forward declarations e remoção de includes desnecessários.
3.  **Compatibilidade**: Manter `CMakeLists.txt` e `compile_commands.json` atualizados para suporte a clangd.

## Estratégia Geral
A refatoração será realizada módulo por módulo, começando pelos mais fundamentais (sem dependências) e subindo na hierarquia.

**Para cada módulo:**
1.  Criar diretório `include/nn/<modulo>`.
2.  Mover headers públicos (`.hpp`) de `src/core/<modulo>` para `include/nn/<modulo>`.
3.  Manter headers privados (detalhes de implementação) em `src/core/<modulo>` (ou `src/core/<modulo>/internal` se necessário).
4.  Atualizar `CMakeLists.txt` do módulo para:
    *   Adicionar `include` como diretório de include público (`PUBLIC`).
    *   Manter `src/core/<modulo>` como diretório de include privado (`PRIVATE`) se houver headers privados.
5.  Atualizar `#include` nos arquivos `.cpp` do próprio módulo.
6.  Atualizar `#include` nos módulos dependentes (usando `grep` e `sed`).
7.  Compilar e rodar testes.

---

## Etapas da Refatoração

### Fase 1: Infraestrutura e Fundamentos

#### 1.1. Preparação
*   Criar estrutura de diretórios base: `include/nn`.
*   Atualizar `CMakeLists.txt` raiz para incluir `include` globalmente (opcional, mas recomendado que cada target defina seus includes).

#### 1.2. Módulo `tensor` (Fundamental)
*   **Ação**:
    *   Mover `Tensor.hpp`, `ITensorBackend.hpp`, `TensorBackendFactory.hpp` para `include/nn/tensor/`.
    *   Manter `EigenTensorBackend.hpp` em `src/core/tensor/` (privado).
*   **CMake**: Atualizar `src/core/tensor/CMakeLists.txt` para exportar `include`.
*   **Refatoração de Código**:
    *   Atualizar includes internos de `tensor`.
    *   Atualizar todos os arquivos do projeto que fazem `#include "Tensor.hpp"` para `#include "nn/tensor/Tensor.hpp"`.

#### 1.3. Módulo `utility` (Fundamental)
*   **Ação**:
    *   Mover `batching.hpp`, `printTensor.hpp`, `comparison.h`, `vectorizationCheck.hpp`, `synthetic_spike_data.hpp`, `EigenParallel.hpp`, `Normalization.hpp`, `Windowing.hpp` para `include/nn/utility/`.
    *   Avaliar se `imgui_glfw.hpp` é público (provavelmente sim, para demos).
*   **CMake**: Atualizar `src/core/utility/CMakeLists.txt`.
*   **Refatoração de Código**: Atualizar referências em todo o projeto.

### Fase 2: Matemática e Lógica Core

#### 2.1. Módulo `linearAlgebra`
*   **Ação**: Mover headers públicos para `include/nn/linearAlgebra/`.
*   **Refatoração**: Atualizar includes.

#### 2.2. Módulo `statistics`
*   **Ação**: Mover headers públicos para `include/nn/statistics/`.
*   **Refatoração**: Atualizar includes.

#### 2.3. Módulo `paraconsistent`
*   **Ação**: Mover headers públicos para `include/nn/paraconsistent/`.
*   **Refatoração**: Atualizar includes.

### Fase 3: Deep Learning Core (Acoplamento Alto)

#### 3.1. Módulo `initializers`
*   **Ação**: Mover headers para `include/nn/initializers/`.
*   **Refatoração**: Atualizar includes.

#### 3.2. Módulo `optimizers`
*   **Ação**: Mover headers para `include/nn/optimizers/`.
*   **Refatoração**: Atualizar includes.

#### 3.3. Módulo `layers`
*   **Ação**: Mover headers para `include/nn/layers/`.
*   **Nota**: Este módulo possui muitos arquivos. Verificar se há headers internos que não devem ser expostos.
*   **Refatoração**: Atualizar includes.

### Fase 4: Dados e IO

#### 4.1. Módulo `dataLoaders`
*   **Ação**: Mover headers para `include/nn/dataLoaders/`.
*   **Refatoração**: Atualizar includes.

#### 4.2. Módulo `saver`
*   **Ação**: Mover headers para `include/nn/saver/`.
*   **Refatoração**: Atualizar includes.

#### 4.3. Módulo `wave` e `wavelet`
*   **Ação**: Mover headers para `include/nn/wave/` e `include/nn/wavelet/`.
*   **Refatoração**: Atualizar includes.

### Fase 5: Limpeza e Verificação Final

*   **Verificação de Includes**: Garantir que nenhum arquivo em `include/` inclua arquivos de `src/`.
*   **Verificação de Dependências**: Garantir que `CMakeLists.txt` de cada módulo declare suas dependências corretamente (`target_link_libraries`).
*   **Sanity Check**: Rodar `cppcheck` e verificar se a análise melhorou (menos erros de include).
*   **Build Final**: Compilação limpa e execução de todos os testes.

---

## Critérios de Aceite por Etapa
*   O projeto deve compilar com `cmake --build build`.
*   Todos os testes (`ctest`) devem passar.
*   Não deve haver regressão em funcionalidades.
