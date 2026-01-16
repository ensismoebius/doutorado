# Guia de Contribuição / Contributing Guide

## Sobre este Repositório / About this Repository

Este é um repositório de pesquisa de doutorado. Embora seja principalmente um projeto individual de pesquisa, contribuições na forma de sugestões, correções de bugs e melhorias são bem-vindas.

*This is a PhD research repository. While it is primarily an individual research project, contributions in the form of suggestions, bug fixes, and improvements are welcome.*

## Como Contribuir / How to Contribute

### Reportar Problemas / Reporting Issues

Se você encontrar um bug ou tiver uma sugestão:

1. Verifique se já existe uma issue relacionada
2. Crie uma nova issue descrevendo:
   - O problema ou sugestão
   - Passos para reproduzir (se aplicável)
   - Comportamento esperado vs atual
   - Ambiente (OS, versão do compilador, etc.)

*If you find a bug or have a suggestion:*

1. *Check if a related issue already exists*
2. *Create a new issue describing:*
   - *The problem or suggestion*
   - *Steps to reproduce (if applicable)*
   - *Expected vs actual behavior*
   - *Environment (OS, compiler version, etc.)*

### Propor Melhorias / Proposing Improvements

Para melhorias no código:

1. Abra uma issue descrevendo a melhoria proposta
2. Aguarde feedback antes de implementar
3. Siga as convenções de código existentes

*For code improvements:*

1. *Open an issue describing the proposed improvement*
2. *Wait for feedback before implementing*
3. *Follow existing code conventions*

## Padrões de Código / Code Standards

### C++ (Framework NN)

- **Padrão:** C++20
- **Estilo:** Seguir `.clang-format` e `.clang-tidy` do projeto
- **Documentação:** Comentar lógica complexa
- **Testes:** Adicionar testes para novas funcionalidades

```bash
# Verificar formatação
cd software/nn
clang-format -i src/**/*.cpp include/**/*.hpp

# Executar análise estática
cmake --build build --target analysis-clang-tidy
cmake --build build --target analysis-cppcheck

# Executar testes
ctest --test-dir build --output-on-failure
```

### Python (Notebooks e Scripts)

- **Padrão:** Python 3.8+
- **Estilo:** PEP 8
- **Documentação:** Docstrings para funções públicas

## Processo de Revisão / Review Process

1. **Fork** o repositório
2. Crie um **branch** para sua feature (`git checkout -b feature/minha-feature`)
3. **Commit** suas mudanças com mensagens descritivas
4. **Push** para o branch (`git push origin feature/minha-feature`)
5. Abra um **Pull Request**

### Mensagens de Commit

Use mensagens claras e descritivas:

```
Add: Nova funcionalidade X
Fix: Correção do bug Y
Refactor: Reorganização do módulo Z
Docs: Atualização da documentação
Test: Adicionar testes para feature W
```

## Estrutura de Testes / Test Structure

Todos os novos códigos devem incluir testes apropriados:

- **Testes unitários:** Para funções e classes individuais
- **Testes de integração:** Para pipelines completos
- **Cobertura:** Manter ≥95% de cobertura de código

## Análise de Segurança / Security Analysis

Antes de submeter código:

1. Execute análise estática com `cppcheck` e `flawfinder`
2. Verifique por vazamentos de memória com `valgrind`
3. Execute testes com sanitizers (`-fsanitize=address`)

```bash
# Build com sanitizers
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fsanitize=undefined"
cmake --build build
```

## Documentação / Documentation

Ao adicionar novas funcionalidades:

- Atualize o README.md relevante
- Adicione comentários no código para lógica complexa
- Inclua exemplos de uso quando apropriado
- Atualize o TODO.md se for parte do pipeline experimental

## Ferramentas Recomendadas / Recommended Tools

### Para desenvolvimento C++:
- **Compilador:** Clang 15+ ou GCC 11+
- **Build:** CMake 3.20+
- **Editor:** VSCode com extensões C++
- **Debugger:** GDB ou LLDB
- **Profiler:** Valgrind, Callgrind

### Para análise Python:
- **Editor:** VSCode ou JupyterLab
- **Linter:** pylint, flake8
- **Formatter:** black

## Questões e Discussões / Questions and Discussions

Para questões gerais sobre a pesquisa ou uso do framework:

- Abra uma **Discussion** no GitHub
- Ou crie uma **Issue** com a tag `question`

## Código de Conduta / Code of Conduct

Este projeto segue princípios básicos de respeito e colaboração:

- Seja respeitoso com outros contribuidores
- Aceite críticas construtivas
- Foque no que é melhor para a pesquisa e comunidade
- Mantenha discussões técnicas e profissionais

*This project follows basic principles of respect and collaboration:*

- *Be respectful to other contributors*
- *Accept constructive criticism*
- *Focus on what's best for the research and community*
- *Keep discussions technical and professional*

## Licença / License

Ao contribuir, você concorda que suas contribuições estarão sob a mesma licença do projeto.

*By contributing, you agree that your contributions will be licensed under the same license as the project.*

## Contato / Contact

Para questões específicas sobre a pesquisa de doutorado, entre em contato com o autor através das informações no README.md principal.

*For specific questions about the PhD research, contact the author through the information in the main README.md.*

---

**Obrigado por contribuir! / Thank you for contributing!**
