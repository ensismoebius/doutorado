# ARCHITECTURE

## Diagrama de camadas (visão geral)

```
app  ────────────────► services ────────────────► domain ───────────────► core
  │                         │                         │                      │
  │                         └──────────────► utils ───┘                      │
  └──────────────────────────────────────────────────────────────────────────┘
                              infra (I/O, dispositivos, bindings externos)
```

**Regra principal:** dependências fluem de camadas externas para internas.
- `infra` e `app` **não** podem ser dependências diretas de `core` ou `domain`.
- `services` pode depender de `domain`, `core` e `utils`.
- `utils` contém funções puras (sem I/O) e pode ser usada por `services`/`domain`.

---

## Estrutura de pastas por responsabilidade

### Python (demo biometria por voz)
```
src/demos/pydemos/
├── app/          # CLI e pontos de entrada
├── core/         # estruturas de dados e configs (sem I/O)
├── domain/       # regras de negócio/políticas (puras)
├── services/     # pipelines e orquestração
│   └── modelos/  # modelos de processamento (SNN)
├── infra/        # I/O (áudio, microfone, plotting)
└── utils/        # funções auxiliares (processamento puro)
```

### C++ (mapeamento conceitual)
- `src/core/` → **core/services** (componentes, camadas de rede, datasets, stats)
- `src/utility/`, `src/wave`, `src/wavelet` → **utils/services** (processamento)
- `src/demos/` → **app** (entrypoints)
- `src/dataLoaders/` → **services/infra** (carregamento + I/O)

> Observação: a árvore C++ permanece funcionalmente inalterada nesta refatoração. O mapeamento acima define **responsabilidade** sem mover arquivos C++.

---

## Convenções de nomenclatura

- Módulos: `snake_case.py`.
- Pacotes por camada: `app`, `core`, `domain`, `services`, `infra`, `utils`.
- Funções puras devem ficar em `utils/` ou `domain/`.

---

## Regras de dependência

1. **core**: sem dependência de I/O nem bibliotecas externas de hardware.
2. **domain**: apenas lógica de negócio/políticas; depende somente de `core` e `utils`.
3. **services**: orquestração, pipelines e modelos; pode depender de `domain`, `core`, `utils`.
4. **infra**: I/O, hardware, filesystem, gráficos, drivers.
5. **app**: entrypoints; depende de `services` e `infra`.
6. **utils**: helpers puros (sem efeitos colaterais) para reuso geral.

---

## Mapa de migração (pydemos)

| Origem (antigo) | Destino (novo) | Justificativa | Impacto de dependências |
|---|---|---|---|
| `arquivo_audio.py` | `infra/arquivo_audio.py` | I/O de WAV é infraestrutura | `core` passa a fornecer `WavInfo` e `infra` usa `core` |
| `captura.py` | `infra/captura.py` | Acesso a microfone e reamostragem | Mantém dependências externas (sounddevice/scipy) isoladas |
| `visualizacao.py` | `infra/visualizacao.py` | Plotting é I/O | Mantém matplotlib em `infra` |
| `caracteristicas.py` | `utils/caracteristicas.py` | processamento puro WPT | Sem I/O; acesso externo (pywt) isolado em utils |
| `codificacao.py` | `utils/codificacao.py` | codificação Poisson é função pura | Sem I/O; isolado de services/app |
| `janelamento.py` | `utils/janelamento.py` | segmentação de sinais (puro) | Sem I/O |
| `ondaletas.py` | `utils/ondaletas.py` | cálculo de nível WPT (puro) | Sem I/O |
| `preprocessamento.py` | `utils/preprocessamento.py` | normalização (puro) | Sem I/O |
| `cadastro.py` | `services/cadastro.py` | pipeline de cadastro | `services` coordena `infra` |
| `conjunto_dados.py` | `services/conjunto_dados.py` | dataset e pipeline | `services` usa `infra` + `utils` |
| `identificacao_locutor.py` | `services/identificacao_locutor.py` | treino/inferência | `services` usa `domain`, `core`, `infra`, `utils` |
| `rede_snn.py` | `services/modelos/rede_snn.py` | modelo computacional | dependência de torch/snn fica em `services` |
| `comandos.py` | `app/comandos.py` | orquestração CLI | `app` chama `services` |
| `main.py` | `app/main.py` | entrypoint CLI | `app` apenas liga CLI

> **Compatibilidade**: wrappers foram removidos para evitar redundância. Imports devem apontar para os módulos nas camadas (app/services/infra/utils/core/domain).

---

## Mapa de dependências (pydemos)

- `app` → `services`, `infra`, `utils`, `core`
- `services` → `domain`, `core`, `utils`, `infra`
- `domain` → `core`
- `infra` → `core`
- `utils` → (sem dependências internas)
- `core` → (sem dependências internas)

---

## Regras para contribuições futuras

1. Novas estruturas de dados/configs devem ir em `core/`.
2. Regras de negócio (ex.: políticas de decisão) vão em `domain/`.
3. Lógica de orquestração/pipelines em `services/`.
4. I/O (arquivos, dispositivos, rede, plots) em `infra/`.
5. Funções puras e matemáticas em `utils/`.
6. `app/` deve permanecer mínimo (parsing, entrada, delegação).

---

## Conformidade

- Nenhuma assinatura pública foi alterada nas camadas (app/services/infra/utils/core/domain).
- Lógica de I/O isolada em `infra/`.
- Processamento puro isolado em `utils/`.
- Orquestração (treino/inferência/CLI) isolada em `services/` e `app/`.
