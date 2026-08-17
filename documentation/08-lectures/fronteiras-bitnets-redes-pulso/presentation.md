# Plano de integração — slides × Efficient Neural Networks Lab

Palestra: "As fronteiras da arquitetura de redes neurais: BitNets e redes de
pulso", `documentation/08-lectures/fronteiras-bitnets-redes-pulso/`.
Software: `software/efficient_nn_lab/`. Objetivo deste documento: um plano
executável, com diffs concretos, para (1) unificar a identidade visual dos
slides com a do software, (2) mapear cada seção dos slides ao trecho exato
do software que demonstra a mesma ideia, e (3) tornar essa transição
clicável a partir do PDF.

**Status**: Fases 1, 3 e 4 estão implementadas (cores/fonte unificadas,
CLI `--demo`/slugs, `abrir-demo.sh`, links nos 8 pontos da seção 3, todos
recompilados e verificados — ver checklists marcados abaixo). O que resta
é exclusivamente o **ensaio ao vivo** (seção 6): testar os links `run:` no
notebook/visualizador reais do dia, e decidir o modo de operação da seção
4.2. Fase 2 (o mapeamento em si) sempre foi só a tabela da seção 3, já
completa desde a primeira versão deste documento.

---

## 1. Estado atual (diagnóstico)

### 1.1 Cores — já convergem em *estrutura*, divergem em *intensidade*

`preamble.tex` já importa a paleta Okabe-Ito da tese e define os mesmos
quatro papéis semânticos que o software usa:

```tex
% preamble.tex, linhas 87-91
\colorlet{bitnetColor}{cbBlue}
\colorlet{snnColor}{cbVermillion}
\colorlet{convergeColor}{cbGreen}
\colorlet{neutralColor}{cbGrey}
```

O software (`app/theme.py`) usa os **mesmos quatro papéis semânticos**
(`BITNET_COLOR`, `SNN_COLOR`, `CONVERGE_COLOR`, `NEUTRAL_COLOR`), mas com
tons deliberadamente mais vívidos — o próprio docstring de `theme.py` já
documenta a intenção ("mirrors the semantic roles used in the LaTeX slide
deck... pushed brighter/more saturated"). O problema: essa promessa nunca
foi cumprida do lado do LaTeX. Os slides ainda usam os tons **originais e
mais dessaturados** da tese (pensados para impressão em papel, contraste
AA/AAA em texto pequeno), não os tons vívidos otimizados para projeção que
o software já usa.

| Papel | Slides hoje (`cbXxx` da tese) | RGB | Software (`theme.py`) | RGB |
|---|---|---|---|---|
| BitNet | `cbBlue` | `(0,114,178)` | `BITNET_COLOR` | `(0,144,255)` |
| SNN | `cbVermillion` | `(213,94,0)` | `SNN_COLOR` | `(255,90,54)` |
| Convergência | `cbGreen` | `(0,158,115)` | `CONVERGE_COLOR` | `(0,195,137)` |
| Neutro | `cbGrey` | `(166,166,166)` | `NEUTRAL_COLOR` | `(107,114,128)` |
| — (não existe ainda) | — | — | `ACCENT_COLOR` (destaques/glow) | `(255,176,0)` |

Lado a lado, hoje, os slides parecem "página de livro" e o software parece
"aplicativo" — a plateia vai notar a troca de paleta a cada vez que a
palestra alternar entre os dois. É o item de maior impacto visual e o mais
barato de corrigir (edição de 6 linhas em um único arquivo).

### 1.2 Fontes — nunca foram alinhadas

- Slides: compilados com `pdflatex` (confirmado em `apresentacao.log`),
  sem `\usepackage{lmodern}` nem tema de fonte — usa a fonte sans padrão do
  Beamer/Computer Modern.
- Software: nenhuma `font-family` é fixada em `app/theme.py` (só
  `font-size`) — o Qt usa a fonte sans padrão do sistema operacional (varia
  por máquina). Os gráficos matplotlib (todos os widgets de
  `widgets/*.py`) usam a fonte padrão do matplotlib, **DejaVu Sans**
  (empacotada com a própria biblioteca, portanto sempre presente no venv).
- **DejaVu Sans já está instalada no sistema** (`fc-list` confirma
  `/usr/share/fonts/TTF/DejaVuSans*.ttf`), e o pacote LaTeX `dejavu`
  (Type 1-convertido, compatível com `pdflatex` puro — não precisa de
  xelatex/lualatex) também já está instalado (`tlmgr info dejavu`) — não é
  preciso instalar nada para alinhar os três.

### 1.3 Estilo geral

- Software: fundo branco (`#FFFFFF`), painéis em cinza muito claro
  (`#F4F6F9`), texto quase-preto (`#111318`), cantos arredondados nos
  botões, caixas com cantos arredondados e "glow" nos diagramas
  (`neuron_view.py`/`weight_view.py`).
- Slides: fundo padrão do Beamer (branco), `beamercolorbox`s com
  `rounded=true,shadow=true` — **já usa cantos arredondados e sombra**,
  visualmente compatível com as caixas do software. Este ponto já está
  bem alinhado; não precisa de mudança estrutural, só a paleta de cores
  (1.1).

### 1.4 Mapeamento de conteúdo — o que já existe, o que falta

A palestra e o software cobrem o mesmo material quase 1:1 (mesmo autor,
mesma tese) — a estrutura de seções dos slides já segue a mesma progressão
pedagógica do software (motivação → mecanismo → problema do gradiente →
resultado → convergência). O que falta é **tornar essa correspondência
explícita e navegável** durante a apresentação ao vivo. É o assunto da
seção 3.

### 1.5 Lacuna de precisão a verbalizar (não corrigir)

`bitnetCamada.tex` ensina a quantização **absmean real** do BitNet b1.58
(escala γ = média(|W|) por tensor). O software's `BitNet -> Quantização` /
`BitNet -> Forward` usam, deliberadamente, um **limiar fixo e simétrico**
— uma simplificação didática documentada no próprio
`software/efficient_nn_lab/README.md` ("Precisão científica"). Isso é
**intencional e não deve ser "corrigido"** (mudar o software quebraria a
progressão didática de outras demos), mas o apresentador deve dizer em voz
alta, ao abrir a demo, algo como: *"o software simplifica o limiar para
ser fixo — na prática o BitNet usa uma escala por tensor, que vocês
acabaram de ver no slide"*. Ver checklist de ensaio (seção 6).

---

## 2. Fase 1 — Unificação visual (cores e fontes)

### 2.1 Cores vívidas nos slides

Substituir o bloco de `\colorlet` em `preamble.tex` (linhas 87-91) por
`\definecolor` com os hexadecimais exatos do software, preservando os
mesmos nomes semânticos (nenhum slide precisa mudar, só o preamble):

```tex
% Semantic roles for this talk's two "frontiers" -- exact hex values from
% software/efficient_nn_lab/src/efficient_nn_lab/app/theme.py, so slides
% and the live software read as one visual system, not two.
\definecolor{bitnetColor}{HTML}{0090FF}
\definecolor{snnColor}{HTML}{FF5A36}
\definecolor{convergeColor}{HTML}{00C389}
\definecolor{neutralColor}{HTML}{6B7280}
\definecolor{accentColor}{HTML}{FFB000}
```

Notas:
- `accentColor` é novo (não existe hoje no preamble) — usar em
  `\alert{...}`/`\colorbox` no lugar do amarelo padrão do Beamer, para
  igualar o "glow" âmbar que o software usa em destaques
  (`ACCENT_COLOR`).
- **Texto sobre `accentColor` deve ser escuro**, não branco — o próprio
  `theme.py` já resolveu esse problema de contraste (`QPushButton:checked`
  usa `color: #1A1200` sobre o amarelo, não branco). Replicar: qualquer
  `\colorbox{accentColor}{...}` ou `beamercolorbox` com esse fundo deve
  forçar texto escuro (`\color{black}` ou similar).
- Isto sai deliberadamente da paleta estritamente calibrada para
  contraste AA/AAA em papel (`visualStyle.tex`, usada pela tese impressa).
  Correto para esta talk especificamente (projeção, não impressão) —
  **não propagar esta mudança de volta para `visualStyle.tex`** nem para
  os capítulos da tese; é um arquivo isolado (`preamble.tex` desta pasta).
- `\setbeamercolor{title}{parent=structure,bg=cyan}` /
  `\setbeamercolor{frametitle}{parent=structure,bg=cyan}` (linhas 68-69)
  usam `cyan` cru do xcolor, não uma cor da paleta — trocar por
  `bg=bitnetColor!85!white` (ou similar) para os títulos também caírem
  dentro da paleta unificada em vez de um ciano dissonante.

### 2.2 Fonte compartilhada (DejaVu Sans nas três superfícies)

DejaVu Sans já é a fonte real dos gráficos matplotlib do software (padrão
da biblioteca) e já está instalada no sistema. Alinhar as outras duas
superfícies a ela em vez de escolher uma fonte nova:

**a) Slides — pacote `dejavu`, sem trocar de motor de compilação**

`fontspec` (a via "óbvia" para trocar de fonte em LaTeX) **exige**
xelatex/lualatex — quebra sob `pdflatex` com um erro fatal
(`fontspec requires XeTeX or LuaTeX`). Como o fluxo de trabalho já
estabelecido usa `pdflatex`, a via correta é o pacote `dejavu` (fontes
DejaVu convertidas para Type 1, compatíveis com `pdflatex` puro — já
instalado, `tlmgr info dejavu` confirma):

```tex
% preamble.tex, mantendo \usepackage[utf8]{inputenc} como estava
\usepackage{DejaVuSans}
```

- Continua compilando com `pdflatex apresentacao.tex`, exatamente como
  antes — nenhuma mudança de motor, nenhum risco de compatibilidade com
  `abntex2cite`/`backref`/`babel`.
- `\usepackage{DejaVuSans}` só redefine `\sfdefault`; como o Beamer já
  roteia o texto do corpo por `\sfdefault` por padrão, isso já é
  suficiente — não precisa de `\renewcommand{\familydefault}{\sfdefault}`.
- **Armadilha real encontrada e corrigida**: combinar Beamer + esse
  pacote de fonte faz o caractere `>` (ASCII simples, fora de modo
  matemático) renderizar como um glifo errado (`¿`) — comprovado com
  `pdftotext`, não é só cosmético. Nenhum `->` ASCII pré-existia no deck
  (todo `\rightarrow` já usava modo matemático); os únicos ASCII `->`
  eram os novos links da seção 4. Corrigido trocando por
  `\textrightarrow{}` no texto visível de cada
  `\abrirNoSoftware{...}`. **Nunca usar `->`/`<-` ASCII cru em texto
  (fora de `$...$`) neste deck** — usar `\textrightarrow{}` /
  `\textleftarrow{}`, ou `$\rightarrow$` dentro de matemática.
- Verificar sempre com `pdftotext -layout apresentacao.pdf - | grep '¿'`
  depois de qualquer edição que adicione texto novo.

**b) Software — fixar `font-family` explicitamente**

Hoje `app/theme.py` não fixa fonte alguma (fica ao sabor do SO). Adicionar
ao `STYLESHEET`:

```python
QWidget {
    color: {TEXT_COLOR};
    font-family: "DejaVu Sans";
    font-size: 11pt;
}
```

matplotlib já usa DejaVu Sans por padrão — nenhuma mudança necessária em
`widgets/*.py`, mas vale adicionar um comentário em `theme.py` apontando
essa dependência implícita, para não ser "corrigida" por engano no futuro.

### 2.3 Checklist da Fase 1

- [x] Editar `preamble.tex`: `\definecolor` vívido (2.1) + `accentColor`.
- [x] Editar `preamble.tex`: `title`/`frametitle` bg de `cyan` para
      `bitnetColor`.
- [x] Adicionar `\usepackage{DejaVuSans}` (pacote `dejavu`, compatível com
      `pdflatex`); `inputenc` mantido como estava (não usa fontspec).
- [x] Recompilar com `pdflatex` (2 passadas, `.bbl` já existente reusado)
      e comparar visualmente com o PDF anterior. Sem erros; a deck ganhou
      +1 página (43 -> 44, referências quebram em uma página a mais por
      causa da fonte mais larga) — sem perda de conteúdo. Duas páginas
      pré-existentes com "Overfull vbox" (já presentes antes desta
      mudança) ficaram marginalmente mais apertadas com a fonte mais
      larga; não chegam a cortar texto. Um bug real de renderização foi
      encontrado e corrigido nesse processo — ver a "armadilha real" na
      seção 2.2.
- [x] Editar `app/theme.py`: adicionar `font-family: "DejaVu Sans"`.
- [x] Rodar a suíte de testes do software (`QT_QPA_PLATFORM=offscreen
      pytest -q`) — 143 passed, sem regressão.

---

## 3. Fase 2 — Mapeamento slide → demo (o conteúdo do plano)

Tabela completa, na ordem em que os slides aparecem em `apresentacao.tex`.
"Demo" usa o `title` exato mostrado na árvore lateral do software; "Passo"
aponta o checkpoint específico a deixar em tela (não "abra a demo e
navegue à vontade" — cada entrada é o quadro exato que ilustra aquele
slide, para o apresentador não perder tempo ao vivo procurando o passo
certo).

### Fundamentos

Seção de abertura, criada depois da primeira versão deste plano: estabelece
arquitetura, peso, ciclo de treino e a leitura matricial antes de qualquer
menção a BitNet ou pulso. Todos os números dos slides são os mesmos das
demos correspondentes.

| Slide (arquivo) | Ideia-chave | Demo do software | Passo/checkpoint | Slug |
|---|---|---|---|---|
| `fundamentosArquitetura.tex` (diagrama 3-2-2-1) | Camadas, pesos, ativação não linear | **Backprop -> Rede de 4 camadas** | Percorrer o forward neurônio a neurônio, na mesma topologia do diagrama | `backprop.mlp` |
| `fundamentosPesos.tex` (`z = Σ wᵢxᵢ`, exemplo `z=-0,05 → y=0,4875`) | O peso é o parâmetro que se aprende; entra na combinação linear | **Backprop -> Rede de 4 camadas** | Passo "Forward: L1-A" — mostra este mesmo `z=-0,05 → y=0,4875` | `backprop.mlp` |
| `fundamentosTreinamento.tex` (ciclo forward/perda/backward/update) | A regra da cadeia dá `∂L/∂w`; o peso anda contra o gradiente | **Backprop -> Forward e backward clássicos** | Os 9 estágios da iteração 1 reproduzem exatamente os números do slide | `backprop.classic` |
| `fundamentosMatrizes.tex` (`z = Wx`, exemplo `2→2→1`) | A camada inteira é uma multiplicação matriz-vetor; `W[i,j]` **é** a seta do grafo | **Backprop -> A rede como matrizes** | Fase do mapeamento (passos "w11: a seta x1 → H1" …) e o forward termo a termo | `backprop.matrix` |
| `fundamentosCadeia.tex` (`∂L/∂y = Wᵀ ∂L/∂z`, cadeia de 5 fatores) | Backward = mesma matriz transposta; multiplicar matrizes é a cadeia em lote | **Backprop -> A rede como matrizes** | Fase backward e os 5 fatores da cadeia; terminar na conferência contra `grad_W1[H1,x1]` | `backprop.matrix` |

Nota de coerência — notação de derivada: slides e software usam **a mesma**
convenção, e `app/math_render.py` foi estendido para sustentá-la:

- **`∂`** para todo gradiente da perda (`∂L/∂w`, `∂L/∂y`, `∂L/∂z`), porque
  `L` depende de muitas variáveis — é a notação de `fundamentosTreinamento.tex`
  e `fundamentosCadeia.tex`;
- **`d`** para derivada de função de uma variável em relação ao seu único
  argumento: `dV/dt` (EDO da membrana, como em `snnLif.tex`), `dQ/dw`, `dS/dv`.

`_DERIVATIVE_RE` aceita os dois operadores e preserva o que o autor escreveu,
em vez de normalizar — a escolha entre `∂` e `d` é uma afirmação matemática,
não um detalhe de formatação.

### Parte I — BitNets

| Slide (arquivo) | Ideia-chave | Demo do software | Passo/checkpoint | Slug |
|---|---|---|---|---|
| `bitnetCamada.tex`, quadro 2 (exemplo numérico, `w=[0.6,-0.3,0.05]`) | Um peso some (vira 0) se cair na zona morta; os outros viram ±1 | **BitNet -> Quantização** | Mova o slider `w` até perto de `0` para reproduzir a zona morta ao vivo, depois até `0.6`/`-0.3` | `bitnet.quant` |
| `bitnetCamada.tex`, quadro 3 (multiplicação vira soma/subtração) | `W̃·X̃` não faz multiplicação real | **BitNet -> Forward** | Checkpoints "Quantizar w1"/"Quantizar w2" → "Multiplicação 1"/"Multiplicação 2" (mostra Q(w2)=0 anulando x2) | `bitnet.forward` |
| `bitnetTreinamento.tex` (STE) | Backward "finge" que a quantização foi identidade | **BitNet -> Backward -> STE** | Checkpoint final (derivada real vs. constante 1 do STE) | `bitnet.ste` |
| — (transição, sem slide dedicado) | Ciclo completo forward→loss→STE→update, peso oculto 0,80→0,84 | **BitNet -> Exemplo guiado** | Sequência fixa completa (10 passos) — usar como *demo bônus* se sobrar tempo após `bitnetTreinamento.tex` | `bitnet.guided` |

### Parte II — Redes de pulso

| Slide (arquivo) | Ideia-chave | Demo do software | Passo/checkpoint | Slug |
|---|---|---|---|---|
| `snnConfusable.tex` ("codificado em taxa e/ou tempo de disparo") | RNA é contínua/síncrona; RNP é pulso esparso no tempo | **SNN -> Sinal e spikes** | Deixe rodar (Play) uma vez — mostra o cruzamento de nível virando spike discreto | `snn.spikes` |
| `snnConfusable.tex` / `snnEsparsidade.tex` (codificação por **taxa**) | Informação também pode estar na *probabilidade* de disparo, não só no tempo exato | **SNN -> Codificação Poisson** | Compare com a demo anterior: mesmo sinal, spikes probabilísticos | `snn.poisson` |
| `snnEsparsidade.tex`, quadro 1 (maioria dos neurônios fica quieta) | Esparsidade = poucos "uns" numa maioria de "zeros" — imagem real, memorável | **SNN -> Codificação Poisson (imagem)** | `t≈15/29`, aponte que cada pixel é um "neurônio" independente sorteando disparo | `snn.poisson_image` |
| `snnLif.tex`, quadros 1-2 (RC, decaimento) | Circuito RC, vazamento exponencial sem estímulo | **SNN -> LIF** | Ajuste `tau`/corrente a zero para reproduzir o decaimento puro do quadro 2 antes de religar a corrente | `snn.lif` |
| `snnLif.tex`, quadro 3 (imagem estática `membranePotentialFull.png`) | Comportamento completo: integra, dispara, reseta | **SNN -> LIF** | Play com os parâmetros padrão — **substitui a imagem estática do slide por uma simulação ao vivo, com os mesmos números do slide se possível** (ver 3.1) | `snn.lif` |
| `snnTreinamento.tex` + `snnBptt.tex` (gradiente substituto, "mesma lógica do STE") | Forward usa degrau real; backward usa curva suave — igual ao BitNet | **SNN -> Surrogate gradient** | Checkpoints "A derivada real" → "O gradiente substituto" (a animação de *sweep* que desenha gradiente e sigmoide juntos, mostrando de onde vem o gradiente) | `snn.surrogate` |

### Convergência

| Slide (arquivo) | Ideia-chave | Demo do software | Passo/checkpoint | Slug |
|---|---|---|---|---|
| `convergenciaQuadrante.tex` (quadrantes: RNA / BitNet / RNP / RNP quantizada) | As duas fronteiras são eixos ortogonais que podem se combinar | **Comparação -> ANN x BitNet x SNN** | Último checkpoint (tabela completa, todas as linhas reveladas) | `comparison` |
| `convergenciaTema.tex` (STE = gradiente substituto, "o mesmo truque") | As duas subáreas chegaram à mesma solução independentemente | (nenhuma nova — referência cruzada verbal a `bitnet.ste` + `snn.surrogate` já mostrados) | — | — |

### Sem correspondência direta no software (não precisam de link)

`estrutura.tex`, `motivacao.tex`, `bitnetLinha.tex` (linha do tempo),
`bitnetProblema.tex` (PTQ vs. QAT), `bitnetResultados.tex` (tabela de
resultados do paper), `bitnetHardwareLimitacoes.tex`,
`snnMotivacao.tex`, `snnHardware.tex`, `snnAplicacoes.tex`,
`desafios.tex`, `conclusao.tex`. Estes são conteúdo histórico, resultados
publicados ou discussão de hardware — o software não reproduz benchmarks
nem hardware real, o que é esperado e já está declarado no "Escopo
negativo" do `README.md` do software.

### Não coberto pelo software hoje (lacuna real, ação opcional)

- **`bitnetProblema.tex`** (PTQ vs. QAT) não tem demo correspondente —
  fora de escopo para esta palestra (adicionar uma demo nova não é
  prioridade dado o prazo de uma semana).
- ~~**`Backprop -> Forward e backward clássicos`** e
  **`Backprop -> Rede de 4 camadas`** não têm slide dedicado~~ —
  **resolvido**: a seção "Fundamentos" foi criada (ver a tabela acima) e
  hoje os três demos do grupo "Backpropagation" têm slide de conteúdo e
  `\DemoSlide` próprios em `apresentacao.tex`. A recomendação original
  (abrir `backprop.classic` na Parte II, depois de `snnMotivacao.tex`)
  continua válida como *reforço* opcional — ver 3.1 —, mas já não é a
  única entrada desses demos na palestra.

### 3.1 Duas inserções de conteúdo recomendadas (não obrigatórias)

1. Depois de `snnMotivacao.tex` (menção às "gerações" de redes neurais),
   abrir rapidamente `Backprop -> Forward e backward clássicos` (passo 1,
   "O neurônio") só para apontar a ativação sigmoide contínua — 30
   segundos, sem rodar a convergência inteira — antes de entrar em
   `snnConfusable.tex`. Ajuda a plateia a ancorar "2ª geração" em algo que
   acabaram de ver, não só ouvir.
2. Em `snnLif.tex`, quadro 3, a imagem estática
   `membranePotentialFull.png` pode ser **substituída** por um link para
   `SNN -> LIF` ao vivo (ver seção 5) — mais impactante que uma figura
   fixa, e o software já produz exatamente esse gráfico (corrente +
   potencial + raster de spikes). Manter a imagem como *fallback* caso o
   software não abra a tempo (problema de projetor, notebook diferente,
   etc.) — não apagar o `\includegraphics`, só adicionar o link ao lado.

---

## 4. Fase 3 — Link clicável do PDF para a seção do software

### 4.1 Mecanismo escolhido: `\href{run:...}` + CLI flag nova no software

Beamer/hyperref suportam links que executam um comando externo:
`\href{run:comando}{texto do link}`. O caminho passado a `run:` é relativo
ao **diretório do PDF**. Plano:

1. **Adicionar parsing de argumento `--demo <slug>` em
   `main.py`/`main_window.py`** (não existe hoje — `main()` só repassa
   `sys.argv` ao `QApplication`, nada mais). Esboço:

   ```python
   # main.py
   import argparse

   _DEMO_SLUGS = {
       "backprop.classic": ("Backpropagation", 0),
       "backprop.mlp": ("Backpropagation", 1),
       "bitnet.quant": ("BitNet", 0),
       "bitnet.forward": ("BitNet", 1),
       "bitnet.ste": ("BitNet", 2),
       "bitnet.guided": ("BitNet", 3),
       "snn.spikes": ("SNN", 0),
       "snn.poisson": ("SNN", 1),
       "snn.poisson_image": ("SNN", 2),
       "snn.lif": ("SNN", 3),
       "snn.surrogate": ("SNN", 4),
       "comparison": ("Comparação", 0),
   }

   def main() -> int:
       parser = argparse.ArgumentParser()
       parser.add_argument("--demo", choices=sorted(_DEMO_SLUGS), default=None)
       args, qt_args = parser.parse_known_args()
       ...
       window = MainWindow(initial_demo=_DEMO_SLUGS.get(args.demo))
   ```

   `MainWindow` ganha um parâmetro opcional `initial_demo: tuple[str,int]
   | None` que, se presente, chama `self._select_demo(...)` e
   `self.tree.setCurrentItem(...)` no item correspondente logo após
   `_build_ui()` — mesmo efeito de clicar manualmente na árvore, só que
   automático na abertura.
   - Índice por posição (grupo, índice-na-lista) em vez de por `title`
     evita repetir strings longas, mas **é frágil a reordenação** de
     `_build_demo_tree()` — alternativa mais robusta: guardar o slug como
     atributo em cada instância de `DemoModule` (`demo.slug = "snn.lif"`)
     e buscar por igualdade de slug em vez de índice. Preferir esta
     segunda forma na implementação real; a tabela acima já lista o slug
     de cada demo para isso.
   - Slug desconhecido/inválido: log de aviso em `stderr`, aplicação abre
     normalmente na tela de boas-vindas (nunca travar a abertura por causa
     de um argumento malformado — crítico durante a palestra).
   - **Implementado**: exatamente a segunda forma acima. `DemoModule.slug`
     (core/demo.py) + `MainWindow(initial_demo_slug=...)` /
     `_select_demo_by_slug` (app/main_window.py) buscam por igualdade de
     slug percorrendo a árvore de demos; `main.py` usa
     `argparse.parse_known_args` (sem `choices=`, para não abortar com
     `SystemExit` num slug desconhecido) e repassa qualquer flag restante
     ao `QApplication`. Verificado manualmente: `--demo snn.lif` seleciona
     e destaca o item certo na árvore lateral; um slug inexistente imprime
     o aviso em stderr e abre a tela de boas-vindas normalmente.

2. **Script wrapper**, `documentation/08-lectures/fronteiras-bitnets-redes-pulso/abrir-demo.sh`:

   ```bash
   #!/usr/bin/env bash
   # Chamado pelos links "run:" do PDF -- caminho relativo ao PDF, por isso
   # resolve o caminho do software a partir do próprio script, não do cwd.
   set -euo pipefail
   cd "$(dirname "${BASH_SOURCE[0]}")/../../../software/efficient_nn_lab"
   exec ./run.sh --demo "$1" &
   ```

   (`&` final para o `run:` do PDF não ficar bloqueado esperando o
   processo Qt terminar.)

3. **Nos slides**, ao lado de cada `\frametitle`/dentro do bloco relevante
   (ver tabela da seção 3), adicionar:

   ```tex
   % use \href{run:...}{...}, não \hyperlink (que só navega dentro do
   % próprio PDF, não executa programas externos)
   {\small\color{neutralColor}%
     \href{run:./abrir-demo.sh snn.lif}{\faIcon{external-link} Abrir no software: SNN -> LIF}%
   }
   ```

   (Sem `\faIcon` se `fontawesome5` não estiver instalado — usar apenas o
   texto "▶ Abrir no software: ...", em `neutralColor`, canto inferior do
   frame, via `textpos` já carregado no preamble.)

### 4.1.1 Nota de implementação — conteúdo pode desaparecer silenciosamente

Descoberto ao implementar: em 3 dos 8 frames (`bitnetTreinamento.tex`,
`snnConfusable.tex`, `snnBptt.tex`) o conteúdo já ocupava a altura inteira
do frame — beamer **não adiciona página nem quebra linha automaticamente
quando um frame transborda**, ele simplesmente corta (silenciosamente,
sem aviso) o que não coube. A linha `\abrirNoSoftware{...}` inserida no
final desses 3 frames desaparecia completamente do PDF (confirmado via
`pdftotext`, não só "visualmente apertado"). Corrigido com a opção nativa
do beamer `\begin{frame}[shrink]` nesses 3 arquivos — encolhe o conteúdo
inteiro do frame automaticamente até caber, sem precisar recalcular
espaçamento manualmente. **Ao adicionar novos links/conteúdo a qualquer
slide no futuro, sempre conferir com `pdftotext -layout` que o texto
esperado realmente aparece na página** — um frame "parece" ter cabido
mesmo quando um trecho final foi descartado, porque não há nenhum sinal
visual de corte.

### 4.2 Limitações reais — testar antes, ter plano B

- **Visualizadores de PDF bloqueiam `run:` por padrão** (é um risco de
  segurança conhecido). Okapi/Okular/Evince normalmente perguntam "deseja
  executar `abrir-demo.sh`?" na primeira vez (aceitável, um clique extra).
  **Muitos visualizadores (leitor embutido do navegador, preview do
  Chrome/Firefox, Preview do macOS) ignoram `run:` silenciosamente por
  design** — o link simplesmente não faz nada, sem erro visível.
- **Ação obrigatória**: testar os links no **visualizador exato** que será
  usado no dia da palestra (mesmo notebook, mesmo SO, mesmo programa),
  com pelo menos 3 dias de antecedência — não na véspera. Se o
  visualizador do dia não suportar `run:`, a Fase 3 vira apenas
  "documentação de qual demo abrir quando" (o valor da seção 3 continua
  inteiro; só perde a automação do clique).
- **Fallback sempre disponível, em toda entrada da tabela da seção 3**: o
  nome exato da demo como aparece na árvore lateral do software (já é o
  caso — a coluna "Demo do software" da tabela é literalmente o texto que
  aparece na UI), então mesmo sem o link o apresentador sabe exatamente
  onde clicar manualmente.
- **Segunda instância vs. janela já aberta**: `run:` sempre inicia um
  **processo novo**. Se o software já estiver aberto (o uso esperado
  numa palestra — abrir uma vez, deixar a janela viva o tempo todo, só
  clicar na árvore lateral manualmente entre uma seção e outra), clicar
  o link do PDF abre uma **segunda janela**, não troca a seção da janela
  existente. Fazer o link "pular" para uma janela já aberta exigiria um
  mecanismo de instância única (socket local/lockfile) — engenharia
  adicional, não recomendada para esta semana (ver 4.3).
  **Recomendação prática**: decidir com antecedência UM modo de operação
  e ensaiar só esse: (a) manter o software sempre aberto e navegar só
  pelo clique manual na árvore lateral (mais simples, mais confiável,
  **recomendado para a apresentação real**), ou (b) fechar o software
  entre seções e usar os links do PDF para reabri-lo já no lugar certo
  (mais impressionante, mais frágil). O link do PDF vale como demonstração
  de engenharia/reprodutibilidade mesmo se (a) for o modo escolhido ao
  vivo.

### 4.3 Fora de escopo para esta semana (documentado, não implementado)

Instância única com IPC (o software já aberto detecta o argumento
`--demo` de uma segunda invocação via socket local — `QLocalServer`/
`QLocalSocket` — e troca de seção na janela existente em vez de abrir uma
nova) resolveria a limitação de 4.2 de forma completa. Estimativa: 3-5h de
trabalho (servidor/cliente local, tratamento de janela já em foco,
testes). Não recomendado antes da palestra da semana que vem; registrar
como *follow-up* pós-palestra caso o formato "PDF clicável" seja usado de
novo no futuro.

---

## 5. Fase 4 — Registro dos passos como comentários no `.tex`

Independente da Fase 3 (links) ser viável a tempo, cada slide da tabela da
seção 3 deve ganhar um comentário LaTeX logo após seu `\begin{frame}`
apontando a demo e o passo exato — útil como *anotação do apresentador*
mesmo que o link em si falhe no dia:

```tex
\begin{frame}
	\frametitle{O neurônio Leaky Integrate-and-Fire (LIF)}
	% [SOFTWARE] SNN -> LIF | slider tau/corrente a zero para o quadro 2;
	% Play com padrões para o quadro 3 (substitui membranePotentialFull.png)
	...
```

Baixo custo (uma linha por slide), zero risco de quebrar a compilação, e
sobrevive independente do resultado da Fase 3.

---

## 6. Checklist de ensaio (antes da palestra)

- [ ] Compilar os slides com a nova paleta/fonte (Fase 1) e revisar as ~26
      páginas visualmente — checar que nenhum texto ficou com contraste
      ruim sobre `accentColor`/`bitnetColor`/`snnColor`.
- [ ] Rodar o software (`./run.sh`) numa janela ao lado do PDF e percorrer
      a tabela da seção 3 uma vez, cronometrando quanto tempo cada demo
      leva para chegar no passo indicado a partir do menu lateral (decide
      se o `--demo`/link vale a pena implementar ou se navegação manual já
      é rápida o suficiente).
- [ ] Testar os links `run:` (Fase 3) no notebook e visualizador de PDF
      reais do dia da palestra — **não deixar para a véspera**.
- [ ] Verificar que o slider `w` de `BitNet -> Quantização` alcança os
      valores exatos do exemplo do slide (`0,6`, `-0,3`, `0,05`) dentro do
      passo do slider (`step=0.05` — confirmar que bate).
- [ ] Decidir e ensaiar o modo de operação da seção 4.2 (deixar aberto +
      clique manual, vs. fechar/reabrir via link) — escolher UM e treinar
      só esse.
- [ ] Preparar a frase de transição para a "lacuna de precisão" da seção
      1.5 (limiar fixo vs. absmean real) — dizer isso ao vivo evita que
      pareça um erro não percebido.
- [ ] Conferir que `SNN -> LIF` com parâmetros padrão realmente reproduz
      algo comparável ao gráfico estático do slide 3 de `snnLif.tex`
      (mesma forma qualitativa: decaimento, subida, disparo, reset) —
      ajustar `tau`/`r`/`v_th` default do software se divergir demais do
      exemplo do slide, para os dois lados contarem a mesma história com
      os mesmos números sempre que possível.

---

## 7. Resumo de arquivos a tocar

| Arquivo | Mudança |
|---|---|
| `documentation/08-lectures/fronteiras-bitnets-redes-pulso/preamble.tex` | Fase 1: cores vívidas, `accentColor`, fonte `DejaVuSans` (pacote `dejavu`, `pdflatex`), `title`/`frametitle` bg |
| `documentation/08-lectures/fronteiras-bitnets-redes-pulso/slides/*.tex` | Fase 4 (comentários) obrigatório; Fase 3 (`\href{run:...}`) se viável |
| `documentation/08-lectures/fronteiras-bitnets-redes-pulso/abrir-demo.sh` | Novo — wrapper de lançamento (Fase 3) |
| `software/efficient_nn_lab/src/efficient_nn_lab/main.py` | Fase 3: parsing de `--demo` |
| `software/efficient_nn_lab/src/efficient_nn_lab/app/main_window.py` | Fase 3: `MainWindow(initial_demo=...)`, slug por `DemoModule` |
| `software/efficient_nn_lab/src/efficient_nn_lab/app/theme.py` | Fase 1: `font-family: "DejaVu Sans"` no stylesheet |

## 8. Ordem de execução recomendada (prazo: uma semana)

1. Fase 1 (cores + fonte) — maior impacto visual, menor risco, ~1-2h.
2. Fase 4 (comentários nos slides) — quase grátis, trava o mapeamento da
   seção 3 por escrito antes de mexer em código, ~30min.
3. Ensaio completo sem links (navegação manual) — valida a seção 3 de
   verdade, ~1h.
4. Fase 3 (CLI `--demo` + `\href{run:...}`) **só se sobrar tempo** depois
   de 1-3 — é a parte de maior risco de falhar silenciosamente no dia (ver
   4.2), então nunca deve ser a única forma de navegação preparada.
