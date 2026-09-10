# -*- coding: utf-8 -*-
"""Brazilian-Portuguese catalog for the didactic surface of the app.

Keys are the exact English source strings passed to ``i18n.t``. A string with
no entry here falls back to English (so partial coverage is safe). Coverage
today: menus and common controls, the guided tour (both pipelines), the verdict
sentences, the colour key, the glossary meanings, and the core "How to read
this" boxes. Deep per-view internals are still English.
"""

CATALOG = {
    # ---- menus / actions / common controls -------------------------
    "&View": "&Exibir",
    "&Help": "&Ajuda",
    "E&xport": "E&xportar",
    "&Experiment": "&Experimento",
    "Theme": "Tema",
    "System": "Do sistema",
    "Light": "Claro",
    "Dark": "Escuro",
    "Language": "Idioma",
    "Low-performance mode": "Modo de baixo desempenho",
    "Export current view…": "Exportar a visão atual…",
    "▶  Start guided tour": "▶  Iniciar tour guiado",
    "Explain the current view": "Explicar a visão atual",
    "Glossary (all terms)…": "Glossário (todos os termos)…",
    "Refresh paraconsistent views": "Atualizar as visões paraconsistentes",
    "Back": "Voltar",
    "◀ Back": "◀ Voltar",
    "Next ▶": "Avançar ▶",
    "Finish ▶": "Concluir ▶",
    "Free explore": "Explorar livremente",
    "Colour key:": "Legenda de cores:",
    "How to read this": "Como ler isto",
    "\N{SPEAKER WITH THREE SOUND WAVES}  Listen": "\N{SPEAKER WITH THREE SOUND WAVES}  Ouvir",
    "Tour ended — every tab is back. Explore freely.":
        "Tour encerrado — todas as abas voltaram. Explore à vontade.",
    "opened the explanation for this view": "explicação desta visão aberta",
    "Language changed — some fixed labels update after a restart.":
        "Idioma alterado — alguns rótulos fixos só mudam ao reiniciar.",

    # ---- guided tour: meeting01 -----------------------------------
    "How a spiking network learns a spoken digit":
        "Como uma rede de disparos aprende um dígito falado",
    "1 · A spoken digit is just a wiggle of air":
        "1 · Um dígito falado é só uma vibração do ar",
    "This is the digit spoken aloud, drawn as air pressure over time — the same "
    "thing your ear receives. Press <b>🔊 Listen</b> to hear it.<br><br>"
    "The whole goal: get a computer to tell digits apart <i>without ever being "
    "told</i> what each one sounds like.":
        "Este é o dígito dito em voz alta, desenhado como a pressão do ar ao longo "
        "do tempo — o mesmo que o seu ouvido recebe. Toque em <b>🔊 Ouvir</b> para "
        "escutá-lo.<br><br>O objetivo: fazer um computador distinguir os dígitos "
        "<i>sem nunca ser informado</i> de como cada um soa.",
    "256 numbers — one 32-millisecond slice of the sound.":
        "256 números — uma fatia de 32 milissegundos do som.",
    "2 · Split the sound into frequency bands":
        "2 · Separe o som em faixas de frequência",
    "The same slice, broken into 16 <b>frequency bands</b> — low pitches on the "
    "left, high on the right (this split is called a <i>wavelet</i>).<br><br>"
    "Speech energy piles up in just a few bands. That pattern is a fingerprint "
    "of the sound — more useful to a model than the raw wiggle.":
        "A mesma fatia, dividida em 16 <b>faixas de frequência</b> — sons graves à "
        "esquerda, agudos à direita (essa divisão se chama <i>wavelet</i>).<br><br>"
        "A energia da fala se concentra em poucas faixas. Esse padrão é uma "
        "impressão digital do som — mais útil a um modelo que a onda crua.",
    "3 · Turn the numbers into spikes":
        "3 · Transforme os números em disparos",
    "Real neurons don't pass numbers around — they fire brief <b>spikes</b>. "
    "Here every value in the window is turned into a spike train.<br><br>"
    "With <i>latency</i> encoding, a louder value fires its spike earlier. "
    "Louder = sooner.":
        "Neurônios reais não trocam números — eles emitem <b>disparos</b> curtos. "
        "Aqui, cada valor da janela vira um trem de disparos.<br><br>"
        "Na codificação por <i>latência</i>, um valor mais alto dispara mais cedo. "
        "Mais alto = mais cedo.",
    "4 · One neuron that leaks, charges, and fires":
        "4 · Um neurônio que vaza, carrega e dispara",
    "Watch the green line: charge <b>builds up</b> inside the neuron as spikes "
    "arrive, and slowly <b>leaks away</b> between them. When it crosses the red "
    "line the neuron <b>fires</b> a spike and the charge drops.<br><br>"
    "That is a <i>leaky integrate-and-fire</i> neuron — the membrane-potential "
    "picture people ask about.":
        "Observe a linha verde: a carga <b>sobe</b> dentro do neurônio conforme os "
        "disparos chegam e <b>vaza</b> devagar entre eles. Ao cruzar a linha "
        "vermelha, o neurônio <b>dispara</b> e a carga cai.<br><br>"
        "Isso é um neurônio <i>integra-e-dispara com vazamento</i> — a figura do "
        "potencial de membrana que costumam perguntar.",
    "Green = charge, red = the firing line, amber ticks = spikes out.":
        "Verde = carga, vermelho = linha de disparo, traços âmbar = disparos de saída.",
    "5 · Squeeze the whole window to 32 numbers":
        "5 · Comprima a janela inteira em 32 números",
    "The full network compresses each window down to just <b>32 numbers</b> — "
    "the <i>latent</i>. Press <b>Project</b>: every window of this fold is run "
    "through and drawn as a dot.<br><br>"
    "If the same digit lands in the same clump, those 32 numbers have captured "
    "what makes the digit that digit.":
        "A rede inteira comprime cada janela em apenas <b>32 números</b> — o "
        "<i>latente</i>. Toque em <b>Projetar</b>: cada janela desta dobra passa "
        "pela rede e vira um ponto.<br><br>"
        "Se o mesmo dígito cai no mesmo aglomerado, esses 32 números capturaram o "
        "que faz o dígito ser aquele dígito.",
    "6 · Rebuild it — and see what was lost":
        "6 · Reconstrua — e veja o que se perdeu",
    "The decoder tries to redraw the original window from those 32 numbers "
    "alone. <span style='color:#4F9DF7'>Blue</span> = original, "
    "<span style='color:#F5A623'>orange</span> = rebuild, "
    "<span style='color:#969aa0'>grey</span> = the difference.<br><br>"
    "A grey line that barely moves means the 32 numbers kept almost everything.":
        "O decodificador tenta redesenhar a janela original só a partir desses 32 "
        "números. <span style='color:#4F9DF7'>Azul</span> = original, "
        "<span style='color:#F5A623'>laranja</span> = reconstrução, "
        "<span style='color:#969aa0'>cinza</span> = a diferença.<br><br>"
        "Uma linha cinza quase parada significa que os 32 números guardaram quase tudo.",
    "You've seen the whole path":
        "Você viu o caminho inteiro",
    "Sound → frequency bands → spikes → one neuron's charge → 32 numbers → "
    "rebuild. <br><br>Now click <b>Free explore</b>: pick any window, any tab, "
    "and follow a single number all the way through. Every plot shows where "
    "its data came from.":
        "Som → faixas de frequência → disparos → carga de um neurônio → 32 números "
        "→ reconstrução.<br><br>Agora clique em <b>Explorar livremente</b>: escolha "
        "qualquer janela, qualquer aba, e siga um único número do começo ao fim. "
        "Cada gráfico mostra de onde vieram seus dados.",

    # ---- guided tour: thesis -------------------------------------
    "How wavelets + paraconsistent logic authenticate a person":
        "Como wavelets + lógica paraconsistente autenticam uma pessoa",
    "1 · Brain or voice signals from one person":
        "1 · Sinais de cérebro ou de voz de uma pessoa",
    "Six channels of <b>EEG</b> (tiny voltages from the scalp) or a voice "
    "recording. The goal: decide whether two recordings come from the "
    "<i>same person</i>.":
        "Seis canais de <b>EEG</b> (pequenas voltagens do couro cabeludo) ou uma "
        "gravação de voz. O objetivo: decidir se duas gravações são da "
        "<i>mesma pessoa</i>.",
    "EEG here is 6 channels × 4096 samples at 1024 readings per second.":
        "O EEG aqui tem 6 canais × 4096 amostras a 1024 leituras por segundo.",
    "2 · Split into frequency bands":
        "2 · Separe em faixas de frequência",
    "The same wavelet idea as before: break each channel into frequency "
    "bands. The hand-designed path then measures several things in every band "
    "— energy, how often it crosses zero, how disordered it is, and more.":
        "A mesma ideia de wavelet de antes: quebre cada canal em faixas de "
        "frequência. O caminho feito à mão então mede várias coisas em cada faixa "
        "— energia, quantas vezes cruza o zero, quão desordenada é, e mais.",
    "3 · 96 hand-picked measurements per recording":
        "3 · 96 medições escolhidas à mão por gravação",
    "Each row is one recording, each column one measurement. Unlike the "
    "spoken-digit network, nothing here is learned — a human chose every "
    "measurement. This is the <i>handcrafted feature matrix</i>.":
        "Cada linha é uma gravação, cada coluna uma medição. Diferente da rede de "
        "dígitos falados, nada aqui é aprendido — um humano escolheu cada medição. "
        "Esta é a <i>matriz de atributos feita à mão</i>.",
    "1974 recordings × 96 measurements for this run.":
        "1974 gravações × 96 medições nesta execução.",
    "4 · Evidence can support AND deny at once":
        "4 · A evidência pode apoiar E negar ao mesmo tempo",
    "<b>Paraconsistent</b> logic allows a claim to be backed and contradicted "
    "at the same time — exactly what noisy biometrics look like.<br><br>"
    "Horizontal <b>G1</b> = net certainty (right = 'same person'). Vertical "
    "<b>G2</b> = how much the evidence fights itself. The ideal spot is the "
    "far right at zero height.":
        "A lógica <b>paraconsistente</b> permite que uma afirmação seja apoiada e "
        "contrariada ao mesmo tempo — exatamente como é a biometria ruidosa.<br><br>"
        "Horizontal <b>G1</b> = certeza líquida (direita = 'mesma pessoa'). "
        "Vertical <b>G2</b> = o quanto a evidência se contradiz. O ponto ideal é o "
        "canto direito, na altura zero.",
    "5 · Which recipe of measurements wins":
        "5 · Qual receita de medições vence",
    "Every combination of wavelet + measurements gets one score: its distance "
    "to that ideal spot, with a penalty for self-contradiction "
    "(<i>D_penalized</i> — smaller is better).<br><br>"
    "This sorted table is the experiment's actual answer.":
        "Cada combinação de wavelet + medições recebe uma nota: a distância até "
        "aquele ponto ideal, com uma penalidade por autocontradição "
        "(<i>D_penalized</i> — quanto menor, melhor).<br><br>"
        "Esta tabela ordenada é a resposta real do experimento.",
    "The row at the top has the least contradiction and the most certainty.":
        "A linha do topo tem a menor contradição e a maior certeza.",
    "Signal → frequency bands → 96 measurements → support-vs-denial plane → "
    "ranking. <br><br>Click <b>Free explore</b> and follow any feature set "
    "through the Triangle tab to see all three stages side by side.":
        "Sinal → faixas de frequência → 96 medições → plano apoio-vs-negação → "
        "ranking.<br><br>Clique em <b>Explorar livremente</b> e siga qualquer "
        "conjunto de atributos pela aba Triângulo para ver os três estágios lado a lado.",

    # ---- verdict sentences --------------------------------------
    "No rebuild yet — train a model fold first.":
        "Ainda sem reconstrução — treine uma dobra do modelo primeiro.",
    "Near-perfect rebuild — the small set of latent numbers kept almost everything (R² {r2}).":
        "Reconstrução quase perfeita — o pequeno conjunto de números latentes guardou quase tudo (R² {r2}).",
    "Close rebuild — the shape is right, fine detail is softened (R² {r2}).":
        "Reconstrução próxima — a forma está certa, os detalhes finos ficam suavizados (R² {r2}).",
    "Rough rebuild — the big movements survive, the sharp parts are lost (R² {r2}).":
        "Reconstrução grosseira — os grandes movimentos sobrevivem, as partes bruscas se perdem (R² {r2}).",
    "Weak rebuild — close to a flat line at the mean (R² {r2}).":
        "Reconstrução fraca — perto de uma linha reta na média (R² {r2}).",
    "Failed rebuild — R² {r2} is below zero, so a flat line at the mean would "
    "match the window more closely; the latent numbers did not capture it.":
        "Reconstrução falhou — R² {r2} está abaixo de zero, então uma linha reta na "
        "média se aproximaria mais da janela; os números latentes não a capturaram.",
    "very sparse": "muito esparso", "sparse": "esparso",
    "moderate": "moderado", "dense": "denso",
    " — the neuron kept {pct}% of the incoming spikes":
        " — o neurônio manteve {pct}% dos disparos que chegaram",
    "{n_out} spikes out of {n_steps} time steps ({dens} firing){kept}. "
    "Each spike is one 'the neuron reacted here' moment.":
        "{n_out} disparos em {n_steps} passos de tempo (disparo {dens}){kept}. "
        "Cada disparo é um momento de 'o neurônio reagiu aqui'.",
    "clean": "limpa", "somewhat conflicting": "um pouco contraditória",
    "highly conflicting": "muito contraditória",
    "supports the match": "apoia a correspondência",
    "denies the match": "nega a correspondência",
    "is undecided": "está indecisa",
    "Evidence {lean} (certainty G1 {g1}) and is {conflict} (contradiction G2 {g2}). "
    "α is support for, β is support against — both can be high at once.":
        "A evidência {lean} (certeza G1 {g1}) e é {conflict} (contradição G2 {g2}). "
        "α é apoio a favor, β é apoio contra — ambos podem ser altos ao mesmo tempo.",
    "concentrated in a few bands": "concentrada em poucas faixas",
    "spread across many bands": "espalhada por muitas faixas",
    "The signal's energy is {spread}; band {top} alone holds {pct}%. "
    "Each band is a frequency range, low bands first.":
        "A energia do sinal está {spread}; só a faixa {top} tem {pct}%. "
        "Cada faixa é um intervalo de frequência, as graves primeiro.",
    "Every window — hundreds of samples — is squeezed to just {dim} numbers. "
    "If similar digits land near each other here, those numbers carry the meaning.":
        "Cada janela — centenas de amostras — é comprimida em apenas {dim} números. "
        "Se dígitos parecidos caem perto aqui, esses números carregam o significado.",

    # ---- colour key (palette.MEANING) ---------------------------
    "original signal": "sinal original",
    "model's rebuild": "reconstrução do modelo",
    "difference (how wrong)": "diferença (o quanto erra)",
    "a spike fired": "um disparo emitido",
    "charge in the neuron": "carga no neurônio",
    "firing line": "linha de disparo",
    "frequency-band energy": "energia da faixa de frequência",
    "one measurement": "uma medição",
    "clean support": "apoio limpo",
    "support + denial at once": "apoio + negação ao mesmo tempo",
    "your selection": "sua seleção",

    # ---- a few view titles ------------------------------------
    "The window we start from — a slice of the signal, mean 0":
        "A janela de onde partimos — uma fatia do sinal, média 0",
    "direct — passes the numbers straight through (no spikes)":
        "direto — passa os números sem alteração (sem disparos)",
    "The signal going in — pick a band on the left to see just that band":
        "O sinal que entra — escolha uma faixa à esquerda para ver só ela",
    "Each dot is one feature recipe · right = 'same person' · "
    "up = the evidence contradicts itself · bottom-right corner is the goal":
        "Cada ponto é uma receita de atributos · direita = 'mesma pessoa' · "
        "cima = a evidência se contradiz · o canto inferior direito é o alvo",

    # ---- progress / busy --------------------------------------
    "Working…": "Processando…",
    "Loading data…": "Carregando dados…",
    "Running the network…": "Executando a rede…",
    "Computing the wavelet…": "Calculando a wavelet…",
    "Projecting the latent space…": "Projetando o espaço latente…",
    "Computing…": "Calculando…",

    # ---- window / status / menu extras -------------------------
    "Experiment Microscope": "Microscópio de Experimentos",
    "FULL RESOLUTION": "RESOLUÇÃO TOTAL",
    "DISPLAY-DOWNSAMPLED": "EXIBIÇÃO REDUZIDA",
    "LOW-PERFORMANCE MODE": "MODO DE BAIXO DESEMPENHO",
    "cursor @ sample {ts}": "cursor na amostra {ts}",
    "No meeting01 run detected under results/meeting01/.":
        "Nenhuma execução do meeting01 encontrada em results/meeting01/.",
    "opened the explanation for this view": "explicação desta visão aberta",
    "this view has no dedicated explanation yet — see Help → Glossary":
        "esta visão ainda não tem explicação própria — veja Ajuda → Glossário",
    "Glossary — every abbreviation and metric":
        "Glossário — cada abreviação e métrica",
    "Glossary": "Glossário",
    "Export figure": "Exportar figura",
    "current view has nothing to export": "a visão atual não tem nada para exportar",
    "export failed: {exc}": "falha ao exportar: {exc}",
    "wrote {written}": "gravado {written}",
    "compute failed: {exc}": "falha no cálculo: {exc}",
    "bookmarked: {name}": "marcado: {name}",
    "restored: {name}": "restaurado: {name}",
    "(selection path not found — state only)":
        "(caminho da seleção não encontrado — apenas o estado)",
    "quantity": "quantidade", "value": "valor", "what it means": "o que significa",

    # ---- tab labels -------------------------------------------
    "Signal": "Sinal",
    "Wavelet Lab": "Laboratório de Wavelet",
    "Wavelet 3D": "Wavelet 3D",
    "Feature Matrix": "Matriz de Atributos",
    "Encoding Lab": "Laboratório de Codificação",
    "SNN Lab": "Laboratório de SNN",
    "SNN 3D": "SNN 3D",
    "Latent Space": "Espaço Latente",
    "Reconstruction": "Reconstrução",
    "Paraconsistent plane": "Plano paraconsistente",
    "Paraconsistent landscape": "Relevo paraconsistente",
    "Pipeline": "Pipeline",
    "Triangle": "Triângulo",
    "Comparison": "Comparação",
    "Ranking": "Classificação",
    "Timeline": "Linha do tempo",

    # ---- dock titles -----------------------------------------
    "Data Explorer": "Explorador de Dados",
    "Inspector": "Inspetor",
    "Raw artifact": "Artefato bruto",
    "Reproduce": "Reproduzir",
    "Meeting01 session": "Sessão do Meeting01",
    "Bookmarks": "Marcadores",
    "Developer": "Desenvolvedor",
    "Guided tour": "Tour guiado",

    # ---- HelpBox titles --------------------------------------
    "Raw signal": "Sinal bruto",
    "Experiment timeline": "Linha do tempo do experimento",
    "NSGA-II search": "Busca NSGA-II",
    "Triangle view": "Visão em triângulo",
    "Latent Space Explorer": "Explorador do Espaço Latente",
    "Meeting01 vs Thesis": "Meeting01 vs Tese",

    # ---- HelpBox bodies -------------------------------------
    '\n<b>What this shows.</b> The signal exactly as the pipeline sees it for the\nselected sample — an audio waveform, or several stacked EEG channels\n(EEG = electroencephalogram, brain electrical activity).\n<br><br>\n<b>Axes.</b> Horizontal = sample number (multiply by 1/sampling-rate for seconds;\nthe rate is in the title). Vertical = amplitude. For meeting01 windows the\namplitude is <b>z-scored</b> (mean 0, spread 1) because that is what the models\nreceive; the unit is shown on the left axis.\n<br><br>\n<b>Colours / lines.</b> One colour per channel, named in the legend. Drag on the\nplot to select a time range — downstream views can restrict to it. The vertical\ncursor line reports the exact value under it.\n<br><br>\n<b>DISPLAY-DOWNSAMPLED</b> in the title (and the status bar) means the drawn\ncurve is decimated for speed; the cursor still reads the full-resolution number.\n<br><br>\n<b>🔊 Listen</b> plays the waveform through your speakers (audio samples only;\nEEG has no sound). The <b>▶</b> transport at the bottom is separate — it steps\nanimation frames, it does not play audio.\n':
        '\n<b>O que isto mostra.</b> O sinal exatamente como o pipeline o vê para a\namostra selecionada — uma onda de áudio, ou vários canais de EEG empilhados\n(EEG = eletroencefalograma, atividade elétrica do cérebro).\n<br><br>\n<b>Eixos.</b> Horizontal = número da amostra (multiplique por 1/taxa-de-amostragem\npara segundos; a taxa está no título). Vertical = amplitude. Para janelas do\nmeeting01 a amplitude é <b>z-score</b> (média 0, dispersão 1) porque é isso que os\nmodelos recebem; a unidade aparece no eixo esquerdo.\n<br><br>\n<b>Cores / linhas.</b> Uma cor por canal, nomeada na legenda. Arraste no gráfico\npara selecionar um intervalo de tempo — visões seguintes podem se restringir a\nele. A linha vertical do cursor informa o valor exato sob ela.\n<br><br>\n<b>EXIBIÇÃO REDUZIDA</b> no título (e na barra de status) significa que a curva\ndesenhada foi decimada por velocidade; o cursor ainda lê o número em resolução total.\n<br><br>\n<b>🔊 Ouvir</b> toca a onda pelos alto-falantes (só amostras de áudio; o EEG não\ntem som). O transporte <b>▶</b> embaixo é separado — ele avança quadros de\nanimação, não toca áudio.\n',
    "\n<b>What this shows.</b> The signal split into frequency <b>sub-bands</b> by a\n<b>wavelet packet</b> decomposition (a binary tree of filters; each leaf is a\nnarrow frequency range). Unlike a plain spectrum this keeps <i>when</i> things\nhappen, not only <i>what</i> frequencies are present.\n<br><br>\n<b>Controls.</b> <i>wavelet</i> picks the filter shape (Haar = blocky and fast;\ndaub4…daub20 = progressively smoother Daubechies filters). <i>level</i> = tree\ndepth, so 2<sup>level</sup> leaves. <i>channel</i> chooses which EEG channel to\ndecompose.\n<br><br>\n<b>Table / plot.</b> Per leaf: <b>energy</b> (sum of squared coefficients = power\nin that band) and <b>relative energy</b> (share of the total, bands sum to 1).\nClick a leaf to see its raw coefficients.\n":
        "\n<b>O que isto mostra.</b> O sinal dividido em <b>sub-faixas</b> de frequência por\numa decomposição em <b>pacote de wavelet</b> (uma árvore binária de filtros; cada\nfolha é uma faixa estreita de frequência). Diferente de um espectro simples, isto\nmantém <i>quando</i> as coisas acontecem, não só <i>quais</i> frequências existem.\n<br><br>\n<b>Controles.</b> <i>wavelet</i> escolhe o formato do filtro (Haar = quadrado e\nrápido; daub4…daub20 = filtros de Daubechies cada vez mais suaves). <i>level</i> =\nprofundidade da árvore, logo 2<sup>level</sup> folhas. <i>channel</i> escolhe qual\ncanal de EEG decompor.\n<br><br>\n<b>Tabela / gráfico.</b> Por folha: <b>energia</b> (soma dos coeficientes ao\nquadrado = potência naquela faixa) e <b>energia relativa</b> (fração do total, as\nfaixas somam 1). Clique numa folha para ver seus coeficientes brutos.\n",
    "\n<b>What this shows.</b> The wavelet-packet decomposition as a 3-D surface instead\nof a table.\n<br><br>\n<b>X</b> = coefficient index within a leaf. <b>Y</b> = leaf number, low to high\nfrequency. <b>Z</b> (height & colour) = |coefficient| magnitude, normalised so\nthe largest is 1. Ridges are bands carrying a lot of the signal's power.\n<br><br>\nThe <b>|z| threshold</b> slider hides small coefficients; <b>isolate leaf</b>\nlifts one band out as a red line. Rotate / zoom / pan with the mouse. Disabled in\nlow-performance mode.\n":
        "\n<b>O que isto mostra.</b> A decomposição em pacote de wavelet como uma superfície\n3-D em vez de uma tabela.\n<br><br>\n<b>X</b> = índice do coeficiente dentro de uma folha. <b>Y</b> = número da folha,\nda baixa para a alta frequência. <b>Z</b> (altura e cor) = magnitude |coeficiente|,\nnormalizada para que a maior seja 1. As cristas são faixas que carregam boa parte\nda potência do sinal.\n<br><br>\nO controle <b>limiar |z|</b> esconde coeficientes pequenos; <b>isolar folha</b>\ndestaca uma faixa como linha vermelha. Gire / amplie / desloque com o mouse.\nDesativado no modo de baixo desempenho.\n",
    "\n<b>What this shows.</b> The full table of handcrafted feature values for a thesis\nrun: one row per sample, one column per feature.\n<br><br>\n<b>Colours.</b> A heatmap — brighter / darker means larger / smaller value; the\ncolour bar gives the scale. The per-column statistics (mean, standard deviation,\nmin, max) sit beside it. Turning on the z-score toggle rescales <i>for display\nonly</i> so columns of different magnitude become comparable — the underlying\nnumbers do not change.\n<br><br>\n<b>Click a cell</b> to read its exact value, the sample's class, and (when the\nwavelet layout allows) which frequency band it came from.\n<br><br>\nFeature names encode the descriptor and scale, e.g. energy / <b>ZCR</b>\n(zero-crossing rate) / entropy / <b>Teager</b> (Teager–Kaiser energy) / jitter /\nshimmer per wavelet band.\n":
        "\n<b>O que isto mostra.</b> A tabela completa dos valores de atributos feitos à mão\npara uma execução da tese: uma linha por amostra, uma coluna por atributo.\n<br><br>\n<b>Cores.</b> Um mapa de calor — mais claro / mais escuro significa valor maior /\nmenor; a barra de cores dá a escala. As estatísticas por coluna (média, desvio\npadrão, mínimo, máximo) ficam ao lado. Ligar o z-score reescala <i>apenas para\nexibição</i> para que colunas de magnitudes diferentes fiquem comparáveis — os\nnúmeros originais não mudam.\n<br><br>\n<b>Clique numa célula</b> para ler seu valor exato, a classe da amostra e (quando\no arranjo da wavelet permite) de qual faixa de frequência ela veio.\n<br><br>\nOs nomes dos atributos codificam o descritor e a escala, p. ex. energia / <b>ZCR</b>\n(taxa de cruzamentos por zero) / entropia / <b>Teager</b> (energia de\nTeager–Kaiser) / jitter / shimmer por faixa de wavelet.\n",
    '\n<b>What this shows.</b> The three spike <b>encodings</b> side by side for one\nwindow, so you can see how each turns a real number into spike events.\n<br><br>\n<b>direct</b> — the normalised sample passes straight through (no spike-time\nconversion). <b>poisson</b> — a random spike train whose average <b>firing\nrate</b> is proportional to the value. <b>latency</b> — time-to-first-spike:\nlarger values fire earlier, small values late or never.\n<br><br>\nHorizontal axis = time step within the window; each row/track is a spike train,\neach mark a spike. <i>seed</i> fixes the randomness of the poisson encoding so the\npicture is reproducible.\n':
        '\n<b>O que isto mostra.</b> As três <b>codificações</b> de disparos lado a lado para\numa janela, para você ver como cada uma transforma um número real em eventos de\ndisparo.\n<br><br>\n<b>direto</b> — a amostra normalizada passa sem alteração (sem conversão para\ntempo de disparo). <b>poisson</b> — um trem de disparos aleatório cuja <b>taxa de\ndisparo</b> média é proporcional ao valor. <b>latência</b> — tempo até o primeiro\ndisparo: valores maiores disparam mais cedo, valores pequenos tarde ou nunca.\n<br><br>\nEixo horizontal = passo de tempo dentro da janela; cada linha/trilha é um trem de\ndisparos, cada marca um disparo. <i>seed</i> fixa a aleatoriedade da codificação\npoisson para que a figura seja reproduzível.\n',
    '\n<b>What this shows.</b> How a meeting01 window becomes spikes.\nSNN = Spiking Neural Network; its neurons fire discrete 0/1 <b>spikes</b> instead\nof sending continuous numbers.\n<br><br>\n<b>Panel 1</b> — the normalised (z-scored) window.\n<b>Panel 2</b> — the input <b>spike train</b> for the chosen encoding\n(<i>direct</i> = pass-through, <i>poisson</i> = firing rate ∝ value,\n<i>latency</i> = bigger value fires earlier). Each mark is one spike; click it\nfor its exact time.\n<b>Panel 3</b> — the <b>membrane voltage</b> v[t] of the recurrent\nLeaky-Integrate-and-Fire (LIF) transform, sweeping the 256 window samples as time\nsteps: <code>v[t] = α·v[t−1] + x[t] − s[t−1]·v_th</code>. The dashed red line is\nthe threshold <b>v_th</b>; a green mark sits on every step where v crossed it and\nthe neuron fired.\n<b>Panel 4</b> (only with a trained model) — one membrane value per encoder LIF\nneuron, a snapshot because that network runs with a single time step.\n<br><br>\n<b>Sliders.</b> <b>α</b> (alpha) = leak, 0…1: near 1 the neuron remembers input\nlonger. <b>v_th</b> = threshold: higher → fewer spikes.\n':
        '\n<b>O que isto mostra.</b> Como uma janela do meeting01 vira disparos.\nSNN = Rede Neural de Disparos; seus neurônios emitem <b>disparos</b> discretos 0/1\nem vez de enviar números contínuos.\n<br><br>\n<b>Painel 1</b> — a janela normalizada (z-score).\n<b>Painel 2</b> — o <b>trem de disparos</b> de entrada para a codificação escolhida\n(<i>direto</i> = passa-direto, <i>poisson</i> = taxa de disparo ∝ valor,\n<i>latência</i> = valor maior dispara mais cedo). Cada marca é um disparo; clique\nnela para o tempo exato.\n<b>Painel 3</b> — a <b>voltagem de membrana</b> v[t] da transformação recorrente\nIntegra-e-Dispara com Vazamento (LIF), percorrendo as 256 amostras da janela como\npassos de tempo: <code>v[t] = α·v[t−1] + x[t] − s[t−1]·v_th</code>. A linha\nvermelha tracejada é o limiar <b>v_th</b>; uma marca verde fica em cada passo em\nque v o cruzou e o neurônio disparou.\n<b>Painel 4</b> (só com um modelo treinado) — um valor de membrana por neurônio LIF\ndo codificador, um instantâneo porque essa rede roda com um único passo de tempo.\n<br><br>\n<b>Controles.</b> <b>α</b> (alfa) = vazamento, 0…1: perto de 1 o neurônio lembra a\nentrada por mais tempo. <b>v_th</b> = limiar: mais alto → menos disparos.\n',
    "\n<b>What this shows.</b> The trained meeting01 SNN autoencoder <i>encoder</i> as\ncolumns of neurons: input → Linear(64) → LIF spikes(64) → latent(32).\n<br><br>\n<b>Node size &amp; colour</b> = that neuron's activity for the selected window\n(|activation|, or spike count for the LIF column; colour bar on the left).\n<b>Edges</b> = the connecting Linear layer's weights, drawn only for the top few\n|weight| per target neuron so the picture stays readable.\n<br><br>\n<b>Controls.</b> top-K edges per neuron, |weight| threshold, activity threshold.\nPress ▶ on the transport bar to <b>flood the signal through the layers</b> one\ncolumn at a time. LIF = Leaky Integrate-and-Fire spiking neuron.\n":
        "\n<b>O que isto mostra.</b> O <i>codificador</i> do autoencoder SNN treinado do\nmeeting01 como colunas de neurônios: entrada → Linear(64) → disparos LIF(64) →\nlatente(32).\n<br><br>\n<b>Tamanho e cor do nó</b> = a atividade daquele neurônio para a janela\nselecionada (|ativação|, ou contagem de disparos na coluna LIF; barra de cores à\nesquerda). <b>Arestas</b> = os pesos da camada Linear que conecta, desenhados só\npara os poucos maiores |peso| por neurônio de destino, para a figura ficar legível.\n<br><br>\n<b>Controles.</b> K arestas por neurônio, limiar de |peso|, limiar de atividade.\nAperte ▶ na barra de transporte para <b>inundar o sinal pelas camadas</b> uma\ncoluna por vez. LIF = neurônio de disparo Integra-e-Dispara com Vazamento.\n",
    "\n<b>What this shows.</b> Every window of one meeting01 <b>fold</b> pushed through\nthe trained autoencoder, then its 32-number <b>latent</b> vector squeezed to 2-D\n(or 3-D) so you can see the structure.\n<br><br>\n<b>This is a PROJECTED view</b>, not a pipeline number.\n<b>PCA</b> (Principal Component Analysis) rotates onto the directions of greatest\nspread — distances stay roughly meaningful, and the title shows how much\nvariability the 2 axes keep. <b>t-SNE</b> only preserves <i>who is near whom</i>;\ngaps and cluster sizes are not meaningful.\n<br><br>\n<b>Colour</b> = digit spoken, or speaker (legend/​controls). Well-separated\ncolours mean the latent space encodes that property. <b>Click a point</b> to\nselect that window everywhere else in the app.\n<br><br>\nThe forward passes run on a background thread — press <i>Project</i> and wait for\nthe status line.\n":
        "\n<b>O que isto mostra.</b> Cada janela de uma <b>dobra</b> do meeting01 passada\npelo autoencoder treinado, e então seu vetor <b>latente</b> de 32 números\ncomprimido para 2-D (ou 3-D) para você ver a estrutura.\n<br><br>\n<b>Esta é uma visão PROJETADA</b>, não um número do pipeline.\n<b>PCA</b> (Análise de Componentes Principais) rotaciona para as direções de maior\ndispersão — as distâncias continuam mais ou menos significativas, e o título\nmostra quanta variabilidade os 2 eixos mantêm. <b>t-SNE</b> só preserva <i>quem\nestá perto de quem</i>; espaços e tamanhos de aglomerados não têm significado.\n<br><br>\n<b>Cor</b> = dígito falado, ou locutor (legenda/​controles). Cores bem\nseparadas significam que o espaço latente codifica aquela propriedade.\n<b>Clique num ponto</b> para selecionar aquela janela em todo o resto do app.\n<br><br>\nAs passagens diretas rodam em uma thread de fundo — aperte <i>Projetar</i> e\naguarde a linha de status.\n",
    '\n<b>What this shows.</b> One audio <b>window</b> (a 256-sample slice) after it has\nbeen pushed through a trained <b>SNN autoencoder</b> (SNN = Spiking Neural\nNetwork; an autoencoder squeezes the window into a small <b>latent</b> vector of\n32 numbers and then rebuilds it).\n<br><br>\n<b>Top plot.</b> Blue = the <i>original</i> encoder input (the flattened,\nspike-encoded window). Orange = the <i>reconstruction</i> the decoder produced\nfrom the latent vector. The closer they sit, the more information the 32-number\nlatent kept.\n<br><b>Bottom plot.</b> The <i>residual</i> = original − reconstruction, point by\npoint. A flat line near zero means a faithful rebuild; spikes mark where the\nmodel lost detail.\n<br><br>\n<b>The numbers</b> (table below): every one has a "what it means" column.\nMSE / MAE measure the error size (0 = perfect); R² is the fraction of the\nsignal\'s variability captured (1 = perfect, 0 = no better than a flat line);\nPearson r is shape agreement ignoring scale (+1 = identical shape).\n<br><br>\n<b>Origin tag</b> <code>[computed]</code> means these curves were recomputed here\nthrough the exact experiment code, not read from a cached file.\n':
        '\n<b>O que isto mostra.</b> Uma <b>janela</b> de áudio (uma fatia de 256 amostras)\ndepois de passar por um <b>autoencoder SNN</b> treinado (SNN = Rede Neural de\nDisparos; um autoencoder comprime a janela num pequeno vetor <b>latente</b> de 32\nnúmeros e então a reconstrói).\n<br><br>\n<b>Gráfico de cima.</b> Azul = a entrada <i>original</i> do codificador (a janela\nachatada e codificada em disparos). Laranja = a <i>reconstrução</i> que o\ndecodificador produziu a partir do vetor latente. Quanto mais próximos, mais\ninformação os 32 números latentes guardaram.\n<br><b>Gráfico de baixo.</b> O <i>resíduo</i> = original − reconstrução, ponto a\nponto. Uma linha reta perto de zero significa uma reconstrução fiel; picos marcam\nonde o modelo perdeu detalhe.\n<br><br>\n<b>Os números</b> (tabela abaixo): cada um tem uma coluna "o que significa".\nMSE / MAE medem o tamanho do erro (0 = perfeito); R² é a fração da variabilidade\ndo sinal capturada (1 = perfeito, 0 = nada melhor que uma linha reta); Pearson r é\na concordância de forma ignorando escala (+1 = forma idêntica).\n<br><br>\n<b>Etiqueta de origem</b> <code>[computed]</code> significa que estas curvas foram\nrecalculadas aqui pelo código exato do experimento, não lidas de um arquivo em cache.\n',
    '\n<b>What this shows.</b> Each feature set placed on the <b>paraconsistent</b>\nplane. Paraconsistent logic lets a claim be supported <i>and</i> denied at once,\nwhich is exactly what noisy biometric evidence looks like.\n<br><br>\n<b>Axes.</b> Horizontal <b>G1 = α − β</b> (certainty): +1 = evidence firmly says\n"same person", −1 = firmly "different", 0 = undecided. Vertical\n<b>G2 = α + β − 1</b> (contradiction): +1 = evidence fully conflicts with itself,\n−1 = evidence missing, 0 = clean. α (alpha) is evidence <i>for</i>, β (beta) is\nevidence <i>against</i>, both 0…1.\n<br><br>\nThe ideal corner is <b>(G1 = 1, G2 = 0)</b> — certainly true, no contradiction.\n<b>D_truth</b> is the straight-line distance to it; <b>D_penalized</b> adds\n(2 − √2)·|G2| for contradiction and is the number the ranking sorts on. Smaller\nis better. Hover a point for all six quantities.\n':
        '\n<b>O que isto mostra.</b> Cada conjunto de atributos colocado no plano\n<b>paraconsistente</b>. A lógica paraconsistente permite que uma afirmação seja\napoiada <i>e</i> negada ao mesmo tempo — exatamente como é a evidência biométrica\nruidosa.\n<br><br>\n<b>Eixos.</b> Horizontal <b>G1 = α − β</b> (certeza): +1 = a evidência diz\nfirmemente "mesma pessoa", −1 = firmemente "diferente", 0 = indeciso. Vertical\n<b>G2 = α + β − 1</b> (contradição): +1 = a evidência se contradiz totalmente,\n−1 = evidência ausente, 0 = limpa. α (alfa) é evidência <i>a favor</i>, β (beta) é\nevidência <i>contra</i>, ambos 0…1.\n<br><br>\nO canto ideal é <b>(G1 = 1, G2 = 0)</b> — certamente verdadeiro, sem contradição.\n<b>D_truth</b> é a distância em linha reta até ele; <b>D_penalized</b> soma\n(2 − √2)·|G2| pela contradição e é o número pelo qual o ranking ordena. Menor é\nmelhor. Passe o mouse sobre um ponto para as seis quantidades.\n',
    '\n<b>What this shows.</b> Every persisted score as a point at\n(<b>D_truth</b>, <b>D_penalized</b>). The dashed diagonal is D_penalized =\nD_truth; the vertical gap above it <i>is</i> the contradiction penalty\n(2 − √2)·|G2|, so points far above the line have self-conflicting evidence.\nPoint colour scales with that gap.\n<br><br>\nD_truth = distance on the (G1, G2) plane to the ideal "certainly true, no\ncontradiction" corner. D_penalized = D_truth + the penalty; it is what the\nranking sorts on. Both: smaller is better.\n<br><br>\nThe seven facet filters (dataset, modality, wavelet, scale, fold, seed,\nstrategy) narrow the cloud. Click a point to inspect it.\n':
        '\n<b>O que isto mostra.</b> Cada nota persistida como um ponto em\n(<b>D_truth</b>, <b>D_penalized</b>). A diagonal tracejada é D_penalized =\nD_truth; o vão vertical acima dela <i>é</i> a penalidade de contradição\n(2 − √2)·|G2|, então pontos bem acima da linha têm evidência que se contradiz.\nA cor do ponto acompanha esse vão.\n<br><br>\nD_truth = distância no plano (G1, G2) até o canto ideal "certamente verdadeiro,\nsem contradição". D_penalized = D_truth + a penalidade; é o que o ranking ordena.\nAmbos: menor é melhor.\n<br><br>\nOs sete filtros de faceta (conjunto de dados, modalidade, wavelet, escala, dobra,\nseed, estratégia) reduzem a nuvem. Clique num ponto para inspecioná-lo.\n',
    '\n<b>What this shows.</b> Every persisted <b>paraconsistent</b> score from every\nexperiment in one sortable table: experiment, run, feature set, modality /\nencoding, seed, and then α, β, G1, G2, D_truth, <b>D_penalized</b>.\n<br><br>\nDefault sort is D_penalized ascending — <b>smaller is better</b> (it is the\ndistance to "certainly true, no contradiction", with a contradiction penalty).\nMissing values show as "—" and sort last, never as 0. Type in the filter box to\nnarrow by any text. Double-click a row to jump to that experiment.\n':
        '\n<b>O que isto mostra.</b> Cada nota <b>paraconsistente</b> persistida de cada\nexperimento numa tabela ordenável: experimento, execução, conjunto de atributos,\nmodalidade / codificação, seed, e então α, β, G1, G2, D_truth, <b>D_penalized</b>.\n<br><br>\nA ordem padrão é D_penalized crescente — <b>menor é melhor</b> (é a distância até\n"certamente verdadeiro, sem contradição", com uma penalidade de contradição).\nValores ausentes aparecem como "—" e vão para o fim, nunca como 0. Digite na\ncaixa de filtro para reduzir por qualquer texto. Duplo clique numa linha para ir\nàquele experimento.\n',
    '\n<b>What this shows.</b> The meeting01 run as a tree: session → dataset / fold →\nconfig → epoch, built live from the structured event log.\n<br><br>\nSelecting a config plots its learning curve below: <b>train loss</b> and\n<b>validation loss</b> per epoch (loss = reconstruction error the optimiser\nminimises), a dashed line at the best epoch, epoch <b>duration</b> on the right\naxis, and the learning rate in the title. A rising validation curve while train\nkeeps falling is the classic overfitting shape — but the tool only shows it, it\ndoes not label it.\n<br><br>\nEmpty until a LOSO run has written <code>results/meeting01/*_events.jsonl</code>.\n':
        '\n<b>O que isto mostra.</b> A execução do meeting01 como uma árvore: sessão →\nconjunto de dados / dobra → configuração → época, construída ao vivo a partir do\nlog estruturado de eventos.\n<br><br>\nSelecionar uma configuração desenha sua curva de aprendizado abaixo: <b>perda de\ntreino</b> e <b>perda de validação</b> por época (perda = erro de reconstrução que\no otimizador minimiza), uma linha tracejada na melhor época, a <b>duração</b> da\népoca no eixo direito, e a taxa de aprendizado no título. Uma curva de validação\nsubindo enquanto o treino segue caindo é o formato clássico de sobreajuste — mas\na ferramenta só o mostra, não o rotula.\n<br><br>\nVazio até uma execução LOSO gravar <code>results/meeting01/*_events.jsonl</code>.\n',
    '\n<b>What this shows.</b> The result of <b>NSGA-II</b> (Non-dominated Sorting\nGenetic Algorithm II), a multi-objective architecture search: it evolves a\npopulation and keeps the candidates that are not beaten on every objective at\nonce.\n<br><br>\n<b>Axes.</b> Vertical = D_penalized mean (paraconsistent quality, smaller\nbetter). Horizontal = the cost you pick: <b>inference cost</b> (spikes + 10 ×\nmultiply-accumulates, hardware-independent), parameter count, estimated latency,\nor latent activity.\n<br><br>\n<b>Green</b> = <b>feasible</b> Pareto-front points (satisfy every hard\nconstraint); <b>orange</b> = infeasible — shown for context, never winners.\nEstimated latency is labelled <b>UNCALIBRATED</b>: it is a rough model output,\nnot a measured millisecond figure — the view never ranks by it. Click a point\nfor its genome and fitness.\n':
        '\n<b>O que isto mostra.</b> O resultado do <b>NSGA-II</b> (Non-dominated Sorting\nGenetic Algorithm II), uma busca de arquitetura multiobjetivo: ele evolui uma\npopulação e mantém os candidatos que não são vencidos em todos os objetivos ao\nmesmo tempo.\n<br><br>\n<b>Eixos.</b> Vertical = média de D_penalized (qualidade paraconsistente, menor é\nmelhor). Horizontal = o custo que você escolhe: <b>custo de inferência</b>\n(disparos + 10 × multiplicações-acumulações, independente de hardware), número de\nparâmetros, latência estimada, ou atividade latente.\n<br><br>\n<b>Verde</b> = pontos <b>viáveis</b> da fronteira de Pareto (satisfazem cada\nrestrição rígida); <b>laranja</b> = inviáveis — mostrados por contexto, nunca\nvencedores. A latência estimada é rotulada <b>NÃO CALIBRADA</b>: é uma saída\naproximada do modelo, não um valor medido em milissegundos — a visão nunca ordena\npor ela. Clique num ponto para seu genoma e aptidão.\n',
    '\n<b>What this shows.</b> A structural, side-by-side comparison of the two\npipelines. It is <b>not</b> a claim that they are scientifically equivalent.\n<br><br>\nEach row is one aspect (dataset, windowing, encoding, wavelet, the autoencoder\nfamilies, cross-validation, …). The relation column is one of:\n<b>same concept</b>, <b>similar implementation</b>, <b>different\nimplementation</b>, <b>not applicable</b>, <b>unknown</b> — colour-coded, with the\nsource of the claim in the tooltip. The wording is deliberately conservative.\n':
        '\n<b>O que isto mostra.</b> Uma comparação estrutural, lado a lado, dos dois\npipelines. <b>Não</b> é uma afirmação de que são cientificamente equivalentes.\n<br><br>\nCada linha é um aspecto (conjunto de dados, janelamento, codificação, wavelet, as\nfamílias de autoencoder, validação cruzada, …). A coluna de relação é uma de:\n<b>mesmo conceito</b>, <b>implementação semelhante</b>, <b>implementação\ndiferente</b>, <b>não aplicável</b>, <b>desconhecido</b> — com código de cor e a\nfonte da afirmação na dica de contexto. A redação é deliberadamente conservadora.\n',
    '\n<b>What this shows.</b> One thesis sample seen through three synchronised lenses\nat once — scrub the sample index and all three panels move together.\n<br><br>\n<b>Wavelet</b> (left) — the per-band <b>energy</b> of that sample\'s signal.\n<b>Features</b> (middle) — that sample\'s handcrafted feature vector as a bar\nchart. <b>Paraconsistent</b> (right) — the run\'s feature set on the G1×G2 plane\n(G1 = certainty α−β, G2 = contradiction α+β−1).\n<br><br>\nWhen the feature layout lines up 1:1 with the wavelet bands, clicking a feature\nbar highlights the band it came from, and vice versa — the "why did this number\nbecome this number" link.\n':
        '\n<b>O que isto mostra.</b> Uma amostra da tese vista por três lentes sincronizadas\nao mesmo tempo — arraste o índice da amostra e os três painéis se movem juntos.\n<br><br>\n<b>Wavelet</b> (esquerda) — a <b>energia</b> por faixa do sinal daquela amostra.\n<b>Atributos</b> (meio) — o vetor de atributos feitos à mão daquela amostra como\num gráfico de barras. <b>Paraconsistente</b> (direita) — o conjunto de atributos\nda execução no plano G1×G2 (G1 = certeza α−β, G2 = contradição α+β−1).\n<br><br>\nQuando o arranjo dos atributos casa 1:1 com as faixas da wavelet, clicar numa\nbarra de atributo destaca a faixa de onde ela veio, e vice-versa — o elo do "por\nque este número virou este número".\n',

    # ---- glossary: expansions --------------------------------
    "Leave-One-Speaker-Out": "Deixar-Um-Locutor-de-Fora",
    "Autoencoder": "Autoencoder (autocodificador)",
    "Spiking Neural Network": "Rede Neural de Disparos",
    "Long Short-Term Memory": "Memória de Longo-Curto Prazo",
    "Gated Recurrent Unit": "Unidade Recorrente com Portas",
    "Principal Component Analysis": "Análise de Componentes Principais",
    "t-distributed Stochastic Neighbour Embedding":
        "Imersão Estocástica de Vizinhos com distribuição t",
    "voltage threshold": "limiar de voltagem",
    "Leaky Integrate-and-Fire": "Integra-e-Dispara com Vazamento",
    "membrane voltage": "voltagem de membrana",
    "Discrete Tunable Wavelet Packet Transform":
        "Transformada Discreta Ajustável em Pacote de Wavelet",
    "Zero-Crossing Rate": "Taxa de Cruzamentos por Zero",
    "Teager–Kaiser energy operator": "operador de energia de Teager–Kaiser",
    "Linear-Frequency Cepstral Coefficients":
        "Coeficientes Cepstrais de Frequência Linear",
    "evidence FOR (α)": "evidência A FAVOR (α)",
    "evidence AGAINST (β)": "evidência CONTRA (β)",
    "certainty degree (G1 = α − β)": "grau de certeza (G1 = α − β)",
    "contradiction degree (G2 = α + β − 1)": "grau de contradição (G2 = α + β − 1)",
    "distance to truth": "distância até a verdade",
    "penalised distance": "distância penalizada",
    "Mean Squared Error": "Erro Quadrático Médio",
    "Mean Absolute Error": "Erro Absoluto Médio",
    "R² (coefficient of determination)": "R² (coeficiente de determinação)",
    "Pearson correlation coefficient": "coeficiente de correlação de Pearson",
    "Equal Error Rate": "Taxa de Erro Igual",
    "Area Under the ROC Curve": "Área Sob a Curva ROC",
    "Non-dominated Sorting Genetic Algorithm II":
        "Algoritmo Genético de Ordenação Não-Dominada II",

    # ---- glossary: meanings ----------------------------------
    "The conference-paper pipeline: it trains autoencoders (SNN vs LSTM/GRU/Transformer) to reconstruct short audio windows and compares them speaker-by-speaker.":
        "O pipeline do artigo de congresso: treina autoencoders (SNN vs LSTM/GRU/Transformer) para reconstruir janelas curtas de áudio e os compara locutor por locutor.",
    "The doctoral pipeline: handcrafted wavelet features on EEG/voice, scored by paraconsistent logic, then used for person authentication.":
        "O pipeline do doutorado: atributos de wavelet feitos à mão sobre EEG/voz, avaliados por lógica paraconsistente e então usados para autenticação de pessoas.",
    "A way to test fairly: every speaker is held out once as the test set while the model trains on all the others, so the score is 'how well does it work on a person it never saw'.":
        "Uma forma justa de testar: cada locutor é deixado de fora uma vez como conjunto de teste enquanto o modelo treina com todos os outros, então a nota é 'quão bem funciona numa pessoa que nunca viu'.",
    "One split of the data into train / validation / test. Results are averaged over all folds.":
        "Uma divisão dos dados em treino / validação / teste. Os resultados são a média de todas as dobras.",
    "The train/test split. 'Outer' because a second, inner split (train/validation) sits inside it for model selection.":
        "A divisão treino/teste. 'Externa' porque uma segunda divisão interna (treino/validação) fica dentro dela para a seleção do modelo.",
    "A short fixed-length slice of the raw signal (here 256 samples). The models work on windows, not whole recordings.":
        "Uma fatia curta de comprimento fixo do sinal bruto (aqui 256 amostras). Os modelos trabalham sobre janelas, não sobre gravações inteiras.",
    "A network that compresses its input to a small vector (the latent) and then rebuilds it; good reconstruction means the latent kept the important information.":
        "Uma rede que comprime sua entrada num vetor pequeno (o latente) e então a reconstrói; uma boa reconstrução significa que o latente guardou a informação importante.",
    "A network whose neurons communicate with discrete 0/1 'spikes' over time instead of continuous numbers — closer to biology and cheaper on neuromorphic hardware.":
        "Uma rede cujos neurônios se comunicam com 'disparos' discretos 0/1 ao longo do tempo em vez de números contínuos — mais perto da biologia e mais barata em hardware neuromórfico.",
    "A classic recurrent network for sequences.":
        "Uma rede recorrente clássica para sequências.",
    "A lighter recurrent network, similar to LSTM.":
        "Uma rede recorrente mais leve, parecida com a LSTM.",
    "An attention-based sequence model (the architecture behind modern language models).":
        "Um modelo de sequência baseado em atenção (a arquitetura por trás dos modelos de linguagem modernos).",
    "A rotation of the data onto the directions of greatest spread; the first few directions give a faithful low-dimensional picture. A projection, not a measured value.":
        "Uma rotação dos dados para as direções de maior dispersão; as primeiras direções dão uma figura fiel em baixa dimensão. Uma projeção, não um valor medido.",
    "A non-linear 2-D layout that keeps near points near; good for spotting clusters, but distances and densities between clusters are not meaningful.":
        "Um arranjo não linear em 2-D que mantém pontos próximos próximos; bom para enxergar aglomerados, mas distâncias e densidades entre aglomerados não têm significado.",
    "The small vector an autoencoder's encoder produces (here 32 numbers). It is the model's compressed description of the window.":
        "O vetor pequeno que o codificador de um autoencoder produz (aqui 32 números). É a descrição comprimida da janela feita pelo modelo.",
    "The decoder's attempt to rebuild the original input from the latent vector.":
        "A tentativa do decodificador de reconstruir a entrada original a partir do vetor latente.",
    "Original minus reconstruction, point by point. Flat and near zero means a faithful rebuild.":
        "Original menos reconstrução, ponto a ponto. Reto e perto de zero significa uma reconstrução fiel.",
    "Spike encoding that passes the normalised sample straight through (no conversion to spike times).":
        "Codificação de disparo que passa a amostra normalizada sem alteração (sem conversão para tempos de disparo).",
    "Rate encoding: each value becomes a random spike train whose average firing rate is proportional to the value.":
        "Codificação por taxa: cada valor vira um trem de disparos aleatório cuja taxa média de disparo é proporcional ao valor.",
    "Time-to-first-spike encoding: larger values spike earlier.":
        "Codificação por tempo-até-o-primeiro-disparo: valores maiores disparam mais cedo.",
    "The membrane voltage a LIF neuron must reach to fire a spike. Higher threshold → fewer spikes.":
        "A voltagem de membrana que um neurônio LIF precisa atingir para emitir um disparo. Limiar mais alto → menos disparos.",
    "The membrane leak factor (0–1). Closer to 1 = the neuron remembers past input longer; closer to 0 = it forgets quickly.":
        "O fator de vazamento da membrana (0–1). Perto de 1 = o neurônio lembra a entrada passada por mais tempo; perto de 0 = esquece rápido.",
    "The standard spiking-neuron model: it adds up incoming current, leaks some away each step, and fires when it crosses the threshold, then resets.":
        "O modelo padrão de neurônio de disparo: soma a corrente que chega, vaza um pouco a cada passo, e dispara ao cruzar o limiar, depois se reinicia.",
    "The running internal voltage of a spiking neuron. It rises with input, leaks between inputs, and resets after a spike.":
        "A voltagem interna corrente de um neurônio de disparo. Sobe com a entrada, vaza entre entradas, e se reinicia após um disparo.",
    "Same as membrane potential — the neuron's internal voltage.":
        "O mesmo que potencial de membrana — a voltagem interna do neurônio.",
    "How many discrete time steps one sample is unrolled over. meeting01's autoencoder uses 1 (the window is fed as a feature vector); the recurrent transform uses 256 (the window samples).":
        "Em quantos passos de tempo discretos uma amostra é desenrolada. O autoencoder do meeting01 usa 1 (a janela entra como vetor de atributos); a transformação recorrente usa 256 (as amostras da janela).",
    "A dot plot: one row per neuron, a mark at every time it fired.":
        "Um gráfico de pontos: uma linha por neurônio, uma marca em cada vez que ele disparou.",
    "Fraction of time steps on which a neuron spiked (0–1).":
        "Fração dos passos de tempo em que um neurônio disparou (0–1).",
    "A short oscillation used to split a signal into frequency bands at several scales at once (unlike a plain spectrum, it keeps time information).":
        "Uma oscilação curta usada para dividir um sinal em faixas de frequência em várias escalas ao mesmo tempo (diferente de um espectro simples, ela mantém a informação de tempo).",
    "A full binary tree of wavelet filters: every band is split again, giving equal-width sub-bands (leaves).":
        "Uma árvore binária completa de filtros de wavelet: cada faixa é dividida de novo, dando sub-faixas de largura igual (folhas).",
    "The wavelet-packet decomposition used by the thesis feature extractor.":
        "A decomposição em pacote de wavelet usada pelo extrator de atributos da tese.",
    "One sub-band at the bottom of the wavelet-packet tree — a narrow frequency range.":
        "Uma sub-faixa no fim da árvore do pacote de wavelet — um intervalo estreito de frequência.",
    "A limited range of frequencies produced by the wavelet decomposition.":
        "Um intervalo limitado de frequências produzido pela decomposição em wavelet.",
    "The simplest wavelet (a single square step). Fast, blocky.":
        "A wavelet mais simples (um único degrau quadrado). Rápida, quadriculada.",
    "A family of smooth wavelets; 'daub10' means 10 filter taps — higher number = smoother, wider support.":
        "Uma família de wavelets suaves; 'daub10' significa 10 coeficientes de filtro — número maior = mais suave, suporte mais largo.",
    "Sum of squared coefficients in a band — how much of the signal's power sits in that frequency range.":
        "Soma dos coeficientes ao quadrado numa faixa — quanto da potência do sinal está naquele intervalo de frequência.",
    "A band's energy divided by the total, so the bands sum to 1.":
        "A energia de uma faixa dividida pelo total, para que as faixas somem 1.",
    "How often the signal changes sign — a rough pitch / noisiness measure.":
        "Quantas vezes o sinal muda de sinal — uma medida grosseira de tom / ruído.",
    "How evenly spread the coefficients are. High = noise-like, low = a few dominant components.":
        "Quão uniformemente distribuídos estão os coeficientes. Alta = parece ruído, baixa = poucos componentes dominantes.",
    "An instantaneous energy estimate sensitive to both amplitude and frequency.":
        "Uma estimativa de energia instantânea sensível tanto à amplitude quanto à frequência.",
    "Cycle-to-cycle variation in period — a voice-quality measure.":
        "Variação de ciclo a ciclo no período — uma medida de qualidade da voz.",
    "Cycle-to-cycle variation in amplitude — a voice-quality measure.":
        "Variação de ciclo a ciclo na amplitude — uma medida de qualidade da voz.",
    "A compact spectral-shape descriptor on a linear frequency axis (the linear-axis cousin of MFCCs).":
        "Um descritor compacto do formato espectral num eixo de frequência linear (o primo de eixo linear dos MFCCs).",
    "Computed from the log spectrum via another transform; captures the overall spectral envelope.":
        "Calculado a partir do espectro em log por outra transformada; captura o envelope espectral geral.",
    "A logic that tolerates contradiction: a statement can be supported and denied at the same time. Used here to score how cleanly a feature set separates people.":
        "Uma lógica que tolera contradição: uma afirmação pode ser apoiada e negada ao mesmo tempo. Usada aqui para avaliar quão limpamente um conjunto de atributos separa pessoas.",
    "Degree to which the feature set supports 'same person' (0–1).":
        "Grau em que o conjunto de atributos apoia 'mesma pessoa' (0–1).",
    "Degree to which it supports 'different person' (0–1).":
        "Grau em que ele apoia 'pessoa diferente' (0–1).",
    "How decisive the evidence is. +1 = fully certain true, −1 = fully certain false, 0 = undecided.":
        "Quão decisiva é a evidência. +1 = totalmente certo verdadeiro, −1 = totalmente certo falso, 0 = indeciso.",
    "How much the evidence conflicts with itself. +1 = fully contradictory, −1 = fully missing, 0 = consistent.":
        "O quanto a evidência se contradiz. +1 = totalmente contraditória, −1 = totalmente ausente, 0 = consistente.",
    "Straight-line distance on the (G1, G2) plane from the point to the ideal 'certainly true, no contradiction' corner (1, 0). Smaller is better.":
        "Distância em linha reta no plano (G1, G2) do ponto até o canto ideal 'certamente verdadeiro, sem contradição' (1, 0). Menor é melhor.",
    "D_truth plus a penalty of (2 − √2) × |G2| for contradiction. This is the number the feature ranking sorts on — smaller is better.":
        "D_truth mais uma penalidade de (2 − √2) × |G2| pela contradição. É o número pelo qual o ranking de atributos ordena — menor é melhor.",
    "The constant (2 − √2) ≈ 0.5858 that turns contradiction |G2| into extra distance in D_penalized.":
        "A constante (2 − √2) ≈ 0,5858 que transforma a contradição |G2| em distância extra em D_penalized.",
    "Average of (original − reconstruction)². 0 = perfect; grows fast with large errors.":
        "Média de (original − reconstrução)². 0 = perfeito; cresce rápido com erros grandes.",
    "Average of |original − reconstruction|. 0 = perfect; in the signal's own units.":
        "Média de |original − reconstrução|. 0 = perfeito; nas unidades do próprio sinal.",
    "Fraction of the signal's variance the reconstruction captured. 1 = perfect, 0 = no better than a flat line, negative = worse than flat.":
        "Fração da variância do sinal que a reconstrução capturou. 1 = perfeito, 0 = nada melhor que uma linha reta, negativo = pior que reta.",
    "How well the two curves rise and fall together, ignoring scale. +1 = identical shape, 0 = unrelated.":
        "Quão bem as duas curvas sobem e descem juntas, ignorando a escala. +1 = forma idêntica, 0 = sem relação.",
    "How well original and reconstruction move together (−1…+1); +1 is identical shape.":
        "Quão bem o original e a reconstrução se movem juntos (−1…+1); +1 é forma idêntica.",
    "The operating point where the chance of wrongly accepting an impostor equals the chance of wrongly rejecting the genuine user. Lower is better.":
        "O ponto de operação em que a chance de aceitar um impostor por engano é igual à chance de rejeitar o usuário genuíno por engano. Menor é melhor.",
    "Probability that a genuine trial scores above an impostor trial. 1 = perfect, 0.5 = chance.":
        "Probabilidade de uma tentativa genuína pontuar acima de uma tentativa impostora. 1 = perfeito, 0,5 = acaso.",
    "The share of total spread a PCA component accounts for; the first two components' shares tell you how much the 2-D picture leaves out.":
        "A fração da dispersão total que um componente do PCA explica; as frações dos dois primeiros componentes dizem quanto a figura 2-D deixa de fora.",
    "A multi-objective search: it evolves a population of architectures and keeps the ones that are not beaten on every objective at once.":
        "Uma busca multiobjetivo: evolui uma população de arquiteturas e mantém as que não são vencidas em todos os objetivos ao mesmo tempo.",
    "The set of solutions where you cannot improve one objective without worsening another — the best available trade-offs.":
        "O conjunto de soluções em que não se pode melhorar um objetivo sem piorar outro — os melhores compromissos disponíveis.",
    "The encoded description of one candidate architecture the search evolves.":
        "A descrição codificada de uma arquitetura candidata que a busca evolui.",
    "A candidate that satisfies every hard constraint (e.g. a latency budget); infeasible ones are shown but never win.":
        "Um candidato que satisfaz cada restrição rígida (p. ex. um orçamento de latência); os inviáveis são mostrados mas nunca vencem.",
    "A hardware-independent proxy for how expensive one forward pass is: spike count plus 10 × multiply-accumulates.":
        "Um indicador independente de hardware de quão cara é uma passagem direta: contagem de disparos mais 10 × multiplicações-acumulações.",
    "The estimated latency is a rough model output, not measured on real hardware — do not rank or quote it as a real millisecond figure.":
        "A latência estimada é uma saída aproximada do modelo, não medida em hardware real — não a use para ranquear nem a cite como um valor real em milissegundos.",
    "A value read straight from a saved experiment artifact.":
        "Um valor lido diretamente de um artefato salvo do experimento.",
    "A value recomputed here through the exact experiment C++ code — deterministic, matches the experiment bit-for-bit.":
        "Um valor recalculado aqui pelo código C++ exato do experimento — determinístico, bate com o experimento bit a bit.",
    "A lossy 2-D/3-D view (PCA, t-SNE) of higher-dimensional data — useful for the eye, not a pipeline number.":
        "Uma visão com perdas em 2-D/3-D (PCA, t-SNE) de dados de dimensão maior — útil para o olho, não um número do pipeline.",
    "A value flagged as an approximation (e.g. LIF parameters from a checkpoint that predates a fix).":
        "Um valor sinalizado como aproximação (p. ex. parâmetros LIF de um checkpoint anterior a uma correção).",
    "The curve you see is decimated for speed; the cursor still reads the full-resolution array.":
        "A curva que você vê foi decimada por velocidade; o cursor ainda lê o vetor em resolução total.",
    "No value is available. Shown as '—', never as 0.":
        "Nenhum valor disponível. Mostrado como '—', nunca como 0.",
    "The signal after subtracting its mean and dividing by its standard deviation, so it is centred on 0 with spread 1 (the models see it this way).":
        "O sinal depois de subtrair sua média e dividir pelo seu desvio padrão, ficando centrado em 0 com dispersão 1 (os modelos o veem assim).",
    "The random-number-generator starting value. Fixing it makes a run reproducible.":
        "O valor inicial do gerador de números aleatórios. Fixá-lo torna uma execução reproduzível.",

    # ---- view controls: shared -------------------------------
    "seed": "seed", "show": "mostrar", "encoding": "codificação",
    "wavelet": "wavelet", "level": "nível", "mode": "modo", "channel": "canal",
    "feature": "atributo", "sample": "amostra", "mean": "média", "std": "desv. pad.",
    "min / max": "mín / máx", "time step": "passo de tempo", "amplitude": "amplitude",
    "compare all": "comparar todas",
    "Select a meeting01 window.": "Selecione uma janela do meeting01.",
    "Select a Phase-00 handcrafted run.":
        "Selecione uma execução feita à mão da Fase-00.",

    # ---- view controls: encoding lab -------------------------
    "Encoding Lab needs a meeting01 window.":
        "O Laboratório de Codificação precisa de uma janela do meeting01.",
    "time step (sample within the window)": "passo de tempo (amostra dentro da janela)",
    "amplitude / spike": "amplitude / disparo",
    "   [{enc} encoding]": "   [codificação {enc}]",
    "encode failed: {exc}": "falha na codificação: {exc}",
    "{label}  [computed] — seed {seed}": "{label}  [calculado] — seed {seed}",

    # ---- view controls: wavelet lab -------------------------
    "No signal selected.": "Nenhum sinal selecionado.",
    "Leaf": "Folha", "Energy": "Energia", "Rel. energy": "Energia rel.",
    "coefficient index (sample within the band)":
        "índice do coeficiente (amostra dentro da faixa)",
    "coefficient value": "valor do coeficiente",
    "Selected object has no 1-D signal to decompose.":
        "O objeto selecionado não tem sinal 1-D para decompor.",
    "   ·   {mode} {wavelet}, {n} bands, {ns} samples in  ·  [computed]":
        "   ·   {mode} {wavelet}, {n} faixas, {ns} amostras na entrada  ·  [calculado]",
    "Frequency band {idx} on its own — {n} numbers (band 0 = lowest pitch)":
        "Faixa de frequência {idx} sozinha — {n} números (faixa 0 = som mais grave)",

    # ---- view controls: signal view -------------------------
    "Play this waveform through the default audio output. The \N{BLACK RIGHT-POINTING TRIANGLE} transport below only steps animation frames — it is not sound.":
        "Toca esta onda pela saída de áudio padrão. O transporte \N{BLACK RIGHT-POINTING TRIANGLE} abaixo apenas avança quadros de animação — não é som.",
    "This object has no raw signal.": "Este objeto não tem sinal bruto.",
    "load_signal failed: {exc}": "load_signal falhou: {exc}",
    "  · amplitude peak-normalised for listening":
        "  · amplitude normalizada pelo pico para a escuta",
    "playing {ms} ms at {hz} Hz": "tocando {ms} ms a {hz} Hz",
    "not audio (multi-channel or sub-3kHz) — nothing to play":
        "não é áudio (multicanal ou abaixo de 3 kHz) — nada para tocar",
    "QtMultimedia not installed — see Listen tooltip":
        "QtMultimedia não instalado — veja a dica do botão Ouvir",
    "  · DISPLAY-DOWNSAMPLED 1:{stride} ({n}→{m} pts; cursor reads full-res)":
        "  · EXIBIÇÃO REDUZIDA 1:{stride} ({n}→{m} pts; o cursor lê em resolução total)",

    # ---- view controls: snn lab ----------------------------
    "click a spike marker for its exact time / membrane / threshold":
        "clique num marcador de disparo para o tempo / membrana / limiar exatos",
    "SNN Lab needs a meeting01 window.":
        "O Laboratório de SNN precisa de uma janela do meeting01.",
    "transform failed: {exc}": "a transformação falhou: {exc}",
    "1 · the window going in": "1 · a janela que entra",
    "2 · spikes going in — {n} of them (click one to inspect)":
        "2 · disparos que entram — {n} deles (clique num para inspecionar)",
    "  (leak α={a}, firing line v_th={v})":
        "  (vazamento α={a}, linha de disparo v_th={v})",
    "charge inside the neuron  (membrane potential v[t])":
        "carga dentro do neurônio  (potencial de membrana v[t])",
    "4 · the trained network's 64 encoder neurons — {n} of them fired for this window (one charge value each, not a trajectory)":
        "4 · os 64 neurônios codificadores da rede treinada — {n} deles dispararam para esta janela (um valor de carga cada, não uma trajetória)",
    "encoder neuron index": "índice do neurônio codificador",
    "charge at readout": "carga na leitura",
    "; trained encoder LIF membrane shown ({n} neurons)":
        "; membrana LIF do codificador treinado mostrada ({n} neurônios)",
    "; no trained .npz for this fold — membrane panel hidden":
        "; sem .npz treinado para esta dobra — painel da membrana oculto",
    "{label}  [computed] — in {i} → out {o} spikes ({k} coincident, {a} LIF-added, {s} LIF-suppressed)":
        "{label}  [calculado] — entrada {i} → saída {o} disparos ({k} coincidentes, {a} acrescentados pelo LIF, {s} suprimidos pelo LIF)",
    "input encoding spike — t = sample {t}  ·  encoding = {enc}  ·  seed {seed}  (no membrane at the encoder input; it is a fixed spike train)":
        "disparo da codificação de entrada — t = amostra {t}  ·  codificação = {enc}  ·  seed {seed}  (sem membrana na entrada do codificador; é um trem de disparos fixo)",
    "recurrent-LIF spike — t = sample {t}  ·  v[t] = {v}  ·  v_th = {vth}  ·  crossed by {d}  ·  layer = recurrent transform":
        "disparo LIF recorrente — t = amostra {t}  ·  v[t] = {v}  ·  v_th = {vth}  ·  cruzou por {d}  ·  camada = transformação recorrente",

    # ---- follow-data bar ----------------------------------
    "RAW": "BRUTO", "WINDOW": "JANELA", "NORMALIZED": "NORMALIZADO",
    "ENCODING": "CODIFICAÇÃO", "WAVELET": "WAVELET", "FEATURES": "ATRIBUTOS",
    "PARACONSISTENT": "PARACONSISTENTE", "LATENT": "LATENTE",
    "RECONSTRUCTION": "RECONSTRUÇÃO", "CLASSIFICATION": "CLASSIFICAÇÃO",
    "not applicable to this selection": "não se aplica a esta seleção",
    "no selection": "nenhuma seleção",
    "this sample is not windowed here": "esta amostra não é janelada aqui",
    "spike encoding is a meeting01 window step":
        "a codificação de disparos é um passo de janela do meeting01",
    "select an individual window / sample":
        "selecione uma janela / amostra individual",
    "select the run node to recompute its feature matrix":
        "selecione o nó da execução para recalcular sua matriz de atributos",
    "handcrafted-feature matrix is a thesis-run view":
        "a matriz de atributos feitos à mão é uma visão de execução da tese",
    "needs a trained model .npz (not yet available)":
        "precisa de um modelo treinado .npz (ainda não disponível)",

    # ---- view controls: feature matrix ---------------------
    "per-column z-score (display only)": "z-score por coluna (só exibição)",
    "No live feature matrix for this object.":
        "Sem matriz de atributos ao vivo para este objeto.",
    "load_features failed: {exc}": "load_features falhou: {exc}",
    "{label}  [{origin}] — {r} samples x {c} features":
        "{label}  [{origin}] — {r} amostras x {c} atributos",
    "{name} @ {sample}  =  {val}  (class {cls})  [{origin}]":
        "{name} @ {sample}  =  {val}  (classe {cls})  [{origin}]",

    # ---- docks: inspector / reproduce / bookmarks / search ---
    "Field": "Campo", "Value": "Valor", "Origin": "Origem",
    "Add current…": "Adicionar atual…", "Remove": "Remover",
    "Add bookmark": "Adicionar marcador", "Name:": "Nome:",
    "Select a run.": "Selecione uma execução.",
    "Configuration": "Configuração",
    "Copy result-file paths": "Copiar caminhos dos arquivos de resultado",
    "Reproduction command": "Comando de reprodução",
    "not executed by this app (§31)": "não executado por este app (§31)",
    "Copy command": "Copiar comando",
    "Copied ✓": "Copiado ✓",
    "Select a run / fold node to see its reproduction recipe.":
        "Selecione um nó de execução / dobra para ver sua receita de reprodução.",
    "(no persisted result files)": "(sem arquivos de resultado persistidos)",
    'search — e.g. "daub10 lfcc eeg" or "fold 0 fsdd"':
        'buscar — p. ex. "daub10 lfcc eeg" ou "fold 0 fsdd"',
    "{shown} of {total} match(es)": "{shown} de {total} correspondência(s)",
    " — refine to see more": " — refine para ver mais",
}
