# -*- coding: utf-8 -*-
"""Catálogo em português (Brasil) para a interface do app.

As chaves são exatamente as strings em inglês passadas para ``i18n.t``. Uma
string sem entrada aqui volta para o inglês (cobertura parcial é segura). O
texto foi escrito para soar natural em português, não traduzido ao pé da letra.
"""

CATALOG = {
    # ---- menus / ações / controles comuns ------------------------
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
    "Presentation mode": "Modo apresentação",
    "Export current view…": "Exportar este painel…",
    "▶  Start guided tour": "▶  Iniciar o tour guiado",
    "Explain the current view": "Explicar este painel",
    "Glossary (all terms)…": "Glossário (todos os termos)…",
    "Refresh paraconsistent views": "Recarregar os painéis paraconsistentes",
    "Reset panel layout": "Restaurar disposição dos painéis",
    "Panel layout reset.": "Disposição dos painéis restaurada.",
    "Back": "Voltar",
    "◀ Back": "◀ Voltar",
    "Next ▶": "Avançar ▶",
    "Finish ▶": "Concluir ▶",
    "Free explore": "Explorar à vontade",
    "Colour key:": "Legenda de cores:",
    "How to read this": "Como interpretar",
    "\N{SPEAKER WITH THREE SOUND WAVES}  Listen": "\N{SPEAKER WITH THREE SOUND WAVES}  Ouvir",
    "Tour ended — every tab is back. Explore freely.":
        "Tour concluído — todas as abas voltaram. Explore à vontade.",
    "opened the explanation for this view": "abri a explicação deste painel",
    "Language changed — some fixed labels update after a restart.":
        "Idioma alterado — alguns rótulos fixos só mudam depois de reiniciar.",

    # ---- tour guiado: meeting01 ----------------------------------
    "How a spiking network learns a spoken digit":
        "Como uma rede neural de disparos aprende um dígito falado",
    "1 · A spoken digit is just a wiggle of air":
        "1 · Um dígito falado nada mais é que o ar vibrando",
    "This is the digit spoken aloud, drawn as air pressure over time — the same "
    "thing your ear receives. Press <b>🔊 Listen</b> to hear it.<br><br>"
    "The whole goal: get a computer to tell digits apart <i>without ever being "
    "told</i> what each one sounds like.":
        "Este é o dígito falado em voz alta, traçado como a variação da pressão do "
        "ar ao longo do tempo — exatamente o que chega ao seu ouvido. Clique em "
        "<b>🔊 Ouvir</b> para escutá-lo.<br><br>"
        "A ideia é ensinar o computador a distinguir os dígitos <i>sem precisar dizer a ele "
        "como cada um deve soar</i>.",
    "256 numbers — one 32-millisecond slice of the sound.":
        "256 números — um trecho de 32 milissegundos do som.",
    "2 · Split the sound into frequency bands":
        "2 · Divida o som em faixas de frequência",
    "The same slice, broken into 16 <b>frequency bands</b> — low pitches on the "
    "left, high on the right (this split is called a <i>wavelet</i>).<br><br>"
    "Speech energy piles up in just a few bands. That pattern is a fingerprint "
    "of the sound — more useful to a model than the raw wiggle.":
        "O mesmo trecho, repartido em 16 <b>faixas de frequência</b> — os graves à "
        "esquerda, os agudos à direita (essa divisão se chama <i>wavelet</i>).<br><br>"
        "A maior parte da energia da fala se concentra em algumas faixas. Esse padrão "
        "funciona como uma espécie de assinatura do som e costuma ser mais informativo "
        "para o modelo do que a onda bruta.",
    "3 · Turn the numbers into spikes":
        "3 · Transforme os números em disparos",
    "Real neurons don't pass numbers around — they fire brief <b>spikes</b>. "
    "Here every value in the window is turned into a spike train.<br><br>"
    "With <i>latency</i> encoding, a louder value fires its spike earlier. "
    "Louder = sooner.":
        "Neurônios biológicos não ficam enviando números contínuos uns para os outros — "
        "eles se comunicam por breves <b>disparos</b>. Aqui, cada valor da janela é "
        "transformado em uma sequência de disparos ao longo do tempo.<br><br>"
        "Na codificação por <i>latência</i>, o valor determina quando ocorre o disparo: "
        "quanto maior o valor, mais cedo ele acontece. Em resumo: <b>valor maior → disparo mais cedo</b>.",
    "4 · One neuron that leaks, charges, and fires":
        "4 · Um neurônio que vaza, acumula carga e dispara",
    "Watch the green line: charge <b>builds up</b> inside the neuron as spikes "
    "arrive, and slowly <b>leaks away</b> between them. When it crosses the red "
    "line the neuron <b>fires</b> a spike and the charge drops.<br><br>"
    "That is a <i>leaky integrate-and-fire</i> neuron — the membrane-potential "
    "picture people ask about.":
        "Acompanhe a linha verde: a carga <b>se acumula</b> dentro do neurônio conforme "
        "os disparos chegam, mas também vai <b>diminuindo</b> entre eles. Quando ultrapassa "
        "a linha vermelha, o neurônio <b>dispara</b> e a carga é reduzida novamente.<br><br>"
        "Esse é o modelo de neurônio <i>Leaky Integrate-and-Fire</i> (LIF): ele integra a "
        "entrada, perde parte da carga com o tempo e dispara quando atinge o limiar.",
    "Green = charge, red = the firing line, amber ticks = spikes out.":
        "Verde = carga, vermelho = a linha de disparo, traços âmbar = disparos na saída.",
    "5 · Squeeze the whole window to 32 numbers":
        "5 · Comprima a janela inteira em 32 números",
    "The full network compresses each window down to just <b>32 numbers</b> — "
    "the <i>latent</i>. Press <b>Project</b>: every window of this fold is run "
    "through and drawn as a dot.<br><br>"
    "If the same digit lands in the same clump, those 32 numbers have captured "
    "what makes the digit that digit.":
        "A rede inteira transforma cada janela em apenas <b>32 números</b> — o <i>vetor "
        "latente</i>. Clique em <b>Projetar</b> para passar cada janela desta dobra pela "
        "rede e representá-la como um ponto no gráfico.<br><br>"
        "Se as janelas do mesmo dígito acabam próximas umas das outras, é um sinal de "
        "que esses 32 números preservaram informações importantes para distinguir aquele dígito.",
    "6 · Rebuild it — and see what was lost":
        "6 · Reconstrua — e veja o que se perdeu",
    "The decoder tries to redraw the original window from those 32 numbers "
    "alone. <span style='color:#4F9DF7'>Blue</span> = original, "
    "<span style='color:#F5A623'>orange</span> = rebuild, "
    "<span style='color:#969aa0'>grey</span> = the difference.<br><br>"
    "A grey line that barely moves means the 32 numbers kept almost everything.":
        "O decodificador tenta reconstruir a janela original usando apenas aqueles 32 "
        "números. <span style='color:#4F9DF7'>Azul</span> = original, "
        "<span style='color:#F5A623'>laranja</span> = reconstrução, "
        "<span style='color:#969aa0'>cinza</span> = a diferença entre as duas.<br><br>"
        "Se a linha cinza fica quase parada, a diferença entre original e reconstrução "
        "é pequena — sinal de que o vetor latente preservou boa parte da informação.",
    "You've seen the whole path":
        "Você percorreu o caminho inteiro",
    "Sound → frequency bands → spikes → one neuron's charge → 32 numbers → "
    "rebuild. <br><br>Now click <b>Free explore</b>: pick any window, any tab, "
    "and follow a single number all the way through. Every plot shows where "
    "its data came from.":
        "Som → faixas de frequência → disparos → carga de um neurônio → 32 números → "
        "reconstrução.<br><br>Agora clique em <b>Explorar à vontade</b>: escolha uma "
        "janela, abra qualquer aba e acompanhe um dado desde a entrada até o resultado. "
        "A ideia é conseguir enxergar de onde cada informação veio.",

    # ---- tour guiado: tese -------------------------------------
    "How wavelets + paraconsistent logic authenticate a person":
        "Como as wavelets e a lógica paraconsistente autenticam uma pessoa",
    "1 · Brain or voice signals from one person":
        "1 · Sinais de cérebro ou de voz de uma pessoa",
    "Six channels of <b>EEG</b> (tiny voltages from the scalp) or a voice "
    "recording. The goal: decide whether two recordings come from the "
    "<i>same person</i>.":
        "São seis canais de <b>EEG</b> (pequenas variações elétricas medidas no couro "
        "cabeludo) ou uma gravação de voz. O objetivo é decidir se duas gravações vieram "
        "da <i>mesma pessoa</i>.",
    "EEG here is 6 channels × 4096 samples at 1024 readings per second.":
        "Aqui o EEG tem 6 canais × 4096 amostras, a 1024 leituras por segundo.",
    "2 · Split into frequency bands":
        "2 · Divida em faixas de frequência",
    "The same wavelet idea as before: break each channel into frequency "
    "bands. The hand-designed path then measures several things in every band "
    "— energy, how often it crosses zero, how disordered it is, and more.":
        "A ideia é a mesma usada antes com wavelets: separar cada canal em diferentes "
        "faixas de frequência. Depois, o extrator feito à mão calcula várias medidas em "
        "cada faixa — por exemplo, energia, cruzamentos por zero e entropia, entre outras.",
    "3 · 96 hand-picked measurements per recording":
        "3 · 96 medidas escolhidas a dedo por gravação",
    "Each row is one recording, each column one measurement. Unlike the "
    "spoken-digit network, nothing here is learned — a human chose every "
    "measurement. This is the <i>handcrafted feature matrix</i>.":
        "Cada linha representa uma gravação e cada coluna representa uma medida. "
        "Diferentemente da rede de dígitos falados, essas medidas não foram descobertas "
        "pelo modelo: elas foram definidas previamente. O resultado é a <i>matriz de atributos manuais</i>.",
    "1974 recordings × 96 measurements for this run.":
        "1974 gravações × 96 medidas nesta execução.",
    "4 · Evidence can support AND deny at once":
        "4 · A evidência pode, ao mesmo tempo, confirmar E negar",
    "<b>Paraconsistent</b> logic allows a claim to be backed and contradicted "
    "at the same time — exactly what noisy biometrics look like.<br><br>"
    "Horizontal <b>G1</b> = net certainty (right = 'same person'). Vertical "
    "<b>G2</b> = how much the evidence fights itself. The ideal spot is the "
    "far right at zero height.":
        "A lógica <b>paraconsistente</b> permite representar uma situação em que existem "
        "evidências a favor e contra a mesma hipótese ao mesmo tempo. Isso é útil em "
        "biometria, porque dados reais podem ser ruidosos e diferentes medidas podem "
        "apontar para conclusões opostas.<br><br>"
        "Na horizontal, <b>G1</b> representa a certeza líquida (à direita = 'mesma pessoa'). "
        "Na vertical, <b>G2</b> representa o grau de contradição entre as evidências. "
        "O ponto ideal fica à direita, com G2 igual a zero.",
    "5 · Which recipe of measurements wins":
        "5 · Qual receita de medidas vence",
    "Every combination of wavelet + measurements gets one score: its distance "
    "to that ideal spot, with a penalty for self-contradiction "
    "(<i>D_penalized</i> — smaller is better).<br><br>"
    "This sorted table is the experiment's actual answer.":
        "Cada combinação de wavelet e medidas recebe uma pontuação. Ela representa a "
        "distância até o ponto ideal, com uma penalidade adicional quando existe contradição "
        "(<i>D_penalized</i> — quanto menor, melhor).<br><br>"
        "A tabela ordenada mostra quais combinações tiveram o melhor desempenho segundo esse critério.",
    "The row at the top has the least contradiction and the most certainty.":
        "A primeira linha é a combinação que melhor equilibra certeza e baixa contradição "
        "segundo o critério usado pelo experimento.",
    "Signal → frequency bands → 96 measurements → support-vs-denial plane → "
    "ranking. <br><br>Click <b>Free explore</b> and follow any feature set "
    "through the Triangle tab to see all three stages side by side.":
        "Sinal → faixas de frequência → 96 medidas → plano de confirmação × negação → "
        "ranking.<br><br>Clique em <b>Explorar à vontade</b> e acompanhe qualquer "
        "conjunto de atributos pela aba Triângulo para ver os três estágios lado a "
        "lado.",

    # ---- frases de veredito -----------------------------------
    "No rebuild yet — train a model fold first.":
        "Ainda não há reconstrução — treine uma dobra do modelo primeiro.",
    "Near-perfect rebuild — the small set of latent numbers kept almost everything (R² {r2}).":
        "Reconstrução quase perfeita — os poucos números do latente preservaram quase tudo (R² {r2}).",
    "Close rebuild — the shape is right, fine detail is softened (R² {r2}).":
        "Reconstrução fiel — a forma está certa, só os detalhes finos ficaram mais suaves (R² {r2}).",
    "Rough rebuild — the big movements survive, the sharp parts are lost (R² {r2}).":
        "Reconstrução aproximada — os movimentos grandes se mantêm, as partes bruscas se perdem (R² {r2}).",
    "Weak rebuild — close to a flat line at the mean (R² {r2}).":
        "Reconstrução fraca — pouco mais que uma linha reta na média (R² {r2}).",
    "Failed rebuild — R² {r2} is below zero, so a flat line at the mean would "
    "match the window more closely; the latent numbers did not capture it.":
        "A reconstrução falhou — o R² {r2} está abaixo de zero, ou seja, uma linha "
        "reta na média ficaria mais perto da janela; os números do latente não a "
        "capturaram.",
    "very sparse": "muito esparso", "sparse": "esparso",
    "moderate": "moderado", "dense": "denso",
    " — the neuron kept {pct}% of the incoming spikes":
        " — o neurônio manteve {pct}% dos disparos que chegaram",
    "{n_out} spikes out of {n_steps} time steps ({dens} firing){kept}. "
    "Each spike is one 'the neuron reacted here' moment.":
        "{n_out} disparos em {n_steps} passos de tempo (disparo {dens}){kept}. "
        "Cada disparo marca um instante em que o neurônio reagiu.",
    "clean": "limpa", "somewhat conflicting": "meio contraditória",
    "highly conflicting": "bastante contraditória",
    "supports the match": "aponta para a mesma pessoa",
    "denies the match": "aponta para pessoas diferentes",
    "is undecided": "não se decide",
    "Evidence {lean} (certainty G1 {g1}) and is {conflict} (contradiction G2 {g2}). "
    "α is support for, β is support against — both can be high at once.":
        "A evidência {lean} (certeza G1 {g1}) e está {conflict} (contradição G2 {g2}). "
        "α é o apoio a favor e β o apoio contra — os dois podem estar altos ao mesmo tempo.",
    "concentrated in a few bands": "concentrada em poucas faixas",
    "spread across many bands": "espalhada por muitas faixas",
    "The signal's energy is {spread}; band {top} alone holds {pct}%. "
    "Each band is a frequency range, low bands first.":
        "A energia do sinal está {spread}; só a faixa {top} concentra {pct}%. "
        "Cada faixa é um intervalo de frequência, começando pelas mais graves.",
    "Every window — hundreds of samples — is squeezed to just {dim} numbers. "
    "If similar digits land near each other here, those numbers carry the meaning.":
        "Cada janela — centenas de amostras — é reduzida a apenas {dim} números. "
        "Se dígitos parecidos aparecem juntos aqui, é porque esses números carregam o significado.",

    # ---- legenda de cores (palette.MEANING) --------------------
    "original signal": "sinal original",
    "model's rebuild": "reconstrução do modelo",
    "difference (how wrong)": "diferença (o tamanho do erro)",
    "a spike fired": "um disparo",
    "charge in the neuron": "carga no neurônio",
    "firing line": "linha de disparo",
    "frequency-band energy": "energia da faixa",
    "one measurement": "uma medida",
    "clean support": "apoio sem conflito",
    "support + denial at once": "apoio e negação juntos",
    "your selection": "sua seleção",

    # ---- alguns títulos de painel ----------------------------
    "The window we start from — a slice of the signal, mean 0":
        "A janela de onde partimos — um trecho do sinal, com média 0",
    "direct — passes the numbers straight through (no spikes)":
        "direto — deixa os números passarem como estão (sem disparos)",
    "The signal going in — pick a band on the left to see just that band":
        "O sinal que entra — escolha uma faixa à esquerda para ver só ela",
    "Each dot is one feature recipe · right = 'same person' · "
    "up = the evidence contradicts itself · bottom-right corner is the goal":
        "Cada ponto é uma receita de atributos · à direita = 'mesma pessoa' · "
        "para cima = a evidência se contradiz · o canto inferior direito é o alvo",

    # ---- progresso / ocupado --------------------------------
    "Working…": "Processando…",
    "Loading data…": "Carregando os dados…",
    "Running the network…": "Rodando a rede…",
    "Computing the wavelet…": "Calculando a wavelet…",
    "Projecting the latent space…": "Projetando o espaço latente…",
    "Computing…": "Calculando…",

    # ---- janela / status / menus ---------------------------
    "Experiment Microscope": "Microscópio de Experimentos",
    "FULL RESOLUTION": "RESOLUÇÃO TOTAL",
    "DISPLAY-DOWNSAMPLED": "EXIBIÇÃO REDUZIDA",
    "LOW-PERFORMANCE MODE": "MODO DE BAIXO DESEMPENHO",
    "cursor @ sample {ts}": "cursor na amostra {ts}",
    "No meeting01 run detected under results/meeting01/.":
        "Nenhuma execução do meeting01 encontrada em results/meeting01/.",
    "this view has no dedicated explanation yet — see Help → Glossary":
        "este painel ainda não tem explicação própria — veja Ajuda → Glossário",
    "Glossary — every abbreviation and metric":
        "Glossário — cada sigla e cada métrica",
    "Glossary": "Glossário",
    "Export figure": "Exportar figura",
    "current view has nothing to export": "não há nada para exportar neste painel",
    "export failed: {exc}": "falha ao exportar: {exc}",
    "wrote {written}": "salvo em {written}",
    "compute failed: {exc}": "falha no cálculo: {exc}",
    "bookmarked: {name}": "marcador criado: {name}",
    "restored: {name}": "restaurado: {name}",
    "(selection path not found — state only)":
        "(caminho da seleção não encontrado — restaurado só o estado)",
    "quantity": "quantidade", "value": "valor", "what it means": "o que significa",

    # ---- rótulos das abas ----------------------------------
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
    "Ranking": "Ranking",
    "Timeline": "Linha do tempo",

    # ---- títulos dos docks --------------------------------
    "Data Explorer": "Explorador de Dados",
    "Inspector": "Inspetor",
    "Raw artifact": "Arquivo bruto",
    "Reproduce": "Reproduzir",
    "Meeting01 session": "Sessão do Meeting01",
    "Bookmarks": "Marcadores",
    "Developer": "Desenvolvedor",
    "Guided tour": "Tour guiado",

    # ---- títulos das caixas de ajuda ----------------------
    "Raw signal": "Sinal bruto",
    "Experiment timeline": "Linha do tempo do experimento",
    "NSGA-II search": "Busca NSGA-II",
    "Triangle view": "Painel em triângulo",
    "Latent Space Explorer": "Explorador do Espaço Latente",
    "folds": "dobras",
    "Project": "Projetar",
    "split": "recorte",
    "max windows": "máx. de janelas",
    "method": "método",
    "colour": "cor",
    "Latent Space Explorer needs a meeting01 fold / window.":
        "O Explorador do Espaço Latente precisa de uma dobra/janela do meeting01.",
    "meeting01 › {ds} › fold {fold} — press Project.":
        "meeting01 › {ds} › dobra {fold} — aperte Projetar.",
    "Select a meeting01 fold first.": "Selecione uma dobra do meeting01 primeiro.",
    "running SNN-AE forward passes on a worker thread…":
        "rodando as passagens do SNN-AE numa thread em segundo plano…",
    "  — LIF params default (pre-fix checkpoint)":
        "  — parâmetros LIF padrão (checkpoint anterior à correção)",
    "  — skipped {n} fold(s) with no trained model: {which}":
        "  — {n} dobra(s) ignorada(s) por não ter modelo treinado: {which}",
    "{n} latents across {nf} fold(s) {folds} ({arch}/{enc}, dim {dim}) — "
    "projection is PROJECTED, not a pipeline value{note}{skip}":
        "{n} latentes em {nf} dobra(s) {folds} ({arch}/{enc}, dim {dim}) — a "
        "projeção é PROJETADA, não um valor do pipeline{note}{skip}",
    "Meeting01 vs Thesis": "Meeting01 × Tese",

    # ---- corpo das caixas de ajuda -----------------------
    '\n<b>What this shows.</b> The signal exactly as the pipeline sees it for the\nselected sample — an audio waveform, or several stacked EEG channels\n(EEG = electroencephalogram, brain electrical activity).\n<br><br>\n<b>Axes.</b> Horizontal = sample number (multiply by 1/sampling-rate for seconds;\nthe rate is in the title). Vertical = amplitude. For meeting01 windows the\namplitude is <b>z-scored</b> (mean 0, spread 1) because that is what the models\nreceive; the unit is shown on the left axis.\n<br><br>\n<b>Colours / lines.</b> One colour per channel, named in the legend. Drag on the\nplot to select a time range — downstream views can restrict to it. The vertical\ncursor line reports the exact value under it.\n<br><br>\n<b>DISPLAY-DOWNSAMPLED</b> in the title (and the status bar) means the drawn\ncurve is decimated for speed; the cursor still reads the full-resolution number.\n<br><br>\n<b>🔊 Listen</b> plays the waveform through your speakers (audio samples only;\nEEG has no sound). The <b>▶</b> transport at the bottom is separate — it steps\nanimation frames, it does not play audio.\n':
        '\n<b>O que este painel mostra.</b> O sinal exatamente como o pipeline o enxerga\npara a amostra selecionada — uma onda de áudio, ou vários canais de EEG\nempilhados (EEG = eletroencefalograma, a atividade elétrica do cérebro).\n<br><br>\n<b>Eixos.</b> Na horizontal, o número da amostra (multiplique por 1/taxa de\namostragem para ter segundos; a taxa está no título). Na vertical, a amplitude.\nNas janelas do meeting01 a amplitude está em <b>z-score</b> (média 0, dispersão\n1), que é como os modelos a recebem; a unidade aparece no eixo esquerdo.\n<br><br>\n<b>Cores e linhas.</b> Uma cor por canal, identificada na legenda. Arraste sobre\no gráfico para selecionar um intervalo de tempo — os painéis seguintes podem se\nrestringir a ele. A linha vertical do cursor informa o valor exato sob ela.\n<br><br>\n<b>EXIBIÇÃO REDUZIDA</b> no título (e na barra de status) quer dizer que a curva\ndesenhada foi reduzida para ganhar velocidade; o cursor continua lendo o valor em\nresolução total.\n<br><br>\n<b>🔊 Ouvir</b> toca a onda nos alto-falantes (só para amostras de áudio; o EEG\nnão tem som). O transporte <b>▶</b> lá embaixo é outra coisa — ele avança quadros\nde animação, não reproduz áudio.\n',
    "\n<b>What this shows.</b> The signal split into frequency <b>sub-bands</b> by a\n<b>wavelet packet</b> decomposition (a binary tree of filters; each leaf is a\nnarrow frequency range). Unlike a plain spectrum this keeps <i>when</i> things\nhappen, not only <i>what</i> frequencies are present.\n<br><br>\n<b>Controls.</b> <i>wavelet</i> picks the filter shape (Haar = blocky and fast;\ndaub4…daub20 = progressively smoother Daubechies filters). <i>level</i> = tree\ndepth, so 2<sup>level</sup> leaves. <i>channel</i> chooses which EEG channel to\ndecompose.\n<br><br>\n<b>Table / plot.</b> Per leaf: <b>energy</b> (sum of squared coefficients = power\nin that band) and <b>relative energy</b> (share of the total, bands sum to 1).\nClick a leaf to see its raw coefficients.\n<br><br>\n<b>Decomposition levels.</b> Below, a separate ladder: the final\n<b>approximation</b> plus every level's <b>detail</b> band, coarsest to finest —\nthis is the regular (non-packet) transform, always recomputed on its own\nregardless of the <i>mode</i> picker above, since packet leaves aren't\norganized into levels. Each row's coefficient count halves going down (real\ncounts, never resampled to match).\n":
        "\n<b>O que este painel mostra.</b> O sinal repartido em <b>sub-faixas</b> de\nfrequência por uma decomposição em <b>pacote de wavelet</b> (uma árvore binária\nde filtros; cada folha é uma faixa estreita de frequência). Diferente de um\nespectro comum, isto preserva <i>quando</i> as coisas acontecem, não só <i>quais</i>\nfrequências estão presentes.\n<br><br>\n<b>Controles.</b> <i>wavelet</i> define o formato do filtro (Haar = quadrado e\nrápido; daub4…daub20 = filtros de Daubechies cada vez mais suaves). <i>nível</i> é\na profundidade da árvore, ou seja, 2<sup>nível</sup> folhas. <i>canal</i> escolhe\nqual canal de EEG decompor.\n<br><br>\n<b>Tabela e gráfico.</b> Para cada folha: a <b>energia</b> (soma dos coeficientes\nao quadrado = a potência naquela faixa) e a <b>energia relativa</b> (a fração do\ntotal; as faixas somam 1). Clique numa folha para ver seus coeficientes brutos.\n<br><br>\n<b>Níveis de decomposição.</b> Abaixo, uma escada separada: a\n<b>aproximação</b> final mais a faixa de <b>detalhe</b> de cada nível, do mais\ngrosso ao mais fino — é a transformada regular (não pacote), sempre recalculada\npor conta própria, seja qual for o <i>modo</i> escolhido acima, já que folhas de\npacote não se organizam em níveis. A quantidade de coeficientes de cada linha\ncai pela metade a cada nível abaixo (contagens reais, nunca reamostradas para\ncombinar).\n",
    "\n<b>What this shows.</b> The wavelet-packet decomposition as a 3-D surface instead\nof a table.\n<br><br>\n<b>X</b> = coefficient index within a leaf. <b>Y</b> = leaf number, low to high\nfrequency. <b>Z</b> (height & colour) = |coefficient| magnitude, normalised so\nthe largest is 1. Ridges are bands carrying a lot of the signal's power.\n<br><br>\nThe <b>|z| threshold</b> slider hides small coefficients; <b>isolate leaf</b>\nlifts one band out as a red line. Rotate / zoom / pan with the mouse. Disabled in\nlow-performance mode.\n<br><br>\n<b>Axes.</b> The bounding box is real — its ticks are the true coefficient\nindex, leaf number, and |coefficient|/max ratio; nothing on it is a cosmetic\nscale factor.\n":
        "\n<b>O que este painel mostra.</b> A decomposição em pacote de wavelet como uma\nsuperfície 3-D, no lugar de uma tabela.\n<br><br>\n<b>X</b> = o índice do coeficiente dentro de uma folha. <b>Y</b> = o número da\nfolha, da frequência mais baixa para a mais alta. <b>Z</b> (altura e cor) = a\nmagnitude |coeficiente|, normalizada para que a maior seja 1. As cristas são as\nfaixas que concentram boa parte da potência do sinal.\n<br><br>\nO controle <b>limiar |z|</b> esconde os coeficientes pequenos; <b>isolar folha</b>\npuxa uma faixa para fora como uma linha vermelha. Use o mouse para girar,\naproximar e deslocar. Fica desativado no modo de baixo desempenho.\n<br><br>\n<b>Eixos.</b> A caixa delimitadora é real — suas marcações são o verdadeiro\níndice do coeficiente, o número da folha e a razão |coeficiente|/máximo; nada\nnela é um fator de escala cosmético.\n",
    "\n<b>What this shows.</b> The full table of handcrafted feature values for a thesis\nrun: one row per sample, one column per feature.\n<br><br>\n<b>Colours.</b> A heatmap — brighter / darker means larger / smaller value; the\ncolour bar gives the scale. The per-column statistics (mean, standard deviation,\nmin, max) sit beside it. Turning on the z-score toggle rescales <i>for display\nonly</i> so columns of different magnitude become comparable — the underlying\nnumbers do not change.\n<br><br>\n<b>Click a cell</b> to read its exact value, the sample's class, and (when the\nwavelet layout allows) which frequency band it came from.\n<br><br>\nFeature names encode the descriptor and scale, e.g. energy / <b>ZCR</b>\n(zero-crossing rate) / entropy / <b>Teager</b> (Teager–Kaiser energy) / jitter /\nshimmer per wavelet band.\n":
        "\n<b>O que este painel mostra.</b> A tabela completa dos valores dos atributos\nmanuais de uma execução da tese: uma linha por amostra, uma coluna por atributo.\n<br><br>\n<b>Cores.</b> Um mapa de calor — mais claro ou mais escuro indica valor maior ou\nmenor; a barra de cores dá a escala. Ao lado ficam as estatísticas de cada coluna\n(média, desvio padrão, mínimo, máximo). Ligar o z-score reescala <i>apenas na\nexibição</i>, para que colunas de magnitudes bem diferentes fiquem comparáveis —\nos números originais não mudam.\n<br><br>\n<b>Clique numa célula</b> para ver o valor exato, a classe da amostra e (quando o\narranjo da wavelet permite) de qual faixa de frequência ele veio.\n<br><br>\nOs nomes dos atributos codificam o descritor e a escala — por exemplo energia /\n<b>ZCR</b> (taxa de cruzamentos por zero) / entropia / <b>Teager</b> (energia de\nTeager–Kaiser) / jitter / shimmer, por faixa de wavelet.\n",
    '\n<b>What this shows.</b> The three spike <b>encodings</b> side by side for one\nwindow, so you can see how each turns a real number into spike events.\n<br><br>\n<b>direct</b> — the normalised sample passes straight through (no spike-time\nconversion). <b>poisson</b> — a random spike train whose average <b>firing\nrate</b> is proportional to the value. <b>latency</b> — time-to-first-spike:\nlarger values fire earlier, small values late or never.\n<br><br>\nHorizontal axis = time step within the window; each row/track is a spike train,\neach mark a spike. <i>seed</i> fixes the randomness of the poisson encoding so the\npicture is reproducible.\n':
        '\n<b>O que este painel mostra.</b> As três <b>codificações</b> de disparos lado a\nlado para uma janela, para você ver como cada uma converte um número real em\neventos de disparo.\n<br><br>\n<b>direto</b> — a amostra normalizada passa como está (nada é convertido em tempo\nde disparo). <b>poisson</b> — um trem de disparos aleatório cuja <b>taxa de\ndisparo</b> média é proporcional ao valor. <b>latência</b> — o tempo até o\nprimeiro disparo: valores maiores disparam mais cedo, valores pequenos disparam\ntarde ou nunca.\n<br><br>\nO eixo horizontal é o passo de tempo dentro da janela; cada linha é um trem de\ndisparos e cada marca, um disparo. O <i>seed</i> fixa a aleatoriedade da\ncodificação poisson para que a figura seja reproduzível.\n',
    '\n<b>What this shows.</b> How a meeting01 window becomes spikes.\nSNN = Spiking Neural Network; its neurons fire discrete 0/1 <b>spikes</b> instead\nof sending continuous numbers.\n<br><br>\n<b>Panel 1</b> — the normalised (z-scored) window.\n<b>Panel 2</b> — the input <b>spike train</b> for the chosen encoding\n(<i>direct</i> = pass-through, <i>poisson</i> = firing rate ∝ value,\n<i>latency</i> = bigger value fires earlier). Each mark is one spike; click it\nfor its exact time.\n<b>Panel 3</b> — the <b>membrane voltage</b> v[t] of the recurrent\nLeaky-Integrate-and-Fire (LIF) transform, sweeping the 256 window samples as time\nsteps: <code>v[t] = α·v[t−1] + x[t] − s[t−1]·v_th</code>. The dashed red line is\nthe threshold <b>v_th</b>; a green mark sits on every step where v crossed it and\nthe neuron fired.\n<b>Panel 4</b> (only with a trained model) — one membrane value per encoder LIF\nneuron, a snapshot because that network runs with a single time step.\n<br><br>\n<b>Sliders.</b> <b>α</b> (alpha) = leak, 0…1: near 1 the neuron remembers input\nlonger. <b>v_th</b> = threshold: higher → fewer spikes.\n':
        '\n<b>O que este painel mostra.</b> Como uma janela do meeting01 vira disparos.\nSNN = Rede Neural de Disparos; seus neurônios emitem <b>disparos</b> discretos de\n0/1 em vez de enviar números contínuos.\n<br><br>\n<b>Painel 1</b> — a janela normalizada (em z-score).\n<b>Painel 2</b> — o <b>trem de disparos</b> de entrada para a codificação\nescolhida (<i>direto</i> = passa direto, <i>poisson</i> = taxa de disparo\nproporcional ao valor, <i>latência</i> = valor maior dispara mais cedo). Cada\nmarca é um disparo; clique nela para ver o instante exato.\n<b>Painel 3</b> — a <b>voltagem de membrana</b> v[t] da transformação recorrente\nde integração-e-disparo com vazamento (LIF), varrendo as 256 amostras da janela\ncomo passos de tempo: <code>v[t] = α·v[t−1] + x[t] − s[t−1]·v_th</code>. A linha\nvermelha tracejada é o limiar <b>v_th</b>; há uma marca verde em cada passo em que\nv o cruzou e o neurônio disparou.\n<b>Painel 4</b> (só com um modelo treinado) — um valor de membrana por neurônio\nLIF do codificador; é um instantâneo, porque essa rede roda com um único passo de\ntempo.\n<br><br>\n<b>Controles deslizantes.</b> <b>α</b> (alfa) = o vazamento, de 0 a 1: perto de 1\no neurônio guarda a entrada por mais tempo. <b>v_th</b> = o limiar: quanto mais\nalto, menos disparos.\n',
    "\n<b>What this shows.</b> The trained meeting01 SNN autoencoder <i>encoder</i> as\ncolumns of neurons: input → Linear(64) → LIF spikes(64) → latent(32).\n<br><br>\n<b>Node size &amp; colour</b> = that neuron's activity for the selected window\n(|activation|, or spike count for the LIF column; colour bar on the left).\n<b>Edges</b> = the connecting Linear layer's weights, drawn only for the top few\n|weight| per target neuron so the picture stays readable.\n<br><br>\n<b>Controls.</b> top-K edges per neuron, |weight| threshold, activity threshold.\nPress ▶ on the transport bar to <b>flood the signal through the layers</b> one\ncolumn at a time. LIF = Leaky Integrate-and-Fire spiking neuron.\n<br><br>\n<b>Axes.</b> The bounding box you see is real: X is the layer index. Y is each\nneuron's slot within its layer — a layout order for spreading neurons apart,\nnot a measurement, which is why its title says so instead of inventing a unit.\n":
        "\n<b>O que este painel mostra.</b> O <i>codificador</i> do autoencoder SNN treinado\ndo meeting01, como colunas de neurônios: entrada → Linear(64) → disparos LIF(64)\n→ latente(32).\n<br><br>\n<b>O tamanho e a cor de cada nó</b> indicam a atividade daquele neurônio para a\njanela selecionada (|ativação|, ou a contagem de disparos na coluna LIF; barra de\ncores à esquerda). <b>As arestas</b> são os pesos da camada Linear que conecta,\ndesenhadas só para os poucos maiores |peso| de cada neurônio de destino, para o\ndesenho não virar bagunça.\n<br><br>\n<b>Controles.</b> número de arestas por neurônio, limiar de |peso| e limiar de\natividade. Aperte ▶ na barra de transporte para <b>fazer o sinal inundar as\ncamadas</b>, uma coluna de cada vez. LIF = neurônio de integração-e-disparo com\nvazamento.\n<br><br>\n<b>Eixos.</b> A caixa delimitadora que você vê é real: X é o índice da camada.\nY é o lugar de cada neurônio dentro da camada — uma ordem de layout para\nespalhar os neurônios, não uma medida, por isso o título diz isso em vez de\ninventar uma unidade.\n",
    "\n<b>What this shows.</b> Every window of one or more meeting01 <b>folds</b>\npushed through each fold's trained autoencoder, then the 32-number <b>latent</b>\nvectors squeezed together to 2-D (or 3-D) so you can see the structure.\n<br><br>\n<b>This is a PROJECTED view</b>, not a pipeline number.\n<b>PCA</b> (Principal Component Analysis) rotates onto the directions of greatest\nspread — distances stay roughly meaningful, and the title shows how much\nvariability the 2 axes keep. <b>t-SNE</b> only preserves <i>who is near whom</i>;\ngaps and cluster sizes are not meaningful.\n<br><br>\n<b>Colour</b> = digit spoken, speaker, or — once you check more than one fold in\nthe list — <b>fold</b>, so overlapping colours mean the folds agree and\nseparated clumps mean one fold's model landed somewhere different.\n<b>Click a point</b> to select that window everywhere else in the app.\n<br><br>\nCheck the folds you want in the list (Ctrl/Shift-click for several), then press\n<i>Project</i> and wait for the status line — the forward passes run on a\nbackground thread. A fold with no trained model is skipped and named in the\nstatus line.\n":
        "\n<b>O que este painel mostra.</b> Todas as janelas de uma ou mais <b>dobras</b>\ndo meeting01 passadas pelo autoencoder treinado de cada dobra; depois, os\n<b>vetores latentes</b> de 32 números são reduzidos juntos a 2-D (ou 3-D) para\nvocê enxergar a estrutura.\n<br><br>\n<b>Esta é uma visão PROJETADA</b>, não um número do pipeline.\nO <b>PCA</b> (Análise de Componentes Principais) gira os dados para as direções\nde maior dispersão — as distâncias continuam mais ou menos significativas, e o\ntítulo mostra quanta variabilidade os 2 eixos preservam. O <b>t-SNE</b> só\npreserva <i>quem está perto de quem</i>; os espaços e os tamanhos dos grupos não\ntêm significado.\n<br><br>\nA <b>cor</b> indica o dígito falado, o locutor, ou — quando você marca mais de\numa dobra na lista — a <b>dobra</b>: cores sobrepostas dizem que as dobras\nconcordam, e grupos separados dizem que o modelo de uma dobra enxergou o mundo\nde outro jeito. <b>Clique num ponto</b> para selecionar aquela janela em todo o\nresto do aplicativo.\n<br><br>\nMarque as dobras que você quer na lista (Ctrl/Shift-clique para várias), depois\naperte <i>Projetar</i> e aguarde a linha de status — as passagens pela rede\nrodam numa thread em segundo plano. Uma dobra sem modelo treinado é ignorada e\nnomeada na linha de status.\n",
    '\n<b>What this shows.</b> One audio <b>window</b> (a 256-sample slice) after it has\nbeen pushed through a trained <b>SNN autoencoder</b> (SNN = Spiking Neural\nNetwork; an autoencoder squeezes the window into a small <b>latent</b> vector of\n32 numbers and then rebuilds it).\n<br><br>\n<b>Top plot.</b> Blue = the <i>original</i> encoder input (the flattened,\nspike-encoded window). Orange = the <i>reconstruction</i> the decoder produced\nfrom the latent vector. The closer they sit, the more information the 32-number\nlatent kept.\n<br><b>Bottom plot.</b> The <i>residual</i> = original − reconstruction, point by\npoint. A flat line near zero means a faithful rebuild; spikes mark where the\nmodel lost detail.\n<br><br>\n<b>The numbers</b> (table below): every one has a "what it means" column.\nMSE / MAE measure the error size (0 = perfect); R² is the fraction of the\nsignal\'s variability captured (1 = perfect, 0 = no better than a flat line);\nPearson r is shape agreement ignoring scale (+1 = identical shape).\n<br><br>\n<b>Origin tag</b> <code>[computed]</code> means these curves were recomputed here\nthrough the exact experiment code, not read from a cached file.\n':
        '\n<b>O que este painel mostra.</b> Uma <b>janela</b> de áudio (um trecho de 256\namostras) depois de passar por um <b>autoencoder SNN</b> treinado (SNN = Rede\nNeural de Disparos; um autoencoder comprime a janela num pequeno <b>vetor\nlatente</b> de 32 números e depois a reconstrói).\n<br><br>\n<b>Gráfico de cima.</b> Em azul, a entrada <i>original</i> do codificador (a\njanela achatada e codificada em disparos). Em laranja, a <i>reconstrução</i> que\no decodificador produziu a partir do vetor latente. Quanto mais próximas as duas\ncurvas, mais informação os 32 números do latente preservaram.\n<br><b>Gráfico de baixo.</b> O <i>resíduo</i> = original − reconstrução, ponto a\nponto. Uma linha reta perto de zero indica uma reconstrução fiel; os picos marcam\nonde o modelo perdeu detalhe.\n<br><br>\n<b>Os números</b> (na tabela abaixo): cada um tem a coluna "o que significa".\nMSE e MAE medem o tamanho do erro (0 = perfeito); o R² é a fração da\nvariabilidade do sinal que foi capturada (1 = perfeito, 0 = nada melhor que uma\nlinha reta); o Pearson r mede a concordância de forma, ignorando a escala (+1 =\nforma idêntica).\n<br><br>\nA <b>etiqueta de origem</b> <code>[computed]</code> quer dizer que essas curvas\nforam recalculadas aqui pelo código exato do experimento, e não lidas de um\narquivo em cache.\n',
    '\n<b>What this shows.</b> Each feature set placed on the <b>paraconsistent</b>\nplane. Paraconsistent logic lets a claim be supported <i>and</i> denied at once,\nwhich is exactly what noisy biometric evidence looks like.\n<br><br>\n<b>Axes.</b> Horizontal <b>G1 = α − β</b> (certainty): +1 = evidence firmly says\n"same person", −1 = firmly "different", 0 = undecided. Vertical\n<b>G2 = α + β − 1</b> (contradiction): +1 = evidence fully conflicts with itself,\n−1 = evidence missing, 0 = clean. α (alpha) is evidence <i>for</i>, β (beta) is\nevidence <i>against</i>, both 0…1.\n<br><br>\nThe ideal corner is <b>(G1 = 1, G2 = 0)</b> — certainly true, no contradiction.\n<b>D_truth</b> is the straight-line distance to it; <b>D_penalized</b> adds\n(2 − √2)·|G2| for contradiction and is the number the ranking sorts on. Smaller\nis better. Hover a point for all six quantities.\n':
        '\n<b>O que este painel mostra.</b> Cada conjunto de atributos posicionado no plano\n<b>paraconsistente</b>. A lógica paraconsistente permite que uma afirmação seja\nao mesmo tempo sustentada <i>e</i> negada — que é exatamente a cara de uma\nevidência biométrica ruidosa.\n<br><br>\n<b>Eixos.</b> Na horizontal, <b>G1 = α − β</b> (a certeza): +1 = a evidência\nafirma com firmeza "mesma pessoa", −1 = afirma com firmeza "pessoa diferente", 0\n= indeciso. Na vertical, <b>G2 = α + β − 1</b> (a contradição): +1 = a evidência\nse contradiz por completo, −1 = evidência ausente, 0 = sem conflito. α (alfa) é a\nevidência <i>a favor</i>, β (beta) é a evidência <i>contra</i>, ambas entre 0 e 1.\n<br><br>\nO canto ideal é <b>(G1 = 1, G2 = 0)</b> — verdadeiro com certeza, sem\ncontradição. <b>D_truth</b> é a distância em linha reta até esse canto;\n<b>D_penalized</b> soma (2 − √2)·|G2| pela contradição, e é o número pelo qual o\nranking é ordenado. Menor é melhor. Passe o mouse sobre um ponto para ver as seis\nquantidades.\n',
    '\n<b>What this shows.</b> Every persisted score as a point at\n(<b>D_truth</b>, <b>D_penalized</b>). The dashed diagonal is D_penalized =\nD_truth; the vertical gap above it <i>is</i> the contradiction penalty\n(2 − √2)·|G2|, so points far above the line have self-conflicting evidence.\nPoint colour scales with that gap.\n<br><br>\nD_truth = distance on the (G1, G2) plane to the ideal "certainly true, no\ncontradiction" corner. D_penalized = D_truth + the penalty; it is what the\nranking sorts on. Both: smaller is better.\n<br><br>\nThe seven facet filters (dataset, modality, wavelet, scale, fold, seed,\nstrategy) narrow the cloud. Click a point to inspect it.\n':
        '\n<b>O que este painel mostra.</b> Cada nota já salva como um ponto em\n(<b>D_truth</b>, <b>D_penalized</b>). A diagonal tracejada é D_penalized =\nD_truth; a distância vertical acima dela <i>é</i> a penalidade de contradição\n(2 − √2)·|G2| — ou seja, pontos bem acima da linha têm uma evidência que se\ncontradiz. A cor do ponto acompanha essa distância.\n<br><br>\nD_truth = a distância, no plano (G1, G2), até o canto ideal "verdadeiro com\ncerteza, sem contradição". D_penalized = D_truth + a penalidade; é por ele que o\nranking ordena. Nos dois casos, menor é melhor.\n<br><br>\nOs sete filtros (conjunto de dados, modalidade, wavelet, escala, dobra, seed,\nestratégia) reduzem a nuvem de pontos. Clique num ponto para inspecioná-lo.\n',
    '\n<b>What this shows.</b> Every persisted <b>paraconsistent</b> score from every\nexperiment in one sortable table: experiment, run, feature set, modality /\nencoding, seed, and then α, β, G1, G2, D_truth, <b>D_penalized</b>.\n<br><br>\nDefault sort is D_penalized ascending — <b>smaller is better</b> (it is the\ndistance to "certainly true, no contradiction", with a contradiction penalty).\nMissing values show as "—" and sort last, never as 0. Type in the filter box to\nnarrow by any text. Double-click a row to jump to that experiment.\n':
        '\n<b>O que este painel mostra.</b> Todas as notas <b>paraconsistentes</b> já\nsalvas, de todos os experimentos, numa única tabela ordenável: experimento,\nexecução, conjunto de atributos, modalidade / codificação, seed e, em seguida, α,\nβ, G1, G2, D_truth e <b>D_penalized</b>.\n<br><br>\nA ordenação padrão é por D_penalized crescente — <b>menor é melhor</b> (é a\ndistância até "verdadeiro com certeza, sem contradição", com uma penalidade pela\ncontradição). Valores ausentes aparecem como "—" e vão para o fim, nunca como 0.\nDigite na caixa de filtro para restringir por qualquer texto. Dê um duplo clique\nnuma linha para pular para aquele experimento.\n',
    '\n<b>What this shows.</b> The meeting01 run as a tree: session → dataset / fold →\nconfig → epoch, built live from the structured event log.\n<br><br>\nSelecting a config plots its learning curve below: <b>train loss</b> and\n<b>validation loss</b> per epoch (loss = reconstruction error the optimiser\nminimises), a dashed line at the best epoch, epoch <b>duration</b> on the right\naxis, and the learning rate in the title. A rising validation curve while train\nkeeps falling is the classic overfitting shape — but the tool only shows it, it\ndoes not label it.\n<br><br>\n<b>Compare folds.</b> Ctrl/Shift-click several configs (or a whole fold row,\nwhich selects every config under it) to overlay their curves — one colour per\nconfig, <b>solid = validation loss</b>, <b>dotted = train loss</b> — so you can\nsee whether the network behaves the same way across folds or diverges on one.\n<br><br>\nEmpty until a LOSO run has written <code>results/meeting01/*_events.jsonl</code>.\n':
        '\n<b>O que este painel mostra.</b> A execução do meeting01 como uma árvore: sessão\n→ conjunto de dados / dobra → configuração → época, montada ao vivo a partir do\nlog estruturado de eventos.\n<br><br>\nAo selecionar uma configuração, a curva de aprendizado aparece abaixo: a\n<b>perda de treino</b> e a <b>perda de validação</b> por época (a perda é o erro\nde reconstrução que o otimizador minimiza), uma linha tracejada na melhor época,\na <b>duração</b> da época no eixo direito e a taxa de aprendizado no título. Uma\ncurva de validação subindo enquanto a de treino continua caindo é o formato\nclássico do sobreajuste — mas a ferramenta só mostra, ela não rotula.\n<br><br>\n<b>Compare dobras.</b> Ctrl/Shift-clique em várias configurações (ou numa linha\nde dobra inteira, o que seleciona todas as configurações dela) para sobrepor as\ncurvas — uma cor por configuração, <b>traço contínuo = perda de validação</b>,\n<b>pontilhado = perda de treino</b> — assim dá para ver se a rede se comporta do\nmesmo jeito em todas as dobras ou se uma delas foge do padrão.\n<br><br>\nFica vazio até uma execução LOSO gravar\n<code>results/meeting01/*_events.jsonl</code>.\n',
    '\n<b>What this shows.</b> The result of <b>NSGA-II</b> (Non-dominated Sorting\nGenetic Algorithm II), a multi-objective architecture search: it evolves a\npopulation and keeps the candidates that are not beaten on every objective at\nonce.\n<br><br>\n<b>Axes.</b> Vertical = D_penalized mean (paraconsistent quality, smaller\nbetter). Horizontal = the cost you pick: <b>inference cost</b> (spikes + 10 ×\nmultiply-accumulates, hardware-independent), parameter count, estimated latency,\nor latent activity.\n<br><br>\n<b>Green</b> = <b>feasible</b> Pareto-front points (satisfy every hard\nconstraint); <b>orange</b> = infeasible — shown for context, never winners.\nEstimated latency is labelled <b>UNCALIBRATED</b>: it is a rough model output,\nnot a measured millisecond figure — the view never ranks by it. Click a point\nfor its genome and fitness.\n':
        '\n<b>O que este painel mostra.</b> O resultado do <b>NSGA-II</b> (Non-dominated\nSorting Genetic Algorithm II), uma busca de arquitetura com vários objetivos:\nele faz evoluir uma população e mantém os candidatos que não são superados em\nnenhum objetivo ao mesmo tempo.\n<br><br>\n<b>Eixos.</b> Na vertical, a média de D_penalized (a qualidade paraconsistente;\nmenor é melhor). Na horizontal, o custo que você escolher: o <b>custo de\ninferência</b> (disparos + 10 × multiplicações-e-acumulações, independente de\nhardware), o número de parâmetros, a latência estimada ou a atividade do latente.\n<br><br>\nEm <b>verde</b>, os pontos <b>viáveis</b> da fronteira de Pareto (que respeitam\ntodas as restrições rígidas); em <b>laranja</b>, os inviáveis — mostrados só como\ncontexto, nunca como vencedores. A latência estimada leva o rótulo <b>NÃO\nCALIBRADA</b>: é uma saída aproximada do modelo, não um valor medido em\nmilissegundos — este painel nunca ordena por ela. Clique num ponto para ver o\ngenoma e a aptidão.\n',
    '\n<b>What this shows.</b> A structural, side-by-side comparison of the two\npipelines. It is <b>not</b> a claim that they are scientifically equivalent.\n<br><br>\nEach row is one aspect (dataset, windowing, encoding, wavelet, the autoencoder\nfamilies, cross-validation, …). The relation column is one of:\n<b>same concept</b>, <b>similar implementation</b>, <b>different\nimplementation</b>, <b>not applicable</b>, <b>unknown</b> — colour-coded, with the\nsource of the claim in the tooltip. The wording is deliberately conservative.\n':
        '\n<b>O que este painel mostra.</b> Uma comparação estrutural, lado a lado, dos dois\npipelines. <b>Não</b> é uma afirmação de que eles sejam cientificamente\nequivalentes.\n<br><br>\nCada linha é um aspecto (conjunto de dados, janelamento, codificação, wavelet, as\nfamílias de autoencoder, validação cruzada, …). A coluna de relação assume um\ndestes valores: <b>mesmo conceito</b>, <b>implementação parecida</b>,\n<b>implementação diferente</b>, <b>não se aplica</b> ou <b>desconhecido</b> — com\ncódigo de cor, e a origem da afirmação na dica ao passar o mouse. A redação é\npropositalmente cautelosa.\n',
    '\n<b>What this shows.</b> One thesis sample seen through three synchronised lenses\nat once — scrub the sample index and all three panels move together.\n<br><br>\n<b>Wavelet</b> (left) — the per-band <b>energy</b> of that sample\'s signal.\n<b>Features</b> (middle) — that sample\'s handcrafted feature vector as a bar\nchart. <b>Paraconsistent</b> (right) — the run\'s feature set on the G1×G2 plane\n(G1 = certainty α−β, G2 = contradiction α+β−1).\n<br><br>\nWhen the feature layout lines up 1:1 with the wavelet bands, clicking a feature\nbar highlights the band it came from, and vice versa — the "why did this number\nbecome this number" link.\n':
        '\n<b>O que este painel mostra.</b> Uma amostra da tese vista por três lentes\nsincronizadas ao mesmo tempo — arraste o índice da amostra e os três painéis\nandam juntos.\n<br><br>\nÀ esquerda, <b>Wavelet</b> — a <b>energia</b> por faixa do sinal daquela amostra.\nNo meio, <b>Atributos</b> — o vetor de atributos manuais daquela amostra em forma\nde barras. À direita, <b>Paraconsistente</b> — o conjunto de atributos da\nexecução no plano G1×G2 (G1 = certeza α−β, G2 = contradição α+β−1).\n<br><br>\nQuando o arranjo dos atributos bate 1 para 1 com as faixas da wavelet, clicar\nnuma barra de atributo destaca a faixa de onde ela veio, e vice-versa — é o elo\ndo "por que este número virou este número".\n',

    # ---- glossário: siglas por extenso ----------------------
    "Leave-One-Speaker-Out": "Deixa-Um-Locutor-de-Fora",
    "Autoencoder": "autoencoder (autocodificador)",
    "Spiking Neural Network": "rede neural de disparos",
    "Long Short-Term Memory": "memória de longo e curto prazo",
    "Gated Recurrent Unit": "unidade recorrente com portas",
    "Principal Component Analysis": "análise de componentes principais",
    "t-distributed Stochastic Neighbour Embedding":
        "imersão estocástica de vizinhos com distribuição t",
    "voltage threshold": "limiar de voltagem",
    "Leaky Integrate-and-Fire": "integração-e-disparo com vazamento",
    "membrane voltage": "voltagem de membrana",
    "Discrete Tunable Wavelet Packet Transform":
        "transformada discreta ajustável em pacote de wavelet",
    "Zero-Crossing Rate": "taxa de cruzamentos por zero",
    "Teager–Kaiser energy operator": "operador de energia de Teager–Kaiser",
    "Linear-Frequency Cepstral Coefficients":
        "coeficientes cepstrais de frequência linear",
    "evidence FOR (α)": "evidência A FAVOR (α)",
    "evidence AGAINST (β)": "evidência CONTRA (β)",
    "certainty degree (G1 = α − β)": "grau de certeza (G1 = α − β)",
    "contradiction degree (G2 = α + β − 1)": "grau de contradição (G2 = α + β − 1)",
    "distance to truth": "distância até a verdade",
    "penalised distance": "distância com penalidade",
    "Mean Squared Error": "erro quadrático médio",
    "Mean Absolute Error": "erro absoluto médio",
    "R² (coefficient of determination)": "R² (coeficiente de determinação)",
    "Pearson correlation coefficient": "coeficiente de correlação de Pearson",
    "Equal Error Rate": "taxa de erro igual",
    "Area Under the ROC Curve": "área sob a curva ROC",
    "Non-dominated Sorting Genetic Algorithm II":
        "algoritmo genético de ordenação por não-dominância II",

    # ---- glossário: significados ----------------------------
    "The conference-paper pipeline: it trains autoencoders (SNN vs LSTM/GRU/Transformer) to reconstruct short audio windows and compares them speaker-by-speaker.":
        "O pipeline do artigo de congresso: treina autoencoders (SNN, LSTM, GRU e Transformer) para reconstruir janelas curtas de áudio e depois os compara locutor por locutor.",
    "The doctoral pipeline: handcrafted wavelet features on EEG/voice, scored by paraconsistent logic, then used for person authentication.":
        "O pipeline do doutorado: atributos de wavelet feitos à mão, extraídos de EEG ou voz, avaliados pela lógica paraconsistente e então usados para autenticar pessoas.",
    "A way to test fairly: every speaker is held out once as the test set while the model trains on all the others, so the score is 'how well does it work on a person it never saw'.":
        "Uma forma justa de avaliar: cada locutor é deixado de fora uma vez, como conjunto de teste, enquanto o modelo treina com todos os outros. Assim a nota responde 'quão bem ele se sai com uma pessoa que nunca viu'.",
    "One split of the data into train / validation / test. Results are averaged over all folds.":
        "Uma divisão dos dados em treino / validação / teste. Os resultados são a média de todas as dobras.",
    "The train/test split. 'Outer' because a second, inner split (train/validation) sits inside it for model selection.":
        "A divisão entre treino e teste. É a 'externa' porque existe uma segunda divisão, interna (treino / validação), que serve para escolher o modelo.",
    "A short fixed-length slice of the raw signal (here 256 samples). The models work on windows, not whole recordings.":
        "Um trecho curto e de tamanho fixo do sinal bruto (aqui, 256 amostras). Os modelos trabalham com janelas, não com gravações inteiras.",
    "A network that compresses its input to a small vector (the latent) and then rebuilds it; good reconstruction means the latent kept the important information.":
        "Uma rede que comprime a entrada num vetor pequeno (o latente) e depois a reconstrói; uma boa reconstrução indica que o latente guardou a informação importante.",
    "A network whose neurons communicate with discrete 0/1 'spikes' over time instead of continuous numbers — closer to biology and cheaper on neuromorphic hardware.":
        "Uma rede cujos neurônios se comunicam por 'disparos' discretos de 0/1 ao longo do tempo, em vez de números contínuos — mais perto da biologia e mais barata em hardware neuromórfico.",
    "A classic recurrent network for sequences.":
        "Uma rede recorrente clássica para sequências.",
    "A lighter recurrent network, similar to LSTM.":
        "Uma rede recorrente mais leve, parecida com a LSTM.",
    "An attention-based sequence model (the architecture behind modern language models).":
        "Um modelo de sequência baseado em atenção (a arquitetura por trás dos modelos de linguagem atuais).",
    "A rotation of the data onto the directions of greatest spread; the first few directions give a faithful low-dimensional picture. A projection, not a measured value.":
        "Um giro dos dados na direção de maior dispersão; as primeiras direções já dão uma imagem fiel em poucas dimensões. É uma projeção, não um valor medido.",
    "A non-linear 2-D layout that keeps near points near; good for spotting clusters, but distances and densities between clusters are not meaningful.":
        "Um arranjo não linear em 2-D que mantém próximos os pontos que já eram próximos; bom para enxergar agrupamentos, mas as distâncias e as densidades entre agrupamentos não querem dizer nada.",
    "The small vector an autoencoder's encoder produces (here 32 numbers). It is the model's compressed description of the window.":
        "O vetor pequeno que o codificador de um autoencoder produz (aqui, 32 números). É a descrição comprimida da janela feita pelo modelo.",
    "The decoder's attempt to rebuild the original input from the latent vector.":
        "A tentativa do decodificador de refazer a entrada original a partir do vetor latente.",
    "Original minus reconstruction, point by point. Flat and near zero means a faithful rebuild.":
        "O original menos a reconstrução, ponto a ponto. Reto e perto de zero significa uma reconstrução fiel.",
    "Spike encoding that passes the normalised sample straight through (no conversion to spike times).":
        "Codificação de disparos que deixa a amostra normalizada passar como está (sem converter em tempos de disparo).",
    "Rate encoding: each value becomes a random spike train whose average firing rate is proportional to the value.":
        "Codificação por taxa: cada valor vira um trem de disparos aleatório cuja taxa média de disparo é proporcional ao valor.",
    "Time-to-first-spike encoding: larger values spike earlier.":
        "Codificação pelo tempo até o primeiro disparo: valores maiores disparam mais cedo.",
    "The membrane voltage a LIF neuron must reach to fire a spike. Higher threshold → fewer spikes.":
        "A voltagem de membrana que um neurônio LIF precisa atingir para disparar. Limiar mais alto, menos disparos.",
    "The membrane leak factor (0–1). Closer to 1 = the neuron remembers past input longer; closer to 0 = it forgets quickly.":
        "O fator de vazamento da membrana (de 0 a 1). Perto de 1, o neurônio guarda a entrada por mais tempo; perto de 0, esquece rápido.",
    "The standard spiking-neuron model: it adds up incoming current, leaks some away each step, and fires when it crosses the threshold, then resets.":
        "O modelo padrão de neurônio de disparo: soma a corrente que chega, deixa um pouco escorrer a cada passo e dispara quando cruza o limiar, voltando então ao repouso.",
    "The running internal voltage of a spiking neuron. It rises with input, leaks between inputs, and resets after a spike.":
        "A voltagem interna de um neurônio de disparo, momento a momento. Sobe com a entrada, escorre entre uma entrada e outra, e volta ao repouso depois de um disparo.",
    "Same as membrane potential — the neuron's internal voltage.":
        "O mesmo que potencial de membrana — a voltagem interna do neurônio.",
    "How many discrete time steps one sample is unrolled over. meeting01's autoencoder uses 1 (the window is fed as a feature vector); the recurrent transform uses 256 (the window samples).":
        "Em quantos passos de tempo uma amostra é desenrolada. O autoencoder do meeting01 usa 1 (a janela entra como um vetor de atributos); a transformação recorrente usa 256 (as amostras da janela).",
    "A dot plot: one row per neuron, a mark at every time it fired.":
        "Um gráfico de pontos: uma linha por neurônio, uma marca em cada instante em que ele disparou.",
    "Fraction of time steps on which a neuron spiked (0–1).":
        "A fração dos passos de tempo em que um neurônio disparou (de 0 a 1).",
    "A short oscillation used to split a signal into frequency bands at several scales at once (unlike a plain spectrum, it keeps time information).":
        "Uma oscilação curta usada para separar um sinal em faixas de frequência, em várias escalas ao mesmo tempo (diferente de um espectro comum, ela preserva a informação de tempo).",
    "A full binary tree of wavelet filters: every band is split again, giving equal-width sub-bands (leaves).":
        "Uma árvore binária completa de filtros de wavelet: cada faixa é dividida de novo, gerando sub-faixas de largura igual (as folhas).",
    "The wavelet-packet decomposition used by the thesis feature extractor.":
        "A decomposição em pacote de wavelet usada pelo extrator de atributos da tese.",
    "One sub-band at the bottom of the wavelet-packet tree — a narrow frequency range.":
        "Uma sub-faixa na base da árvore do pacote de wavelet — um intervalo estreito de frequência.",
    "A limited range of frequencies produced by the wavelet decomposition.":
        "Um intervalo limitado de frequências, produzido pela decomposição em wavelet.",
    "The simplest wavelet (a single square step). Fast, blocky.":
        "A wavelet mais simples (um único degrau quadrado). Rápida e de contornos duros.",
    "A family of smooth wavelets; 'daub10' means 10 filter taps — higher number = smoother, wider support.":
        "Uma família de wavelets suaves; 'daub10' quer dizer 10 coeficientes de filtro — quanto maior o número, mais suave e mais largo o suporte.",
    "Sum of squared coefficients in a band — how much of the signal's power sits in that frequency range.":
        "A soma dos coeficientes ao quadrado numa faixa — o quanto da potência do sinal está naquele intervalo de frequência.",
    "A band's energy divided by the total, so the bands sum to 1.":
        "A energia de uma faixa dividida pelo total, de modo que as faixas somem 1.",
    "How often the signal changes sign — a rough pitch / noisiness measure.":
        "Com que frequência o sinal troca de sinal — uma medida grosseira de altura tonal e de ruído.",
    "How evenly spread the coefficients are. High = noise-like, low = a few dominant components.":
        "O quanto os coeficientes estão espalhados de forma uniforme. Alta = parece ruído; baixa = poucos componentes dominam.",
    "An instantaneous energy estimate sensitive to both amplitude and frequency.":
        "Uma estimativa de energia instantânea, sensível tanto à amplitude quanto à frequência.",
    "Cycle-to-cycle variation in period — a voice-quality measure.":
        "A variação de um ciclo para o outro no período — uma medida de qualidade da voz.",
    "Cycle-to-cycle variation in amplitude — a voice-quality measure.":
        "A variação de um ciclo para o outro na amplitude — uma medida de qualidade da voz.",
    "A compact spectral-shape descriptor on a linear frequency axis (the linear-axis cousin of MFCCs).":
        "Um descritor compacto do formato do espectro num eixo de frequência linear (o primo, de eixo linear, dos MFCCs).",
    "Computed from the log spectrum via another transform; captures the overall spectral envelope.":
        "Calculado a partir do espectro em escala logarítmica, por meio de outra transformada; captura o contorno geral do espectro.",
    "A logic that tolerates contradiction: a statement can be supported and denied at the same time. Used here to score how cleanly a feature set separates people.":
        "Uma lógica que tolera contradição: uma afirmação pode ser sustentada e negada ao mesmo tempo. Aqui, serve para avaliar com que nitidez um conjunto de atributos separa as pessoas.",
    "Degree to which the feature set supports 'same person' (0–1).":
        "O quanto o conjunto de atributos sustenta a hipótese de 'mesma pessoa' (de 0 a 1).",
    "Degree to which it supports 'different person' (0–1).":
        "O quanto ele sustenta a hipótese de 'pessoa diferente' (de 0 a 1).",
    "How decisive the evidence is. +1 = fully certain true, −1 = fully certain false, 0 = undecided.":
        "O quanto a evidência é decisiva. +1 = verdadeiro com certeza total, −1 = falso com certeza total, 0 = indeciso.",
    "How much the evidence conflicts with itself. +1 = fully contradictory, −1 = fully missing, 0 = consistent.":
        "O quanto a evidência se contradiz. +1 = totalmente contraditória, −1 = totalmente ausente, 0 = consistente.",
    "Straight-line distance on the (G1, G2) plane from the point to the ideal 'certainly true, no contradiction' corner (1, 0). Smaller is better.":
        "A distância em linha reta, no plano (G1, G2), do ponto até o canto ideal 'verdadeiro com certeza, sem contradição' (1, 0). Menor é melhor.",
    "D_truth plus a penalty of (2 − √2) × |G2| for contradiction. This is the number the feature ranking sorts on — smaller is better.":
        "O D_truth mais uma penalidade de (2 − √2) × |G2| pela contradição. É o número pelo qual o ranking de atributos é ordenado — menor é melhor.",
    "The constant (2 − √2) ≈ 0.5858 that turns contradiction |G2| into extra distance in D_penalized.":
        "A constante (2 − √2) ≈ 0,5858 que converte a contradição |G2| em distância extra dentro do D_penalized.",
    "Average of (original − reconstruction)². 0 = perfect; grows fast with large errors.":
        "A média de (original − reconstrução)². 0 = perfeito; cresce rápido quando há erros grandes.",
    "Average of |original − reconstruction|. 0 = perfect; in the signal's own units.":
        "A média de |original − reconstrução|. 0 = perfeito; nas unidades do próprio sinal.",
    "Fraction of the signal's variance the reconstruction captured. 1 = perfect, 0 = no better than a flat line, negative = worse than flat.":
        "A fração da variância do sinal que a reconstrução capturou. 1 = perfeito, 0 = nada melhor que uma linha reta, negativo = pior que uma linha reta.",
    "How well the two curves rise and fall together, ignoring scale. +1 = identical shape, 0 = unrelated.":
        "O quanto as duas curvas sobem e descem juntas, ignorando a escala. +1 = forma idêntica, 0 = sem relação.",
    "How well original and reconstruction move together (−1…+1); +1 is identical shape.":
        "O quanto o original e a reconstrução andam juntos (de −1 a +1); +1 é forma idêntica.",
    "The operating point where the chance of wrongly accepting an impostor equals the chance of wrongly rejecting the genuine user. Lower is better.":
        "O ponto de operação em que a chance de aceitar um impostor por engano se iguala à chance de rejeitar o usuário legítimo por engano. Menor é melhor.",
    "Probability that a genuine trial scores above an impostor trial. 1 = perfect, 0.5 = chance.":
        "A probabilidade de uma tentativa legítima pontuar acima de uma tentativa impostora. 1 = perfeito, 0,5 = puro acaso.",
    "The share of total spread a PCA component accounts for; the first two components' shares tell you how much the 2-D picture leaves out.":
        "A parcela da dispersão total que um componente do PCA explica; a soma dos dois primeiros componentes diz o quanto a imagem 2-D deixa de fora.",
    "A multi-objective search: it evolves a population of architectures and keeps the ones that are not beaten on every objective at once.":
        "Uma busca com vários objetivos: faz evoluir uma população de arquiteturas e mantém as que não são superadas em nenhum objetivo ao mesmo tempo.",
    "The set of solutions where you cannot improve one objective without worsening another — the best available trade-offs.":
        "O conjunto de soluções em que não dá para melhorar um objetivo sem piorar outro — os melhores compromissos disponíveis.",
    "The encoded description of one candidate architecture the search evolves.":
        "A descrição codificada de uma arquitetura candidata, aquilo que a busca faz evoluir.",
    "A candidate that satisfies every hard constraint (e.g. a latency budget); infeasible ones are shown but never win.":
        "Um candidato que respeita todas as restrições rígidas (por exemplo, um limite de latência); os inviáveis aparecem, mas nunca vencem.",
    "A hardware-independent proxy for how expensive one forward pass is: spike count plus 10 × multiply-accumulates.":
        "Um indicador de custo de uma passagem pela rede, independente de hardware: a contagem de disparos mais 10 × as multiplicações-e-acumulações.",
    "The estimated latency is a rough model output, not measured on real hardware — do not rank or quote it as a real millisecond figure.":
        "A latência estimada é uma saída aproximada do modelo, não medida em hardware real — não a use para ordenar nem a cite como um valor de verdade em milissegundos.",
    "A value read straight from a saved experiment artifact.":
        "Um valor lido diretamente de um arquivo de resultado já salvo do experimento.",
    "A value recomputed here through the exact experiment C++ code — deterministic, matches the experiment bit-for-bit.":
        "Um valor recalculado aqui pelo código C++ exato do experimento — determinístico, idêntico ao experimento bit a bit.",
    "A lossy 2-D/3-D view (PCA, t-SNE) of higher-dimensional data — useful for the eye, not a pipeline number.":
        "Uma visão com perda de informação, em 2-D ou 3-D (PCA, t-SNE), de dados de dimensão maior — serve para o olho, não é um número do pipeline.",
    "A value flagged as an approximation (e.g. LIF parameters from a checkpoint that predates a fix).":
        "Um valor marcado como aproximação (por exemplo, parâmetros LIF de um checkpoint anterior a uma correção).",
    "The curve you see is decimated for speed; the cursor still reads the full-resolution array.":
        "A curva que você vê foi reduzida para ganhar velocidade; o cursor continua lendo o vetor em resolução total.",
    "No value is available. Shown as '—', never as 0.":
        "Não há valor disponível. Aparece como '—', nunca como 0.",
    "The signal after subtracting its mean and dividing by its standard deviation, so it is centred on 0 with spread 1 (the models see it this way).":
        "O sinal depois de subtrair a média e dividir pelo desvio padrão, ficando centrado em 0 com dispersão 1 (é assim que os modelos o enxergam).",
    "The random-number-generator starting value. Fixing it makes a run reproducible.":
        "O valor inicial do gerador de números aleatórios. Fixá-lo torna a execução reproduzível.",

    # ---- controles compartilhados dos painéis --------------
    "seed": "seed", "show": "exibir", "encoding": "codificação",
    "wavelet": "wavelet", "level": "nível", "mode": "modo", "channel": "canal",
    "feature": "atributo", "sample": "amostra", "mean": "média", "std": "desv. padrão",
    "min / max": "mín / máx", "time step": "passo de tempo", "amplitude": "amplitude",
    "compare all": "comparar todas",
    "Select a meeting01 window.": "Selecione uma janela do meeting01.",
    "Select a Phase-00 handcrafted run.":
        "Selecione uma execução manual da Fase 00.",

    # ---- Laboratório de Codificação ------------------------
    "Encoding Lab needs a meeting01 window.":
        "O Laboratório de Codificação precisa de uma janela do meeting01.",
    "time step (sample within the window)": "passo de tempo (amostra dentro da janela)",
    "amplitude / spike": "amplitude / disparo",
    "   [{enc} encoding]": "   [codificação {enc}]",
    "encode failed: {exc}": "falha ao codificar: {exc}",
    "{label}  [computed] — seed {seed}": "{label}  [calculado] — seed {seed}",

    # ---- Laboratório de Wavelet ---------------------------
    "No signal selected.": "Nenhum sinal selecionado.",
    "Leaf": "Folha", "Energy": "Energia", "Rel. energy": "Energia rel.",
    "coefficient index (sample within the band)":
        "índice do coeficiente (amostra dentro da faixa)",
    "coefficient value": "valor do coeficiente",
    "Selected object has no 1-D signal to decompose.":
        "O objeto selecionado não tem um sinal 1-D para decompor.",
    "   ·   {mode} {wavelet}, {n} bands, {ns} samples in  ·  [computed]":
        "   ·   {mode} {wavelet}, {n} faixas, {ns} amostras na entrada  ·  [calculado]",
    "Frequency band {idx} on its own — {n} numbers (band 0 = lowest pitch)":
        "A faixa de frequência {idx} isolada — {n} números (faixa 0 = som mais grave)",
    "Decomposition levels (coarsest → finest)":
        "Níveis de decomposição (do mais grosso ao mais fino)",
    "decomposition levels failed: {err}": "falha nos níveis de decomposição: {err}",
    "{label} — {n} coefficients [computed]": "{label} — {n} coeficientes [calculado]",

    # ---- painel do Sinal ---------------------------------
    "Play this waveform through the default audio output. The \N{BLACK RIGHT-POINTING TRIANGLE} transport below only steps animation frames — it is not sound.":
        "Toca esta forma de onda pela saída de áudio padrão. O transporte \N{BLACK RIGHT-POINTING TRIANGLE} abaixo só avança quadros de animação — ele não é som.",
    "This object has no raw signal.": "Este objeto não tem sinal bruto.",
    "load_signal failed: {exc}": "load_signal falhou: {exc}",
    "  · amplitude peak-normalised for listening":
        "  · amplitude normalizada pelo pico, só para a escuta",
    "playing {ms} ms at {hz} Hz": "tocando {ms} ms a {hz} Hz",
    "not audio (multi-channel or sub-3kHz) — nothing to play":
        "não é áudio (multicanal ou abaixo de 3 kHz) — nada a tocar",
    "QtMultimedia not installed — see Listen tooltip":
        "QtMultimedia não está instalado — veja a dica do botão Ouvir",
    "  · DISPLAY-DOWNSAMPLED 1:{stride} ({n}→{m} pts; cursor reads full-res)":
        "  · EXIBIÇÃO REDUZIDA 1:{stride} ({n}→{m} pts; o cursor lê em resolução total)",

    # ---- Laboratório de SNN -----------------------------
    "click a spike marker for its exact time / membrane / threshold":
        "clique num marcador de disparo para ver o instante, a membrana e o limiar exatos",
    "SNN Lab needs a meeting01 window.":
        "O Laboratório de SNN precisa de uma janela do meeting01.",
    "transform failed: {exc}": "a transformação falhou: {exc}",
    "1 · the window going in": "1 · a janela que entra",
    "2 · spikes going in — {n} of them (click one to inspect)":
        "2 · disparos na entrada — {n} deles (clique num para inspecionar)",
    "  (leak α={a}, firing line v_th={v})":
        "  (vazamento α={a}, linha de disparo v_th={v})",
    "charge inside the neuron  (membrane potential v[t])":
        "carga dentro do neurônio  (potencial de membrana v[t])",
    "4 · the trained network's 64 encoder neurons — {n} of them fired for this window (one charge value each, not a trajectory)":
        "4 · os 64 neurônios codificadores da rede treinada — {n} deles dispararam para esta janela (um valor de carga por neurônio, não uma trajetória)",
    "encoder neuron index": "índice do neurônio codificador",
    "charge at readout": "carga na leitura",
    "; trained encoder LIF membrane shown ({n} neurons)":
        "; membrana LIF do codificador treinado exibida ({n} neurônios)",
    "; no trained .npz for this fold — membrane panel hidden":
        "; nenhum .npz treinado para esta dobra — painel da membrana oculto",
    "{label}  [computed] — in {i} → out {o} spikes ({k} coincident, {a} LIF-added, {s} LIF-suppressed)":
        "{label}  [calculado] — {i} na entrada → {o} na saída ({k} coincidem, {a} acrescentados pelo LIF, {s} suprimidos pelo LIF)",
    "input encoding spike — t = sample {t}  ·  encoding = {enc}  ·  seed {seed}  (no membrane at the encoder input; it is a fixed spike train)":
        "disparo da codificação de entrada — t = amostra {t}  ·  codificação = {enc}  ·  seed {seed}  (não há membrana na entrada do codificador; é um trem de disparos fixo)",
    "recurrent-LIF spike — t = sample {t}  ·  v[t] = {v}  ·  v_th = {vth}  ·  crossed by {d}  ·  layer = recurrent transform":
        "disparo LIF recorrente — t = amostra {t}  ·  v[t] = {v}  ·  v_th = {vth}  ·  cruzou por {d}  ·  camada = transformação recorrente",

    # ---- barra Siga os Dados ----------------------------
    "RAW": "BRUTO", "WINDOW": "JANELA", "NORMALIZED": "NORMALIZADO",
    "ENCODING": "CODIFICAÇÃO", "WAVELET": "WAVELET", "FEATURES": "ATRIBUTOS",
    "PARACONSISTENT": "PARACONSISTENTE", "LATENT": "LATENTE",
    "RECONSTRUCTION": "RECONSTRUÇÃO", "CLASSIFICATION": "CLASSIFICAÇÃO",
    "not applicable to this selection": "não se aplica a esta seleção",
    "no selection": "nada selecionado",
    "this sample is not windowed here": "esta amostra não é dividida em janelas aqui",
    "spike encoding is a meeting01 window step":
        "a codificação de disparos é uma etapa das janelas do meeting01",
    "select an individual window / sample":
        "selecione uma janela ou amostra individual",
    "select the run node to recompute its feature matrix":
        "selecione o nó da execução para recalcular a matriz de atributos dela",
    "handcrafted-feature matrix is a thesis-run view":
        "a matriz de atributos manuais é um painel de execução da tese",
    "needs a trained model .npz (not yet available)":
        "precisa de um modelo treinado .npz (ainda não disponível)",

    # ---- Autoencoder (grafo encoder + decoder) ---------
    "Autoencoder graph": "Grafo do autoencoder",
    "\n"
    "<b>What this panel shows.</b> The whole trained autoencoder — the half that\n"
    "<i>compresses</i> (encoder) and the half that <i>rebuilds</i> (decoder) — as one\n"
    "left-to-right drawing: input (256) → 64 → <b>latent code (32)</b> → 64 →\n"
    "reconstruction (256).\n"
    "<br><br>\n"
    "<b>Every dot is a neuron.</b> Colour and size show how strongly it lit up for the\n"
    "chosen window. <b>The lines are the weights</b>: blue adds, red subtracts; only the\n"
    "strongest few per neuron are drawn or it becomes a smear.\n"
    "<br><br>\n"
    "<b>Press ▶</b> on the transport bar to watch the signal <b>travel through</b> the\n"
    "network one column at a time, in to the latent code and back out.\n"
    "<br><br>\n"
    "<b>Zoom in</b> (mouse wheel) on a spiking column, or tick <i>Neuron detail</i>:\n"
    "each LIF neuron then shows its built-up charge, its firing line and whether it\n"
    "fired. Click a neuron for its exact numbers. LIF = leaky integrate-and-fire neuron.\n"
    "<br><br>\n"
    "<b>Model.</b> The dropdown lists every trained model for this window's fold; it\n"
    "starts on the auto-picked winner. Choose another and press <i>Run this model</i>\n"
    "to see that exact checkpoint's structure and behaviour instead — the \"Model\n"
    "Structure\" tab updates to match whichever model is currently loaded here.\n":
        "\n"
        "<b>O que este painel mostra.</b> O autoencoder inteiro treinado — a metade que\n"
        "<i>comprime</i> (encoder) e a que <i>reconstrói</i> (decoder) — num só desenho,\n"
        "da esquerda para a direita: entrada (256) → 64 → <b>código latente (32)</b> → 64 →\n"
        "reconstrução (256).\n"
        "<br><br>\n"
        "<b>Cada bolinha é um neurônio.</b> A cor e o tamanho mostram o quanto ele se\n"
        "acendeu para a janela escolhida. <b>As linhas são os pesos</b>: azul soma, vermelho\n"
        "subtrai; só as mais fortes de cada neurônio aparecem, senão vira um borrão.\n"
        "<br><br>\n"
        "<b>Aperte ▶</b> na barra de transporte para ver o sinal <b>atravessar</b> a rede,\n"
        "uma coluna de cada vez, até o código latente e de volta.\n"
        "<br><br>\n"
        "<b>Dê zoom</b> (roda do mouse) numa coluna de disparos, ou marque\n"
        "<i>Detalhe do neurônio</i>: cada neurônio LIF passa a mostrar a carga acumulada,\n"
        "a linha de disparo e se disparou. Clique num neurônio para ver os números exatos.\n"
        "LIF = neurônio integra-e-dispara com vazamento.\n"
        "<br><br>\n"
        "<b>Modelo.</b> O menu lista todo modelo treinado para a dobra desta janela;\n"
        "começa no vencedor escolhido automaticamente. Escolha outro e aperte <i>Rodar\n"
        "este modelo</i> para ver a estrutura e o comportamento exatos daquele\n"
        "checkpoint — a aba \"Estrutura do modelo\" acompanha o que estiver carregado aqui.\n",
    "Neuron detail": "Detalhe do neurônio",
    "Select a meeting01 window with a trained model.":
        "Selecione uma janela do meeting01 que tenha modelo treinado.",
    "The autoencoder graph needs an individual meeting01 window.":
        "O grafo do autoencoder precisa de uma janela individual do meeting01.",
    "load_ae_trace failed: {exc}": "load_ae_trace falhou: {exc}",
    "input": "entrada",
    "latent": "código latente",
    "reconstruction": "reconstrução",
    "  — LIF params are constructed defaults (pre-fix checkpoint)":
        "  — os parâmetros LIF são valores padrão (checkpoint anterior à correção)",
    "{arch} / {enc} · {shape} neurons — press ▶ to send the signal through{note}{picked}":
        "{arch} / {enc} · {shape} neurônios — aperte ▶ para o sinal percorrer a rede{note}{picked}",
    "model": "modelo",
    "Run this model": "Rodar este modelo",
    "{role} · {arch}/{enc} · v_th={vth} α={a} · run {run}":
        "{role} · {arch}/{enc} · v_th={vth} α={a} · execução {run}",
    "Pick a model from the list first.": "Escolha um modelo na lista primeiro.",
    "  — explicitly selected": "  — selecionado manualmente",
    "Encoder · ": "Encoder · ",
    "Decoder · ": "Decoder · ",
    "{half}{name} — neuron {i} of {n}: activation {a}":
        "{half}{name} — neurônio {i} de {n}: ativação {a}",
    "fired ({g} over the line)": "disparou ({g} acima da linha)",
    "did not fire ({g} below the line)": "não disparou ({g} abaixo da linha)",
    "  ·  membrane {vm}, firing line {vth} — {verdict}":
        "  ·  carga {vm}, linha de disparo {vth} — {verdict}",

    # ---- Model Structure --------------------------------
    "Model Structure": "Estrutura do modelo",
    "\n"
    "<b>What this panel shows.</b> The selected model's whole <b>structure</b> — every\n"
    "layer, its shape, how many trainable weights it has, and (for a spiking layer)\n"
    "its trained firing threshold. This is a property of the <b>checkpoint</b>, not of\n"
    "any particular window: it never changes when you browse to a different sample.\n"
    "<br><br>\n"
    "<b>Structure vs. activation</b> — confusable pair: this tab is the blueprint (the\n"
    "same for every window); the \"Autoencoder\" tab is the blueprint <i>at work</i>\n"
    "(activations differ per window). Use this tab to answer \"how big / how deep is\n"
    "this network\", the other to answer \"what did it do with this signal\".\n"
    "<br><br>\n"
    "It follows whichever model is loaded in the \"Autoencoder\" tab — browse a window\n"
    "there, or pick a specific trained model and press <i>Run this model</i>, and this\n"
    "table updates the same way.\n":
        "\n"
        "<b>O que este painel mostra.</b> A <b>estrutura</b> inteira do modelo\n"
        "selecionado — cada camada, seu formato, quantos pesos treináveis ela tem e,\n"
        "para uma camada de disparos, seu limiar treinado. Isso é uma propriedade do\n"
        "<b>checkpoint</b>, não de uma janela específica: nunca muda quando você navega\n"
        "para outra amostra.\n"
        "<br><br>\n"
        "<b>Estrutura × ativação</b> — par que se confunde: esta aba é a planta (igual\n"
        "para toda janela); a aba \"Autoencoder\" é a planta <i>em funcionamento</i>\n"
        "(as ativações mudam a cada janela). Use esta aba para responder \"quão grande\n"
        "/ quão profunda é esta rede\"; a outra, para \"o que ela fez com este sinal\".\n"
        "<br><br>\n"
        "Ela acompanha o modelo carregado na aba \"Autoencoder\" — navegue até uma\n"
        "janela lá, ou escolha um modelo treinado específico e aperte <i>Rodar este\n"
        "modelo</i>, e esta tabela é atualizada da mesma forma.\n",
    "Select a meeting01 window (Autoencoder tab) to load a model.":
        "Selecione uma janela do meeting01 (aba Autoencoder) para carregar um modelo.",
    "half": "metade", "layer": "camada", "shape (in → out)": "formato (entrada → saída)",
    "trainable weights": "pesos treináveis", "extra": "extra",
    "Encoder": "Encoder", "Decoder": "Decoder",
    "v_th = {v}": "v_th = {v}",
    "{arch}/{enc} · dataset {ds} fold {fold} · {role} run {run}\n"
    "{neurons} neurons · {params} trainable weights (Linear only — bias not exposed by the binding)":
        "{arch}/{enc} · conjunto {ds} dobra {fold} · {role} execução {run}\n"
        "{neurons} neurônios · {params} pesos treináveis (só Linear — o binding não expõe o bias)",

    # ---- Matriz de Atributos ---------------------------
    "per-column z-score (display only)": "z-score por coluna (só na exibição)",
    "No live feature matrix for this object.":
        "Não há matriz de atributos ao vivo para este objeto.",
    "load_features failed: {exc}": "load_features falhou: {exc}",
    "{label}  [{origin}] — {r} samples x {c} features":
        "{label}  [{origin}] — {r} amostras × {c} atributos",
    "{name} @ {sample}  =  {val}  (class {cls})  [{origin}]":
        "{name} @ {sample}  =  {val}  (classe {cls})  [{origin}]",

    # ---- docks: inspetor / reproduzir / marcadores / busca ---
    "Field": "Campo", "Value": "Valor", "Origin": "Origem",
    "Add current…": "Adicionar o atual…", "Remove": "Remover",
    "Add bookmark": "Adicionar marcador", "Name:": "Nome:",
    "Select a run.": "Selecione uma execução.",
    "Configuration": "Configuração",
    "Copy result-file paths": "Copiar os caminhos dos arquivos de resultado",
    "Reproduction command": "Comando para reproduzir",
    "not executed by this app (§31)": "este app não executa (§31)",
    "Copy command": "Copiar comando",
    "Copied ✓": "Copiado ✓",
    "Select a run / fold node to see its reproduction recipe.":
        "Selecione um nó de execução ou de dobra para ver a receita de reprodução.",
    "(no persisted result files)": "(nenhum arquivo de resultado salvo)",
    'search — e.g. "daub10 lfcc eeg" or "fold 0 fsdd"':
        'buscar — ex.: "daub10 lfcc eeg" ou "fold 0 fsdd"',
    "{shown} of {total} match(es)": "{shown} de {total} resultado(s)",
    " — refine to see more": " — refine a busca para ver mais",

    # ---- linha do tempo do experimento ----------------
    "No meeting01 event stream yet.": "Ainda não há fluxo de eventos do meeting01.",
    "node": "nó", "detail": "detalhe", "epoch": "época", "loss": "perda",
    "train": "treino", "val": "validação",
    "epoch duration (s)": "duração da época (s)",
    "comparing {n} configs — solid = val loss, dotted = train loss":
        "comparando {n} configurações — traço contínuo = perda de validação, pontilhado = perda de treino",
    "event stream unreadable: {exc}": "não foi possível ler o fluxo de eventos: {exc}",
    "No meeting01 event stream under results/meeting01/ (*_events.jsonl). Run a LOSO fold to populate this.":
        "Não há fluxo de eventos do meeting01 em results/meeting01/ (*_events.jsonl). Rode uma dobra LOSO para preencher isto.",
    "session — git {git}, seed {seed}, {nc} config(s), {nf} fold(s)":
        "sessão — git {git}, seed {seed}, {nc} config(s), {nf} dobra(s)",
    "{n} config(s)": "{n} config(s)",
    "  ·  best val {v}": "  ·  melhor validação {v}",
    " @ epoch {e}": " @ época {e}",
    "train {tr}   val {val}": "treino {tr}   validação {val}",
    "epoch {e}": "época {e}",
    "{cfg} — no epoch data": "{cfg} — sem dados de época",
    "  ·  lr {lr}": "  ·  lr {lr}",
    "  ·  ~{s}s/epoch": "  ·  ~{s}s/época",
    "Select a meeting01 or thesis object.":
        "Selecione um objeto do meeting01 ou da tese.",

    # ---- inspetor de arquivo bruto ------------------------
    "File": "Arquivo", "Path": "Caminho", "Format": "Formato", "Size": "Tamanho",
    "unknown": "desconhecido", "bytes": "bytes",
}
