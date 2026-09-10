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
}
