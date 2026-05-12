# SynesthAI

> Uma câmera sinestésica em tempo real. O que ela vê, ela toca.

SynesthAI transforma sua webcam num instrumento multissensorial: suas mãos viram um theremin, as cores ao seu redor disparam vozes harmônicas, seu sorriso muda o acorde de fundo e qualquer movimento do corpo vira percussão. Tudo isso enquanto partículas, trilhas coloridas e uma onda de áudio ao vivo são desenhadas por cima da imagem.

Detecção de mão via **MediaPipe HandLandmarker** (CNN do Google — funciona em qualquer iluminação e tom de pele), rosto/sorriso via Haar cascade do OpenCV, áudio via síntese aditiva em tempo real. Roda em **Python 3.14**.

---

## Demo de funcionamento

| Sentido capturado | Como vira som / imagem |
|---|---|
| Posição das mãos | Theremin pentatônico — X = nota, Y = volume |
| Mão esquerda | Dispara vozes harmônicas curtas com vibrato |
| Cor dominante do ambiente | Toca a nota cromática correspondente |
| Rosto sorrindo | Drone vira acorde **maior em Fá** |
| Rosto sério | Drone vira acorde **menor em Ré** |
| Movimento corporal grande | **Kick** sintetizado (sweep de baixo) + partículas |
| Movimento corporal pequeno | **Snare** sintetizado (ruído envelopado) |

Tudo acontece simultaneamente, no mesmo frame, em tempo real.

---

## Instalação

Requer **Python 3.10+** (testado em 3.14) e uma webcam.

```bash
git clone https://github.com/welerson-py/synesth.git
cd synesth
python -m pip install -r requirements.txt
```

> No Windows, se `pip` não estiver no PATH, use sempre `python -m pip` no lugar de `pip`.

Dependências:

- `opencv-python` — captura, Haar cascades (rosto/sorriso), processamento de imagem
- `numpy` — síntese de áudio e operações vetoriais
- `sounddevice` — saída de áudio em tempo real via callback
- `mediapipe` — detecção de mão por CNN (HandLandmarker)

> Na primeira execução, o programa baixa automaticamente o modelo `hand_landmarker.task` (~8 MB) do Google.

---

## Como rodar

```bash
python synesthAI.py
```

Uma janela "SynesthAI" abre com a câmera. Aperte **Q** pra sair.

### Atalhos durante a execução

| Tecla | Efeito |
|-------|--------|
| `Q` | Sai do programa |
| `K` | Liga/desliga modo **caleidoscópio** (espelha os 4 quadrantes) |
| `M` | Muta o áudio |
| `ESPAÇO` | Salva print PNG na pasta atual |

A detecção de mão funciona out of the box: assim que sua mão aparece no quadro, o esqueleto colorido dos 21 pontos é desenhado e o theremin começa a tocar. Sem calibração.

---

## Como funciona por dentro

### Áudio (`AudioEngine`)

Síntese aditiva em tempo real via callback do `sounddevice` a 44.1 kHz, blocos de 512 amostras (~11.6 ms de latência). Tudo é gerado por amostra, sem samples pré-gravados:

- **Theremin**: senóide fundamental + 2º e 3º harmônicos, com suavização de frequência e amplitude pra eliminar cliques.
- **Drone**: três senóides em proporções de intervalo (maior ou menor) somadas em fase contínua.
- **Vozes de cor**: senóides com vibrato leve (5.5 Hz) e envelope linear.
- **Kick**: sweep de frequência (95 Hz → 35 Hz) com envelope exponencial.
- **Snare**: ruído branco com envelope curto.
- Tudo passa por `tanh` no final pra soft clip.

### Visão

- **Mãos**: MediaPipe **HandLandmarker** (modelo `.task` baixado automaticamente). Detecta até 2 mãos e devolve 21 landmarks por mão. O centróide vem da média dos 5 pontos da palma (`0, 5, 9, 13, 17`).
- **Rosto / emoção**: `haarcascade_frontalface_default.xml` + `haarcascade_smile.xml`.
- **Movimento**: `cv2.absdiff` entre frames consecutivos. Antes de medir, as áreas das mãos e do rosto são **mascaradas** — assim a percussão só dispara em movimento *além* do que está fazendo o theremin.
- **Cor dominante**: histograma de matiz HSV com 12 bins, ignorando pixels muito escuros/dessaturados. Só dispara nota nova quando a cor muda de verdade (>12° de matiz).

### Visuais

- `ParticleSystem`: partículas com gravidade e vida finita, emitidas em explosões de movimento.
- `Trail`: rastros coloridos seguindo cada mão.
- Halo no rosto, onda do theremin desenhada na faixa inferior, HUD com emoção / nota / cor / FPS.
- Caleidoscópio: espelha os 4 quadrantes em tempo real.

---

## Dicas pra ficar incrível

- **Boa iluminação** é essencial — detecção de pele e de rosto dependem disso. De frente pra uma janela ajuda muito.
- Encoste fundos escuros pra dar contraste com as mãos.
- Mova as mãos devagar — o theremin reage à posição absoluta, não ao gesto.
- Sorria largo pra ouvir o acorde virar maior.
- Pinte cores vivas pela cena (camiseta vermelha, objeto azul) pra ver as vozes de cor disparando.
- Aperte `K` depois de já estar tocando: caleidoscópio + theremin + percussão = experiência cheia.

---

## Ajustes

Pra trocar a escala do theremin, edite `PENT_FREQS` no `synesthAI.py` (lista de frequências em Hz). É pentatônica em C por padrão — qualquer posição da mão soa musical.

Pra mudar a sensibilidade da percussão, ajuste os limiares no `main()`:

```python
if mot > 0.05  and now - last_kick  > 0.35: ...   # kick
elif mot > 0.025 and now - last_snare > 0.3:  ...   # snare
```

---

## Estrutura

```
synesth/
├── synesthAI.py       # Programa principal (motor de áudio + visão + visuais)
├── requirements.txt   # Dependências
├── .gitignore
└── README.md
```

---

## Licença

MIT. Faça o que quiser — só não me culpe se o gato sair correndo do kick.
