# SynesthAI

> Uma câmera sinestésica de **corpo inteiro** em tempo real. O que ela vê, ela toca.

SynesthAI detecta **33 pontos do seu corpo** com MediaPipe Pose e transforma cada membro num controle musical: pulsos viram theremin, joelhos viram percussão, inclinação da cabeça vira pitch bend, e seu sorriso muda o acorde de fundo. Tudo isso com um esqueleto colorido animado por cima, partículas, trilhas e uma onda de áudio ao vivo.

Roda em **Python 3.14** usando MediaPipe, OpenCV, NumPy e sounddevice.

---

## Mapeamento corpo → som

| Parte do corpo | Vira |
|---|---|
| **Pulso direito (X)** | Nota do theremin (escala pentatônica) |
| **Pulso direito (Y)** | Volume do theremin |
| **Pulso esquerdo (Y)** | Deslocamento de oitava (alto = +1, baixo = −1) |
| **Distância entre os pulsos** | Profundidade do vibrato |
| **Inclinação da cabeça** | Pitch bend (±8%) |
| **Joelho esquerdo levantado** | Kick (sweep de baixo) + partículas |
| **Joelho direito levantado** | Snare (ruído envelopado) + partículas |
| **Rosto sorrindo** | Drone vira acorde maior em Fá |
| **Rosto sério** | Drone vira acorde menor em Ré |
| **Cor dominante do ambiente** | Voz harmônica curta com vibrato |

Tudo simultâneo, em tempo real, no mesmo frame.

---

## Instalação

```bash
git clone https://github.com/welerson-py/synesth.git
cd synesth
python -m pip install -r requirements.txt
python synesthAI.py
```

> No Windows, se `pip` não estiver no PATH, use `python -m pip` no lugar de `pip`.

Na primeira execução, o programa baixa automaticamente o modelo `pose_landmarker_lite.task` (~6 MB) do Google.

---

## Como rodar

```bash
python synesthAI.py
```

Posicione-se de pé na frente da câmera de forma que **cabeça, tronco, mãos e joelhos** apareçam no quadro (uns 2 metros de distância funciona bem). Você vai ver o esqueleto colorido te seguindo.

### Atalhos

| Tecla | Efeito |
|-------|--------|
| `Q` | Sai |
| `K` | Caleidoscópio (espelha 4 quadrantes) |
| `M` | Muta o áudio |
| `ESPAÇO` | Salva print PNG |

---

## Cores do esqueleto

| Membro | Cor |
|---|---|
| Tronco / quadris | branco |
| Braço esquerdo | verde |
| Braço direito | magenta |
| Perna esquerda | ciano |
| Perna direita | amarelo |
| Rosto | lilás claro |

Junta dos pulsos é desenhada maior — são os pontos que controlam o theremin.

---

## Como funciona por dentro

### Detecção corporal

[**MediaPipe PoseLandmarker**](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) (modelo `pose_landmarker_lite`) — uma CNN treinada que devolve 33 landmarks por pessoa, com nível de visibilidade por ponto. O programa só usa pontos com `visibility > 0.4` pra evitar falsos.

### Áudio

Síntese aditiva em tempo real via callback do `sounddevice` a 44.1 kHz, blocos de 512 amostras (~11.6 ms de latência):

- **Theremin**: fundamental + 2º e 3º harmônicos, com vibrato controlado por `vibrato_amount` (distância entre os pulsos) e pitch bend pela inclinação da cabeça.
- **Drone**: três senóides em proporções de acorde (maior ou menor conforme você sorri).
- **Kick**: sweep exponencial de 95 → 35 Hz com envelope rápido.
- **Snare**: ruído branco envelopado.
- **Vozes de cor**: senóides curtas com vibrato.
- Soft clip via `tanh` no final.

### Triggers de percussão

Joelhos são analisados por frame: se `knee.y < hip.y − 25 px` num frame e não estava no frame anterior, **dispara**. Esse design evita repetição enquanto o joelho fica levantado — você precisa abaixar e levantar de novo pra disparar de novo. Pisar no chão ou marchar funciona ótimo.

---

## Dicas

- **Câmera afastada** o suficiente pra pegar você inteiro, da cabeça aos pés (a detecção de joelhos precisa ver os joelhos!).
- **Iluminação razoável** ajuda o detector, mas MediaPipe é tolerante.
- Sorria largo pra ouvir o acorde virar maior.
- Marche no lugar pra acionar kick/snare alternados.
- Levante o braço esquerdo bem alto pra subir uma oitava enquanto toca theremin com o direito.
- Abra os braços bem aberto pra ouvir o vibrato aparecer.

---

## Estrutura

```
synesth/
├── synesthAI.py                # programa principal (audio + pose + visuais)
├── requirements.txt            # dependencias
├── pose_landmarker_lite.task   # baixado automaticamente, ignorado pelo git
├── .gitignore
└── README.md
```

---

## Licença

MIT. Faça o que quiser — só não me culpe se o gato sair correndo do kick.
