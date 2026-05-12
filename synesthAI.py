"""
SynesthAI - A camera sinestesica em tempo real.

Transforma o que a camera ve em som, cor e movimento:
  - Maos viram um theremin pentatonico (X = nota, Y = volume), com
    deteccao por CNN (MediaPipe HandLandmarker).
  - Cores dominantes do ambiente disparam vozes harmonicas.
  - Rosto sorrindo muda o drone de fundo de menor para maior.
  - Movimento corporal (fora de maos e rosto) vira percussao.
  - Esqueleto da mao, particulas, trilhas e onda de audio na tela.

Controles:
  Q        -> sair
  K        -> liga/desliga modo caleidoscopio
  M        -> muta o audio
  ESPACO   -> salva print PNG
"""

import math
import os
import sys
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
import sounddevice as sd

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ===================== AUDIO =====================

SAMPLE_RATE = 44100
BLOCK_SIZE = 512


class AudioEngine:
    """Sintetizador em tempo real via callback do sounddevice."""

    def __init__(self):
        # Theremin
        self.phase_theremin = 0.0
        self.target_freq = 0.0
        self.current_freq = 0.0
        self.target_vol = 0.0
        self.last_vol = 0.0

        # Drone harmonico
        self.drone_phases = [0.0, 0.0, 0.0]
        self.drone_root = 146.83  # D3
        self.drone_intervals = (1.0, 1.19, 1.5)  # menor
        self.drone_vol = 0.0
        self.drone_target = 0.0

        # Vozes de cor (notas curtas tocadas pela cor dominante)
        self.color_voices = []  # [freq, phase, t_left, amp, total]

        # Percussao
        self.kick_env = 0.0
        self.snare_env = 0.0

        # Mute
        self.muted = False
        self.stream = None

    # ----- API -----
    def trigger_kick(self):
        self.kick_env = 1.0

    def trigger_snare(self):
        self.snare_env = 0.8

    def add_color_voice(self, freq, amp=0.12, life=0.9):
        if len(self.color_voices) > 10:
            self.color_voices.pop(0)
        self.color_voices.append([freq, 0.0, life, amp, life])

    def set_mode(self, happy):
        if happy:
            self.drone_intervals = (1.0, 1.2599, 1.4983)
            self.drone_root = 174.61  # F3 maior
        else:
            self.drone_intervals = (1.0, 1.1892, 1.4983)
            self.drone_root = 146.83  # D3 menor

    # ----- callback -----
    def callback(self, outdata, frames, time_info, status):
        out = np.zeros(frames, dtype=np.float32)

        if self.muted:
            outdata[:, 0] = 0
            outdata[:, 1] = 0
            return

        # Theremin
        self.current_freq += (self.target_freq - self.current_freq) * 0.25
        if self.target_vol > 0.001 and self.current_freq > 20:
            f = self.current_freq
            phase_inc = 2 * math.pi * f / SAMPLE_RATE
            phases = self.phase_theremin + phase_inc * np.arange(frames)
            wave = (
                np.sin(phases) * 0.65
                + np.sin(phases * 2.0) * 0.18
                + np.sin(phases * 3.0) * 0.07
            )
            vol_curve = np.linspace(self.last_vol, self.target_vol, frames)
            out += (wave * vol_curve * 0.32).astype(np.float32)
            self.phase_theremin = (self.phase_theremin + phase_inc * frames) % (2 * math.pi)
            self.last_vol = self.target_vol
        else:
            fade = np.linspace(self.last_vol, 0, frames)
            if self.last_vol > 0.001 and self.current_freq > 20:
                phase_inc = 2 * math.pi * self.current_freq / SAMPLE_RATE
                phases = self.phase_theremin + phase_inc * np.arange(frames)
                out += (np.sin(phases) * fade * 0.32).astype(np.float32)
                self.phase_theremin = (self.phase_theremin + phase_inc * frames) % (2 * math.pi)
            self.last_vol = 0.0

        # Drone
        self.drone_vol += (self.drone_target - self.drone_vol) * 0.05
        if self.drone_vol > 0.001:
            for i, ratio in enumerate(self.drone_intervals):
                f = self.drone_root * ratio
                phase_inc = 2 * math.pi * f / SAMPLE_RATE
                phases = self.drone_phases[i] + phase_inc * np.arange(frames)
                out += (np.sin(phases) * self.drone_vol * 0.07).astype(np.float32)
                self.drone_phases[i] = (self.drone_phases[i] + phase_inc * frames) % (2 * math.pi)

        # Vozes de cor
        new = []
        for v in self.color_voices:
            f, phase, t_left, amp, total = v
            phase_inc = 2 * math.pi * f / SAMPLE_RATE
            phases = phase + phase_inc * np.arange(frames)
            env_start = max(0.0, t_left / total)
            t_after = t_left - frames / SAMPLE_RATE
            env_end = max(0.0, t_after / total)
            env_curve = np.linspace(env_start, env_end, frames)
            vibrato = 1 + 0.005 * np.sin(2 * math.pi * 5.5 * np.arange(frames) / SAMPLE_RATE)
            out += (np.sin(phases * vibrato) * amp * env_curve * 0.55).astype(np.float32)
            if t_after > 0:
                new.append([f, (phases[-1] + phase_inc) % (2 * math.pi), t_after, amp, total])
        self.color_voices = new

        # Kick
        if self.kick_env > 0.001:
            t = np.arange(frames) / SAMPLE_RATE
            env_curve = np.maximum(0, self.kick_env - t * 5.0)
            freq_curve = 35 + 60 * env_curve
            phase = np.cumsum(2 * math.pi * freq_curve / SAMPLE_RATE)
            out += (np.sin(phase) * env_curve * 0.7).astype(np.float32)
            self.kick_env = max(0.0, self.kick_env - frames / SAMPLE_RATE * 5.0)

        # Snare
        if self.snare_env > 0.001:
            noise = np.random.uniform(-1, 1, frames).astype(np.float32)
            t = np.arange(frames) / SAMPLE_RATE
            env_curve = np.maximum(0, self.snare_env - t * 8.0)
            out += noise * env_curve * 0.3
            self.snare_env = max(0.0, self.snare_env - frames / SAMPLE_RATE * 8.0)

        out = np.tanh(out * 1.1) * 0.8
        outdata[:, 0] = out
        outdata[:, 1] = out

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=2,
            callback=self.callback,
            blocksize=BLOCK_SIZE,
            dtype="float32",
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass


# ===================== VISAO =====================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

PENT_FREQS = [
    220.00, 246.94, 277.18, 329.63, 369.99,
    440.00, 493.88, 554.37, 659.25, 739.99,
    880.00, 987.77, 1108.73, 1318.51,
]

# Conexoes do esqueleto da mao (21 landmarks no modelo do MediaPipe)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indicador
    (5, 9), (9, 10), (10, 11), (11, 12),     # medio
    (9, 13), (13, 14), (14, 15), (15, 16),   # anelar
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # minimo
]


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_hand_model():
    """Garante o arquivo do modelo de maos local; baixa se nao existir."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "hand_landmarker.task")
    if not os.path.exists(path):
        print(f"Baixando modelo de deteccao de maos (~8 MB) de {MODEL_URL} ...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print(f"Modelo salvo em {path}")
    return path


class HandTracker:
    """Detector/rastreador de maos usando MediaPipe HandLandmarker."""

    def __init__(self, model_path):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_hands=2,
            running_mode=mp_vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self.t0 = time.time()

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - self.t0) * 1000)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)
        hands = []
        if not result.hand_landmarks:
            return hands
        for i, lm_list in enumerate(result.hand_landmarks):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            # centroide pela palma (0, 5, 9, 13, 17)
            palm = [0, 5, 9, 13, 17]
            cx = int(sum(pts[k][0] for k in palm) / len(palm))
            cy = int(sum(pts[k][1] for k in palm) / len(palm))
            hands.append({
                "cx": cx, "cy": cy,
                "bbox": bbox,
                "landmarks": pts,
            })
        # ordena esquerda -> direita do frame
        hands.sort(key=lambda h: h["cx"])
        return hands

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass


def detect_face_emotion(gray):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    roi = gray[y : y + h, x : x + w]
    smiles = smile_cascade.detectMultiScale(roi, 1.7, 22)
    emotion = "feliz" if len(smiles) > 0 else "calmo"
    return (x, y, w, h), emotion


def motion_in_area(prev_gray, gray, exclude_rects):
    if prev_gray is None:
        return 0.0, None
    diff = cv2.absdiff(prev_gray, gray)
    _, thr = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    for x, y, w, h in exclude_rects:
        cv2.rectangle(thr, (x, y), (x + w, y + h), 0, -1)
    amount = float(np.sum(thr) / 255.0 / thr.size)
    return amount, thr


def dominant_hue(frame):
    small = cv2.resize(frame, (60, 40))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 70)
    if not mask.any():
        return None
    hues = hsv[:, :, 0][mask]
    hist, edges = np.histogram(hues, bins=12, range=(0, 180))
    dom = int(np.argmax(hist))
    return float((edges[dom] + edges[dom + 1]) / 2.0)


def hue_to_bgr(hue):
    hsv = np.uint8([[[int(hue) % 180, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def hue_to_freq(hue):
    scale = [
        261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88,
        523.25, 587.33, 659.26, 698.46, 783.99,
    ]
    idx = int((hue / 180.0) * len(scale)) % len(scale)
    return scale[idx]


# ===================== VISUAIS =====================


class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, x, y, color, count=10, speed=4):
        for _ in range(count):
            angle = np.random.uniform(0, 2 * math.pi)
            spd = np.random.uniform(0.5, 1.5) * speed
            self.particles.append({
                "x": float(x), "y": float(y),
                "vx": math.cos(angle) * spd,
                "vy": math.sin(angle) * spd - 1.5,
                "life": 1.0,
                "color": color,
                "size": int(np.random.randint(2, 6)),
            })

    def update_and_draw(self, img):
        h, w = img.shape[:2]
        survivors = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.18
            p["life"] -= 0.022
            if p["life"] <= 0:
                continue
            if not (0 <= p["x"] < w and 0 <= p["y"] < h):
                continue
            alpha = p["life"]
            color = tuple(int(c * alpha) for c in p["color"])
            cv2.circle(img, (int(p["x"]), int(p["y"])), p["size"], color, -1, cv2.LINE_AA)
            survivors.append(p)
        self.particles = survivors


class Trail:
    def __init__(self, color, max_len=28):
        self.points = deque(maxlen=max_len)
        self.color = color

    def add(self, x, y):
        self.points.append((int(x), int(y)))

    def fade(self):
        if self.points:
            self.points.popleft()

    def draw(self, img):
        pts = list(self.points)
        n = len(pts)
        for i in range(1, n):
            alpha = i / n
            thickness = max(1, int(9 * alpha))
            color = tuple(int(c * alpha) for c in self.color)
            cv2.line(img, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)


def draw_hand_skeleton(img, landmarks, color):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, landmarks[a], landmarks[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(landmarks):
        radius = 6 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)


def draw_hud(img, info):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 25), -1)
    cv2.rectangle(overlay, (0, h - 38), (w, h), (15, 15, 25), -1)
    img[:] = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)

    cv2.putText(img, "SynesthAI", (15, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "sinestesia em tempo real",
                (185, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 230), 1, cv2.LINE_AA)
    cv2.putText(img, f"emocao: {info['emotion']}",
                (15, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(img, f"nota: {info['note']:>4.0f} Hz",
                (240, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 110), 1, cv2.LINE_AA)
    cv2.putText(img, f"cor: {info['color']}",
                (430, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 220), 1, cv2.LINE_AA)
    cv2.putText(img, info["hint"],
                (610, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 200, 255), 1, cv2.LINE_AA)


def draw_wave_strip(img, freq, vol):
    h, w = img.shape[:2]
    strip_h = 50
    y0 = h - 90
    strip = np.zeros((strip_h, w, 3), dtype=np.uint8)
    if vol > 0.01 and freq > 20:
        xs = np.arange(w)
        ys = (np.sin(xs * (freq / 18000.0) * 2 * math.pi) * 18 * min(vol * 1.5, 1.0) + strip_h // 2).astype(int)
        color = hue_to_bgr(int((freq / 1300.0) * 180) % 180)
        pts = np.column_stack((xs, ys)).reshape(-1, 1, 2)
        cv2.polylines(strip, [pts], False, color, 2, cv2.LINE_AA)
    img[y0 : y0 + strip_h, :] = cv2.addWeighted(img[y0 : y0 + strip_h, :], 0.35, strip, 0.85, 0)


def apply_kaleidoscope(img):
    h, w = img.shape[:2]
    hw = w // 2
    hh = h // 2
    tl = img[:hh, :hw].copy()
    img[:hh, hw : hw * 2] = cv2.flip(tl, 1)
    img[hh : hh * 2, :hw] = cv2.flip(tl, 0)
    img[hh : hh * 2, hw : hw * 2] = cv2.flip(tl, -1)


# ===================== MAIN =====================


def main():
    try:
        model_path = ensure_hand_model()
    except Exception as e:
        print(f"ERRO: nao consegui obter o modelo de maos: {e}")
        print(f"Baixe manualmente: {MODEL_URL}")
        print(f"E salve em: {os.path.dirname(os.path.abspath(__file__))}\\hand_landmarker.task")
        sys.exit(1)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nao encontrei a camera.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    audio = AudioEngine()
    try:
        audio.start()
    except Exception as e:
        print(f"Aviso: nao foi possivel iniciar o audio ({e}).")

    hand_tracker = HandTracker(model_path)
    particles = ParticleSystem()
    trail_right = Trail((100, 255, 160))
    trail_left = Trail((255, 130, 255))

    prev_gray = None
    smooth_freq = 0.0
    smooth_vol = 0.0
    last_color_emit = 0.0
    last_kick = 0.0
    last_snare = 0.0
    last_dom_hue = -999.0
    kaleidoscope = False

    print("SynesthAI rodando. Q sai | K caleidoscopio | M mute | ESPACO printa.")

    fps_t0 = time.time()
    fps_counter = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            now = time.time()

            # --- rosto / emocao
            face_box, emotion = detect_face_emotion(gray)
            if face_box is not None:
                audio.drone_target = 0.35
                audio.set_mode(emotion == "feliz")
                fx, fy, fw_, fh_ = face_box
                color = (110, 220, 255) if emotion == "feliz" else (220, 160, 255)
                overlay = frame.copy()
                cv2.circle(overlay, (fx + fw_ // 2, fy + fh_ // 2),
                           int(fw_ * 0.65), color, -1)
                frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)
                cv2.rectangle(frame, (fx, fy), (fx + fw_, fy + fh_), (255, 255, 255), 1, cv2.LINE_AA)
            else:
                audio.drone_target *= 0.92

            # --- maos via MediaPipe
            hands = hand_tracker.detect(frame)

            if len(hands) >= 1:
                # mao a direita do frame (espelhado -> direita do usuario)
                rh = hands[-1]
                rx, ry = rh["cx"], rh["cy"]
                idx = int((rx / w) * (len(PENT_FREQS) - 1))
                target_freq = PENT_FREQS[idx]
                target_vol = max(0.0, min(1.0, 1.0 - (ry / h))) * 0.85
                smooth_freq += (target_freq - smooth_freq) * 0.3
                smooth_vol += (target_vol - smooth_vol) * 0.25
                audio.target_freq = smooth_freq
                audio.target_vol = smooth_vol
                trail_right.add(rx, ry)
                col = hue_to_bgr(int((rx / w) * 180))
                draw_hand_skeleton(frame, rh["landmarks"], col)
                cv2.circle(frame, (rx, ry), int(18 * target_vol + 6), col, 2, cv2.LINE_AA)

                if len(hands) >= 2:
                    lh = hands[0]
                    lx, ly = lh["cx"], lh["cy"]
                    trail_left.add(lx, ly)
                    draw_hand_skeleton(frame, lh["landmarks"], (255, 130, 255))
                    # mao esquerda toca voz harmonica (com debounce)
                    if now - last_color_emit > 0.4:
                        f = PENT_FREQS[int((lx / w) * (len(PENT_FREQS) - 1))]
                        audio.add_color_voice(f, amp=0.14, life=0.7)
                        last_color_emit = now
                else:
                    trail_left.fade()
            else:
                smooth_vol *= 0.55
                if smooth_vol < 0.01:
                    smooth_vol = 0.0
                audio.target_vol = smooth_vol
                trail_right.fade()
                trail_left.fade()

            trail_right.draw(frame)
            trail_left.draw(frame)

            # --- movimento (excluindo rosto e maos)
            exclude_rects = []
            if face_box is not None:
                fx, fy, fw_, fh_ = face_box
                exclude_rects.append((max(0, fx - 25), max(0, fy - 25), fw_ + 50, fh_ + 50))
            for hnd in hands:
                bx, by, bw, bh = hnd["bbox"]
                pad = 30
                exclude_rects.append((max(0, bx - pad), max(0, by - pad), bw + 2 * pad, bh + 2 * pad))

            mot, mot_mask = motion_in_area(prev_gray, gray, exclude_rects)
            if mot > 0.05 and now - last_kick > 0.35:
                audio.trigger_kick()
                last_kick = now
                if mot_mask is not None:
                    ys, xs = np.where(mot_mask > 0)
                    if len(xs) > 50:
                        sample = np.random.choice(len(xs), size=min(60, len(xs)), replace=False)
                        for idx in sample[:6]:
                            px, py = int(xs[idx]), int(ys[idx])
                            particles.emit(px, py, (120, 220, 255), count=3, speed=4)
            elif mot > 0.025 and now - last_snare > 0.3:
                audio.trigger_snare()
                last_snare = now

            # --- cor dominante
            dom = dominant_hue(frame)
            if dom is not None:
                hue_changed = abs(dom - last_dom_hue) > 12 or last_dom_hue < -100
                if hue_changed and now - last_color_emit > 0.6:
                    audio.add_color_voice(hue_to_freq(dom), amp=0.12, life=0.85)
                    last_color_emit = now
                    last_dom_hue = dom
                    col = hue_to_bgr(int(dom))
                    particles.emit(w - 30, 75, col, count=4, speed=2)
                col_box = hue_to_bgr(int(dom))
                cv2.rectangle(frame, (w - 30, 70), (w - 10, 90), col_box, -1)

            if kaleidoscope:
                apply_kaleidoscope(frame)

            particles.update_and_draw(frame)
            draw_wave_strip(frame, audio.current_freq, audio.last_vol)

            # FPS
            fps_counter += 1
            if now - fps_t0 >= 1.0:
                fps = fps_counter / (now - fps_t0)
                fps_counter = 0
                fps_t0 = now

            draw_hud(frame, {
                "emotion": emotion or "---",
                "note": audio.current_freq if audio.last_vol > 0.01 else 0.0,
                "color": f"{int(dom)}" if dom is not None else "---",
                "hint": f"FPS {fps:4.1f} maos: {len(hands)} {'MUDO' if audio.muted else ''} {'KALEIDO' if kaleidoscope else ''}",
            })

            cv2.imshow("SynesthAI", frame)
            prev_gray = gray

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("k"):
                kaleidoscope = not kaleidoscope
            elif key == ord("m"):
                audio.muted = not audio.muted
            elif key == 32:
                fn = f"synesth_{int(now)}.png"
                cv2.imwrite(fn, frame)
                print(f"Print salvo: {fn}")
    finally:
        audio.stop()
        hand_tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
