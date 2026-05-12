"""
SynesthAI - A camera sinestesica de CORPO INTEIRO em tempo real.

Detecta 33 pontos do corpo (MediaPipe PoseLandmarker) e os mapeia em som:

  - Pulso direito   -> theremin (X = nota, Y = volume)
  - Pulso esquerdo  -> deslocamento de oitava (alto = +1 oitava, baixo = -1)
  - Joelho esquerdo levantado -> KICK
  - Joelho direito  levantado -> SNARE
  - Distancia entre pulsos    -> vibrato / "shimmer"
  - Inclinacao da cabeca      -> pitch bend
  - Rosto sorrindo  -> drone vira acorde maior; serio -> menor
  - Cor dominante do ambiente -> voz harmonica curta

Esqueleto colorido (cada membro com cor propria), particulas, trilhas,
onda de audio ao vivo e HUD por cima da imagem.

Controles:
  Q        -> sair
  K        -> caleidoscopio
  M        -> mute
  ESPACO   -> print PNG
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
    def __init__(self):
        self.phase_theremin = 0.0
        self.target_freq = 0.0
        self.current_freq = 0.0
        self.target_vol = 0.0
        self.last_vol = 0.0

        self.drone_phases = [0.0, 0.0, 0.0]
        self.drone_root = 146.83
        self.drone_intervals = (1.0, 1.19, 1.5)
        self.drone_vol = 0.0
        self.drone_target = 0.0

        self.color_voices = []

        self.kick_env = 0.0
        self.snare_env = 0.0

        # vibrato controlado pela distancia entre pulsos
        self.vibrato_amount = 0.0  # 0..1

        self.muted = False
        self.stream = None

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
            self.drone_root = 174.61
        else:
            self.drone_intervals = (1.0, 1.1892, 1.4983)
            self.drone_root = 146.83

    def callback(self, outdata, frames, time_info, status):
        out = np.zeros(frames, dtype=np.float32)

        if self.muted:
            outdata[:, 0] = 0
            outdata[:, 1] = 0
            return

        # Theremin com vibrato
        self.current_freq += (self.target_freq - self.current_freq) * 0.25
        if self.target_vol > 0.001 and self.current_freq > 20:
            f = self.current_freq
            t = np.arange(frames) / SAMPLE_RATE
            # vibrato: 6 Hz, profundidade controlada por vibrato_amount
            vib = 1.0 + (0.015 * self.vibrato_amount) * np.sin(2 * math.pi * 6.0 * t)
            phase_inc = 2 * math.pi * f / SAMPLE_RATE
            phases = self.phase_theremin + np.cumsum(phase_inc * vib)
            wave = (
                np.sin(phases) * 0.65
                + np.sin(phases * 2.0) * 0.18
                + np.sin(phases * 3.0) * 0.07
            )
            vol_curve = np.linspace(self.last_vol, self.target_vol, frames)
            out += (wave * vol_curve * 0.32).astype(np.float32)
            self.phase_theremin = phases[-1] % (2 * math.pi)
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

# Indices dos 33 landmarks do MediaPipe Pose
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

POSE_CONNECTIONS = {
    "torso": [(11, 12), (12, 24), (24, 23), (23, 11)],
    "left_arm": [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19)],
    "right_arm": [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)],
    "left_leg": [(23, 25), (25, 27), (27, 29), (27, 31), (29, 31)],
    "right_leg": [(24, 26), (26, 28), (28, 30), (28, 32), (30, 32)],
    "face": [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)],
}

LIMB_COLORS = {
    "torso":     (240, 240, 240),
    "left_arm":  (100, 255, 160),
    "right_arm": (255, 130, 255),
    "left_leg":  (100, 220, 255),
    "right_leg": (255, 220, 110),
    "face":      (200, 200, 255),
}

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def ensure_pose_model():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "pose_landmarker_lite.task")
    if not os.path.exists(path):
        print(f"Baixando modelo de pose (~6 MB)...")
        urllib.request.urlretrieve(POSE_MODEL_URL, path)
        print(f"Modelo salvo em {path}")
    return path


class BodyTracker:
    """PoseLandmarker - 33 pontos do corpo."""

    def __init__(self, model_path):
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
        self.t0 = time.time()

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - self.t0) * 1000)
        result = self.landmarker.detect_for_video(mp_image, ts_ms)
        if not result.pose_landmarks:
            return None
        lm_list = result.pose_landmarks[0]
        pts = []
        for lm in lm_list:
            x = int(lm.x * w)
            y = int(lm.y * h)
            vis = float(getattr(lm, "visibility", 1.0) or 0.0)
            pts.append((x, y, vis))
        return pts

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
    return (x, y, w, h), ("feliz" if len(smiles) > 0 else "calmo")


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


def draw_skeleton(img, pts, min_vis=0.3):
    """Desenha o esqueleto, com cada membro em cor propria."""
    if pts is None:
        return
    for limb, conns in POSE_CONNECTIONS.items():
        color = LIMB_COLORS[limb]
        for a, b in conns:
            if a >= len(pts) or b >= len(pts):
                continue
            pa, pb = pts[a], pts[b]
            if pa[2] < min_vis or pb[2] < min_vis:
                continue
            cv2.line(img, (pa[0], pa[1]), (pb[0], pb[1]), color, 3, cv2.LINE_AA)

    # Juntas principais
    big_joints = [
        (NOSE, (255, 255, 255), 6),
        (LEFT_SHOULDER, LIMB_COLORS["left_arm"], 5),
        (RIGHT_SHOULDER, LIMB_COLORS["right_arm"], 5),
        (LEFT_ELBOW, LIMB_COLORS["left_arm"], 4),
        (RIGHT_ELBOW, LIMB_COLORS["right_arm"], 4),
        (LEFT_WRIST, LIMB_COLORS["left_arm"], 8),
        (RIGHT_WRIST, LIMB_COLORS["right_arm"], 8),
        (LEFT_HIP, (240, 240, 240), 5),
        (RIGHT_HIP, (240, 240, 240), 5),
        (LEFT_KNEE, LIMB_COLORS["left_leg"], 6),
        (RIGHT_KNEE, LIMB_COLORS["right_leg"], 6),
        (LEFT_ANKLE, LIMB_COLORS["left_leg"], 5),
        (RIGHT_ANKLE, LIMB_COLORS["right_leg"], 5),
    ]
    for idx, color, radius in big_joints:
        if idx < len(pts) and pts[idx][2] >= min_vis:
            cv2.circle(img, (pts[idx][0], pts[idx][1]), radius, color, -1, cv2.LINE_AA)


def draw_hud(img, info):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 25), -1)
    cv2.rectangle(overlay, (0, h - 38), (w, h), (15, 15, 25), -1)
    img[:] = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)

    cv2.putText(img, "SynesthAI", (15, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "corpo inteiro - sinestesia",
                (185, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 230), 1, cv2.LINE_AA)
    cv2.putText(img, f"emocao: {info['emotion']}",
                (15, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(img, f"nota: {info['note']:>4.0f} Hz",
                (220, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 110), 1, cv2.LINE_AA)
    cv2.putText(img, f"oitava: {info['octave']:+d}",
                (400, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"cor: {info['color']}",
                (560, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 220), 1, cv2.LINE_AA)
    cv2.putText(img, info["hint"],
                (730, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 200, 255), 1, cv2.LINE_AA)


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
        model_path = ensure_pose_model()
    except Exception as e:
        print(f"ERRO: nao consegui obter o modelo de pose: {e}")
        print(f"Baixe manualmente: {POSE_MODEL_URL}")
        print(f"Salve em: {os.path.dirname(os.path.abspath(__file__))}\\pose_landmarker_lite.task")
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
        print(f"Aviso: audio nao iniciou ({e}).")

    body = BodyTracker(model_path)
    particles = ParticleSystem()

    # trilhas dos end-effectors
    trail_right_wrist = Trail((100, 255, 160))
    trail_left_wrist = Trail((255, 130, 255))
    trail_left_ankle = Trail((100, 220, 255), max_len=18)
    trail_right_ankle = Trail((255, 220, 110), max_len=18)

    smooth_freq = 0.0
    smooth_vol = 0.0
    last_color_emit = 0.0
    last_dom_hue = -999.0
    kaleidoscope = False

    # Estado pra detectar joelhos levantados (kick / snare)
    prev_left_knee_up = False
    prev_right_knee_up = False

    fps_t0 = time.time()
    fps_counter = 0
    fps = 0.0

    print("SynesthAI corpo inteiro rodando.")
    print("Q sai | K caleidoscopio | M mute | ESPACO printa.")

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

            # --- rosto / emocao ---
            face_box, emotion = detect_face_emotion(gray)
            if face_box is not None:
                audio.drone_target = 0.4
                audio.set_mode(emotion == "feliz")
                fx, fy, fw_, fh_ = face_box
                halo = (110, 220, 255) if emotion == "feliz" else (220, 160, 255)
                overlay = frame.copy()
                cv2.circle(overlay, (fx + fw_ // 2, fy + fh_ // 2),
                           int(fw_ * 0.65), halo, -1)
                frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)
            else:
                audio.drone_target *= 0.92

            # --- pose ---
            pts = body.detect(frame)

            octave_shift = 0  # -1, 0, +1
            note_hz_hud = 0.0

            if pts is not None:
                rw = pts[RIGHT_WRIST]
                lw = pts[LEFT_WRIST]
                nose = pts[NOSE]
                ls = pts[LEFT_SHOULDER]
                rs = pts[RIGHT_SHOULDER]
                lh = pts[LEFT_HIP]
                rh = pts[RIGHT_HIP]
                lk = pts[LEFT_KNEE]
                rk = pts[RIGHT_KNEE]
                la = pts[LEFT_ANKLE]
                ra = pts[RIGHT_ANKLE]

                # ---- THEREMIN: pulso direito ----
                if rw[2] > 0.4:
                    rx, ry = rw[0], rw[1]
                    idx = int((rx / w) * (len(PENT_FREQS) - 1))
                    target_freq = PENT_FREQS[idx]

                    # oitava controlada pelo pulso esquerdo (Y normalizado)
                    if lw[2] > 0.4 and ls[2] > 0.4 and lh[2] > 0.4:
                        # 0 = altura dos ombros (alto), 1 = altura dos quadris (baixo)
                        ref_top = ls[1]
                        ref_bot = lh[1]
                        span = max(20, ref_bot - ref_top)
                        rel = (lw[1] - ref_top) / span
                        if rel < 0.0:
                            octave_shift = +1
                        elif rel > 1.2:
                            octave_shift = -1
                        target_freq *= (2.0 ** octave_shift)

                    # pitch bend pela inclinacao da cabeca (nariz vs centro dos ombros)
                    if nose[2] > 0.4 and ls[2] > 0.4 and rs[2] > 0.4:
                        shoulder_mid_x = (ls[0] + rs[0]) / 2.0
                        shoulder_w = max(20.0, abs(rs[0] - ls[0]))
                        tilt = (nose[0] - shoulder_mid_x) / shoulder_w
                        tilt = max(-0.4, min(0.4, tilt))
                        target_freq *= (1.0 + tilt * 0.08)

                    target_vol = max(0.0, min(1.0, 1.0 - (ry / h))) * 0.85

                    smooth_freq += (target_freq - smooth_freq) * 0.3
                    smooth_vol += (target_vol - smooth_vol) * 0.25
                    audio.target_freq = smooth_freq
                    audio.target_vol = smooth_vol
                    note_hz_hud = smooth_freq

                    trail_right_wrist.add(rx, ry)
                    col = hue_to_bgr(int((rx / w) * 180))
                    cv2.circle(frame, (rx, ry), int(20 * target_vol + 8), col, 2, cv2.LINE_AA)
                else:
                    smooth_vol *= 0.5
                    if smooth_vol < 0.01:
                        smooth_vol = 0.0
                    audio.target_vol = smooth_vol
                    trail_right_wrist.fade()

                # ---- VIBRATO: distancia entre pulsos ----
                if rw[2] > 0.4 and lw[2] > 0.4 and ls[2] > 0.4 and rs[2] > 0.4:
                    dist = math.hypot(rw[0] - lw[0], rw[1] - lw[1])
                    shoulder_w = max(20.0, abs(rs[0] - ls[0]))
                    spread = max(0.0, min(2.0, dist / (shoulder_w + 1e-6)))
                    audio.vibrato_amount = min(1.0, max(0.0, (spread - 0.8) / 1.2))
                    # linha conectando as duas maos com brilho proporcional
                    if spread > 0.5:
                        cv2.line(frame, (rw[0], rw[1]), (lw[0], lw[1]),
                                 (255, 220, 120), 1, cv2.LINE_AA)
                else:
                    audio.vibrato_amount *= 0.9

                # ---- PULSO ESQUERDO -> voz de cor curta quando muda de altura ----
                if lw[2] > 0.4:
                    trail_left_wrist.add(lw[0], lw[1])
                    if now - last_color_emit > 0.5:
                        f = PENT_FREQS[int((lw[0] / w) * (len(PENT_FREQS) - 1))]
                        audio.add_color_voice(f * (2.0 ** octave_shift), amp=0.10, life=0.5)
                        last_color_emit = now
                else:
                    trail_left_wrist.fade()

                # ---- JOELHO ESQUERDO -> KICK ----
                # "levantou" = joelho subiu acima do quadril
                if lk[2] > 0.4 and lh[2] > 0.4:
                    left_knee_up = lk[1] < lh[1] - 25
                    if left_knee_up and not prev_left_knee_up:
                        audio.trigger_kick()
                        particles.emit(lk[0], lk[1], LIMB_COLORS["left_leg"],
                                       count=25, speed=6)
                    prev_left_knee_up = left_knee_up
                else:
                    prev_left_knee_up = False

                # ---- JOELHO DIREITO -> SNARE ----
                if rk[2] > 0.4 and rh[2] > 0.4:
                    right_knee_up = rk[1] < rh[1] - 25
                    if right_knee_up and not prev_right_knee_up:
                        audio.trigger_snare()
                        particles.emit(rk[0], rk[1], LIMB_COLORS["right_leg"],
                                       count=18, speed=5)
                    prev_right_knee_up = right_knee_up
                else:
                    prev_right_knee_up = False

                # trilha dos tornozelos (pra ver o movimento dos pes)
                if la[2] > 0.4:
                    trail_left_ankle.add(la[0], la[1])
                else:
                    trail_left_ankle.fade()
                if ra[2] > 0.4:
                    trail_right_ankle.add(ra[0], ra[1])
                else:
                    trail_right_ankle.fade()

                draw_skeleton(frame, pts)
            else:
                smooth_vol *= 0.5
                audio.target_vol = smooth_vol
                audio.vibrato_amount *= 0.9
                trail_right_wrist.fade()
                trail_left_wrist.fade()
                trail_left_ankle.fade()
                trail_right_ankle.fade()
                prev_left_knee_up = False
                prev_right_knee_up = False

            # trilhas
            trail_right_wrist.draw(frame)
            trail_left_wrist.draw(frame)
            trail_left_ankle.draw(frame)
            trail_right_ankle.draw(frame)

            # --- cor dominante ---
            dom = dominant_hue(frame)
            if dom is not None:
                hue_changed = abs(dom - last_dom_hue) > 12 or last_dom_hue < -100
                if hue_changed and now - last_color_emit > 0.7:
                    audio.add_color_voice(hue_to_freq(dom), amp=0.10, life=0.7)
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

            pose_status = "OK" if pts is not None else "---"
            draw_hud(frame, {
                "emotion": emotion or "---",
                "note": note_hz_hud if audio.last_vol > 0.01 else 0.0,
                "octave": octave_shift,
                "color": f"{int(dom)}" if dom is not None else "---",
                "hint": f"pose:{pose_status} FPS {fps:4.1f} {'MUDO' if audio.muted else ''} {'KALEIDO' if kaleidoscope else ''}",
            })

            cv2.imshow("SynesthAI", frame)

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
        body.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
