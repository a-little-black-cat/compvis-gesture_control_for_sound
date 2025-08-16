import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading
import math
from itertools import product
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
import io


class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawing_utils = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def check_gestures(self, results, frame):
        saw_left = False
        saw_right = False
        freq = 0
        amplitude = 0
        room_size = 0

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label

                if handedness == "Left":
                    saw_left = True
                    Lpalm_pos = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                    freq = 220 + (880 - 220) * (1.0 - Lpalm_pos.x)
                    amplitude = 0.1 + 0.4 * (1.0 - Lpalm_pos.y)

                if handedness == "Right":
                    saw_right = True
                    thumbTip_pos = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                    indexTip_pos = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

                    tipDistance_reverb = math.hypot(
                        thumbTip_pos.x - indexTip_pos.x, thumbTip_pos.y - indexTip_pos.y
                    )

                    min_dist = 0.03
                    max_dist = 0.4
                    normalized_dist = (tipDistance_reverb - min_dist) / (max_dist - min_dist)
                    normalized_dist = min(max(normalized_dist, 0.0), 1.0)
                    room_size = 0.1 + normalized_dist * 0.9

        return saw_left, saw_right, freq, amplitude, room_size

    def draw_landmarks(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )


class AudioGeneration:
    def __init__(self, app_instance=None):
        self.fs = 16000
        self.freq = 440.0
        self.amplitude = 0.2
        self.running = False
        self.lock = threading.Lock()
        self.phase_offset = 0
        self.roomSize = 0.3
        self.app_instance = app_instance
        self.fig = None
        self.ax = None
        self._setup_plot()

    def _setup_plot(self):
        self.fig = plt.Figure(figsize=(3, 3), dpi=100, facecolor='None')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_title("Reverb Room Size")
        self.ax.set_axis_off()
        self.ax.set_facecolor((0.0, 0.0, 0.0, 0.0))

    def start_audio(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.audioGen, daemon=True).start()

    def stop_audio(self):
        with self.lock:
            self.running = False

    def set_parameters(self, freq, amplitude, roomSize):
        with self.lock:
            alpha = 0.2
            self.freq = self.freq * (1 - alpha) + freq * alpha
            self.amplitude = self.amplitude * (1 - alpha) + amplitude * alpha
            self.roomSize = self.roomSize * (1 - alpha) + roomSize * alpha

    def apply_reverb(self, signal, reverb_time, decay_factor=0.6, num_echoes=5):
        reverb_signal = np.copy(signal)
        delay_samples = int((reverb_time / num_echoes) * self.fs)
        for i in range(1, num_echoes + 1):
            decay = decay_factor ** i
            echo = np.zeros_like(signal)
            if delay_samples * i < len(signal):
                echo[delay_samples * i:] = signal[:-delay_samples * i] * decay
                reverb_signal += echo
        return np.clip(reverb_signal, -1.0, 1.0)

    def audioGen(self):
        def callback(outdata, frames, time, status):
            with self.lock:
                if not self.running:
                    raise sd.CallbackStop()
                t = (np.arange(frames) + self.phase_offset) / self.fs
                samples = self.amplitude * np.sin(2 * np.pi * self.freq * t).astype(np.float32)
                samples = self.apply_reverb(samples, self.roomSize)
                outdata[:, 0] = samples
                self.phase_offset += frames
                self.phase_offset %= self.fs

        try:
            with sd.OutputStream(channels=1, callback=callback, samplerate=self.fs, dtype='float32', blocksize=1024):
                while True:
                    with self.lock:
                        if not self.running:
                            break
                    sd.sleep(100)
        except Exception as e:
            print(f"Audio stream error: {e}")
            with self.lock:
                self.running = False

    def visualizeReverb(self, roomSize):
        self.ax.clear()
        r_val = roomSize / 2
        r = [-r_val, r_val]
        vertices = np.array(list(product(r, r, r)))

        faces = [
            [vertices[0], vertices[1], vertices[3], vertices[2]],
            [vertices[4], vertices[5], vertices[7], vertices[6]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[2], vertices[6], vertices[4]],
            [vertices[1], vertices[3], vertices[7], vertices[5]]
        ]

        self.ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.2))
        self.ax.set_xlim([-r_val, r_val])
        self.ax.set_ylim([-r_val, r_val])
        self.ax.set_zlim([-r_val, r_val])

        elev_val = 50 * time.time()
        roll_val = 20 * time.time()
        self.ax.view_init(elev=elev_val, azim=45, roll=roll_val)
        self.ax.set_title(f"Reverb: {roomSize:.2f}")

        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', transparent=True)
        buf.seek(0)

        pil_image = Image.open(buf)
        return pil_image.convert('RGBA')