import cv2
import mediapipe as mp
import threading
import numpy as np
import sounddevice as sd
import tkinter as tk
from PIL import Image, ImageTk

class HandTracker:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.drawing_utils = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results

class AudioGeneration:
    def __init__(self):
        self.fs = 44100
        self.amplitude = 0.8
        self.freq = 440
        self.running = False
        self.phase = 0.0
        self.lock = threading.Lock()

    def start_audio(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.audioGen, daemon=True).start()

    def stop_audio(self):
        with self.lock:
            self.running = False

    def audioGen(self):
        def callback(outdata, frames, time, status):
            with self.lock:
                if not self.running:
                    raise sd.CallbackStop()

                t = (np.arange(frames) + self.phase_offset) / self.fs
                samples = self.amplitude * np.sin(2 * np.pi * self.freq * t).astype(np.float32)

                outdata[:, 0] = samples

                self.phase_offset += frames
                self.phase_offset %= self.fs
        self.phase_offset = 0
        with sd.OutputStream(channels=1, callback=callback, samplerate=self.fs, dtype='float32'):
            while True:
                with self.lock:
                    if not self.running:
                        break
                sd.sleep(100)


    def set_frequency(self, freq):
        with self.lock:
            self.freq = freq

    def set_amplitude(self,amplitude):
        with self.lock:
            self.amplitude=amplitude


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracker with Audio")

        self.cap = cv2.VideoCapture(0)
        self.hand_tracker = HandTracker()
        self.audio_gen = AudioGeneration()

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.audio_gen.start_audio()
        self.update_video()

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            results = self.hand_tracker.detect_hands(frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.hand_tracker.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )

                    #palm_y : amplitude
                    #palm_x : frequency

                    palm_pos = hand_landmarks.landmark[0]
                    normalized_x = 1.0 - palm_pos.x
                    freq = 220 + (880 - 220) * normalized_x

                    self.audio_gen.set_frequency(freq)

                    normalized_y = 1.0 - palm_pos.y
                    amplitude = 0.1 + 0.4 * normalized_y
                    self.audio_gen.set_amplitude(amplitude)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.imgtk = img

        self.root.after(10, self.update_video)

    def on_close(self):
        self.audio_gen.stop_audio()
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
