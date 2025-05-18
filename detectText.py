import cv2
import mediapipe as mp
import threading
import math
import array  # For safe audio buffer creation
import sounddevice as sd
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

class HandTracker:
    def __init__(self):
        # Initialize MediaPipe for detecting and tracking hands
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.drawing_utils = mp.solutions.drawing_utils

    def detect_hands(self, frame):
        """
        Detect hand landmarks in a given BGR frame.
        Returns the detection results.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results

class AudioGeneration:
    def __init__(self):
        self.fs = 16000  # Sample rate
        self.freq = 440  # Frequency of sine wave
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
        duration = 0.1  # chunk duration in seconds
        chunk_size = int(self.fs * duration)
        t = np.arange(chunk_size) / self.fs

        def callback(outdata, frames, time, status):
            with self.lock:
                if not self.running:
                    raise sd.CallbackStop()

                # Calculate the samples with phase continuity
                nonlocal t
                samples = np.sin(2 * np.pi * self.freq * t + self.phase).astype(np.float32) * 0.5
                outdata[:] = samples.reshape(-1, 1)

                # Update phase for continuity
                self.phase += 2 * np.pi * self.freq * frames / self.fs
                self.phase = self.phase % (2 * np.pi)

                # Update time array to continue waveform properly
                t += frames / self.fs
                t = t % duration

        with sd.OutputStream(channels=1, callback=callback, samplerate=self.fs, dtype='float32'):
            while True:
                with self.lock:
                    if not self.running:
                        break
                sd.sleep(100)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Tracker with Audio")

        self.cap = cv2.VideoCapture(0)
        self.hand_tracker = HandTracker()
        self.audio_gen = AudioGeneration()

        # Set up GUI canvas for displaying video
        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.pack()

        self.update_video()  # Start video capture loop

    def update_video(self):
        """Capture frame, process hand tracking, and update GUI."""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Mirror the video for natural interaction
            results = self.hand_tracker.detect_hands(frame)

            if results.multi_hand_landmarks:
                self.audio_gen.start_audio()
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks and connections on the frame
                    self.hand_tracker.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )
            else:
                self.audio_gen.stop_audio()

            # Convert OpenCV image to a format compatible with Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(img))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
            self.canvas.imgtk = img

        # Schedule the next frame update after 10 ms
        self.root.after(10, self.update_video)

    def on_close(self):
        """Release resources on window close."""
        self.audio_gen.stop_audio()
        self.cap.release()
        self.root.destroy()

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
